# -*- coding: utf-8 -*-
"""
Optimized training script for MFENet / GR-ConvNet-style grasp detection.
"""

import argparse
import datetime
import json
import logging
import os
import random
import types

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


def parse_args():
    parser = argparse.ArgumentParser(description='Train optimized grasp network')

    parser.add_argument('--network', type=str, default='mfenet')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--use-depth', type=int, default=1)
    parser.add_argument('--use-rgb', type=int, default=1)
    parser.add_argument('--use-dropout', type=int, default=1)
    parser.add_argument('--dropout-prob', type=float, default=0.1)
    parser.add_argument('--channel-size', type=int, default=32)
    parser.add_argument('--iou-threshold', type=float, default=0.25)

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--ds-shuffle', action='store_true', default=False)
    parser.add_argument('--ds-rotate', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batches-per-epoch', type=int, default=1000)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true', default=True)

    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False)
    parser.add_argument('--random-seed', type=int, default=123)

    return parser.parse_args()


def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def patch_predict_if_needed(net):
    if hasattr(net, 'predict'):
        return net

    def _predict(self, x):
        out = self.forward(x)
        if isinstance(out, dict):
            return out
        if isinstance(out, (list, tuple)) and len(out) == 4:
            return {'pos': out[0], 'cos': out[1], 'sin': out[2], 'width': out[3]}
        raise RuntimeError('Model forward() must return dict or 4-tuple (pos, cos, sin, width).')

    net.predict = types.MethodType(_predict, net)
    return net


def build_optimizer(net, args):
    optim_name = args.optim.lower()
    if optim_name == 'adamw':
        return optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim_name == 'adam':
        return optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim_name == 'sgd':
        return optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    raise NotImplementedError(f'Optimizer {args.optim} is not implemented')


@torch.no_grad()
def validate(net, device, val_data, iou_threshold):
    net.eval()
    results = {'correct': 0, 'failed': 0, 'loss': 0.0, 'losses': {}}
    ld = len(val_data)
    if ld == 0:
        return results

    for x, y, didx, rot, zoom_factor in val_data:
        xc = x.to(device, non_blocking=True)
        yc = [yy.to(device, non_blocking=True) for yy in y]

        lossd = net.compute_loss(xc, yc)
        loss = lossd['loss']
        results['loss'] += loss.item() / ld

        for ln, l in lossd['losses'].items():
            results['losses'][ln] = results['losses'].get(ln, 0.0) + l.item() / ld

        q_out, ang_out, w_out = post_process_output(
            lossd['pred']['pos'], lossd['pred']['cos'], lossd['pred']['sin'], lossd['pred']['width']
        )

        s = evaluation.calculate_iou_match(
            q_out,
            ang_out,
            val_data.dataset.get_gtbb(didx, rot, zoom_factor),
            no_grasps=1,
            grasp_width=w_out,
            threshold=iou_threshold,
        )

        if s:
            results['correct'] += 1
        else:
            results['failed'] += 1

    return results


def train_one_epoch(epoch, net, device, train_data, optimizer, scaler, batches_per_epoch, grad_clip=1.0, vis=False):
    net.train()
    results = {'loss': 0.0, 'losses': {}}

    batch_idx = 0
    data_iter = iter(train_data)
    while batch_idx < batches_per_epoch:
        try:
            x, y, _, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_data)
            x, y, _, _, _ = next(data_iter)

        batch_idx += 1
        xc = x.to(device, non_blocking=True)
        yc = [yy.to(device, non_blocking=True) for yy in y]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler.is_enabled()):
            lossd = net.compute_loss(xc, yc)
            loss = lossd['loss']

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

        if batch_idx % 100 == 0:
            logging.info('Epoch: %d, Batch: %d, Loss: %.4f', epoch, batch_idx, loss.item())

        results['loss'] += loss.item()
        for ln, l in lossd['losses'].items():
            results['losses'][ln] = results['losses'].get(ln, 0.0) + l.item()

        if vis and batch_idx % 200 == 0:
            imgs = []
            n_img = min(4, x.shape[0])
            for idx in range(n_img):
                imgs.extend(
                    [x[idx].detach().cpu().numpy().squeeze()] +
                    [yi[idx].detach().cpu().numpy().squeeze() for yi in y] +
                    [x[idx].detach().cpu().numpy().squeeze()] +
                    [pc[idx].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()]
                )
            gridshow(
                'Display',
                imgs,
                [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                [cv2.COLORMAP_BONE] * 10 * n_img,
                10,
            )
            cv2.waitKey(1)

    results['loss'] /= max(batch_idx, 1)
    for l in list(results['losses'].keys()):
        results['losses'][l] /= max(batch_idx, 1)

    return results


def run():
    args = parse_args()
    seed_everything(args.random_seed)

    save_folder = os.path.join(
        args.logdir,
        datetime.datetime.now().strftime('%y%m%d_%H%M') + '_' + args.description.replace(' ', '_')
    )
    os.makedirs(save_folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(save_folder, 'log.log')),
            logging.StreamHandler(),
        ]
    )

    with open(os.path.join(save_folder, 'commandline_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    tb = tensorboardX.SummaryWriter(save_folder)
    device = get_device(args.force_cpu)
    logging.info('Using device: %s', device)

    logging.info('Loading %s dataset...', args.dataset)
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
    )

    dataset_length = getattr(dataset, 'length', len(dataset))
    indices = list(range(dataset_length))
    split = int(np.floor(args.split * dataset_length))

    if args.ds_shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    common_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=val_sampler,
        drop_last=False,
        **common_loader_kwargs,
    )

    logging.info('Loading network: %s', args.network)
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    Network = get_network(args.network)
    net = Network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size,
    ).to(device)

    net = patch_predict_if_needed(net)

    optimizer = build_optimizer(net, args)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = GradScaler(enabled=(args.amp and device.type == 'cuda'))

    logging.info('Start training...')
    best_iou = 0.0

    for epoch in range(args.epochs):
        logging.info('Beginning Epoch %02d', epoch)

        train_results = train_one_epoch(
            epoch=epoch,
            net=net,
            device=device,
            train_data=train_data,
            optimizer=optimizer,
            scaler=scaler,
            batches_per_epoch=args.batches_per_epoch,
            grad_clip=args.grad_clip,
            vis=args.vis,
        )

        scheduler.step()

        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar(f'train_loss/{n}', l, epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        denom = max((test_results['correct'] + test_results['failed']), 1)
        val_iou = test_results['correct'] / denom

        logging.info('%d/%d = %.4f', test_results['correct'], denom, val_iou)

        tb.add_scalar('loss/IOU', val_iou, epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar(f'val_loss/{n}', l, epoch)

        if val_iou > best_iou or epoch == 0 or (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_folder, f'epoch_{epoch:02d}_iou_{val_iou:.4f}.pt')
            torch.save(net, ckpt_path)
            best_iou = max(best_iou, val_iou)
            logging.info('Model saved to %s', ckpt_path)

    tb.close()
    logging.info('Training completed. Best IOU: %.4f', best_iou)


if __name__ == '__main__':
    run()
