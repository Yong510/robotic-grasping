import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


def parse_args():
    parser = argparse.ArgumentParser(description='Train network with resume capability')
    parser.add_argument('--network', type=str, default='grconvnet3')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--use-depth', type=int, default=1)
    parser.add_argument('--use-rgb', type=int, default=1)
    parser.add_argument('--use-dropout', type=int, default=1)
    parser.add_argument('--dropout-prob', type=float, default=0.1)
    parser.add_argument('--channel-size', type=int, default=32)
    parser.add_argument('--iou-threshold', type=float, default=0.25)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--ds-shuffle', action='store_true', default=False)
    parser.add_argument('--ds-rotate', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batches-per-epoch', type=int, default=1000)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False)
    parser.add_argument('--random-seed', type=int, default=123)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=0, help='Epoch to start from (if resuming)')
    return parser.parse_args()


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    results = {'loss': 0, 'losses': {}}
    net.train()
    batch_idx = 0
    while batch_idx < batches_per_epoch:
        for sample in train_data:
            if batch_idx >= batches_per_epoch:
                break
            try:
                x, y, _, _, _ = sample
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)
                loss = lossd['loss']

                if batch_idx % 100 == 0:
                    logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

                results['loss'] += loss.item()
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_idx += 1
            except Exception as e:
                logging.warning(f"Skipping batch due to error: {e}")
                continue

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx
    return results


def run():
    args = parse_args()
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    save_folder = os.path.join(args.logdir, net_desc)
    os.makedirs(save_folder, exist_ok=True)
    tb = tensorboardX.SummaryWriter(save_folder)

    params_path = os.path.join(save_folder, 'commandline_args.json')
    with open(params_path, 'w') as f:
        json.dump(vars(args), f)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    device = get_device(args.force_cpu)
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb)
    logging.info('Dataset size is {}'.format(dataset.length))

    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        drop_last=True
    )
    logging.info('Done')

    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    )
    net = net.to(device)

    # 加载 checkpoint（如果指定）
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        net.load_state_dict(checkpoint.state_dict())
        logging.info(f'Resumed model from {args.resume}')
        # 如果 checkpoint 包含 epoch 信息，可以提取，这里简化，用户可通过 --start-epoch 指定
    logging.info('Done')

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    for epoch in range(start_epoch, args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # 每个 epoch 保存一次模型
        torch.save(net, os.path.join(save_folder, 'epoch_{:02d}_iou_0.00'.format(epoch)))
        logging.info(f'Model saved at epoch {epoch}')

    logging.info('Training completed.')


if __name__ == '__main__':
    run()
