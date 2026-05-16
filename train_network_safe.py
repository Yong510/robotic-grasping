import argparse
import datetime
import json
import logging
import os
import random
import csv

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data

from hardware.device import get_device
from inference.models import get_network
from utils.data import get_dataset
from utils.visualisation.gridshow import gridshow


def parse_args():
    parser = argparse.ArgumentParser(description='Train network without validation')

    # Network
    parser.add_argument('--network', type=str, default='mfenet')
    parser.add_argument('--input-size', type=int, default=300)
    parser.add_argument('--use-depth', type=int, default=1)
    parser.add_argument('--use-rgb', type=int, default=1)
    parser.add_argument('--use-dropout', type=int, default=1)
    parser.add_argument('--dropout-prob', type=float, default=0.1)
    parser.add_argument('--channel-size', type=int, default=32)

    # Dataset
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--split', type=float, default=1.0)
    parser.add_argument('--ds-shuffle', action='store_true', default=False)
    parser.add_argument('--ds-rotate', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=24)

    # Training
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batches-per-epoch', type=int, default=500)
    parser.add_argument('--optim', type=str, default='adam')

    # Logging
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False)
    parser.add_argument('--random-seed', type=int, default=123)

    return parser.parse_args()


def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    results = {
        'loss': 0.0,
        'losses': {}
    }

    net.train()
    batch_idx = 0

    while batch_idx < batches_per_epoch:
        for batch in train_data:
            if batch_idx >= batches_per_epoch:
                break

            try:
                x, y, _, _, _ = batch
                xc = x.to(device, non_blocking=True)
                yc = [yy.to(device, non_blocking=True) for yy in y]

                lossd = net.compute_loss(xc, yc)
                loss = lossd['loss']

                if batch_idx % 100 == 0:
                    logging.info('Epoch: %d, Batch: %d, Loss: %.6f', epoch, batch_idx, loss.item())

                results['loss'] += loss.item()
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0.0
                    results['losses'][ln] += l.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_idx += 1

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
                        10
                    )
                    cv2.waitKey(1)

            except Exception as e:
                logging.warning("Skip broken batch: %s", repr(e))
                continue

    if batch_idx > 0:
        results['loss'] /= batch_idx
        for l in results['losses']:
            results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()
    seed_everything(args.random_seed)

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    save_folder = os.path.join(args.logdir, net_desc)
    os.makedirs(save_folder, exist_ok=True)

    tb = tensorboardX.SummaryWriter(save_folder)

    with open(os.path.join(save_folder, 'commandline_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(save_folder, 'train_loss.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss'])

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_folder, 'log.log'),
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    device = get_device(args.force_cpu)

    logging.info('Loading %s Dataset...', args.dataset)
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb
    )

    dataset_length = getattr(dataset, 'length', len(dataset))
    indices = list(range(dataset_length))

    if args.ds_shuffle:
        np.random.shuffle(indices)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    ).to(device)

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    logging.info('Start training only, no validation.')
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch %02d', epoch)

        train_results = train(
            epoch, net, device, train_data, optimizer,
            args.batches_per_epoch, vis=args.vis
        )

        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_results['loss']])

        # 每个 epoch 都保存
        save_path = os.path.join(save_folder, 'epoch_%02d_trainloss_%0.6f.pt' % (epoch, train_results['loss']))
        torch.save(net, save_path)
        logging.info('Model saved: %s', save_path)

    tb.close()
    logging.info('Training completed.')


if __name__ == '__main__':
    run()
