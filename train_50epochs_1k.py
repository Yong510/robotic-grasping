import argparse, datetime, json, logging, os, sys
import cv2, numpy as np, tensorboardX, torch, torch.optim as optim
from torch.utils.data import DataLoader
from hardware.device import get_device
from inference.models import get_network
from utils.data import get_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--network', default='mfenet')
    p.add_argument('--input-size', type=int, default=300)
    p.add_argument('--use-depth', type=int, default=1)
    p.add_argument('--use-rgb', type=int, default=1)
    p.add_argument('--use-dropout', type=int, default=1)
    p.add_argument('--dropout-prob', type=float, default=0.1)
    p.add_argument('--channel-size', type=int, default=32)
    p.add_argument('--dataset', required=True)
    p.add_argument('--dataset-path', required=True)
    p.add_argument('--split', type=float, default=0.9)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batches-per-epoch', type=int, default=1000)
    p.add_argument('--description', default='')
    p.add_argument('--logdir', default='logs/')
    return p.parse_args()

def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
    net.train()
    results = {'loss': 0, 'losses': {}}
    batch_idx = 0
    while batch_idx < batches_per_epoch:
        for sample in train_data:
            if batch_idx >= batches_per_epoch: break
            try:
                x, y, _, _, _ = sample
                xc, yc = x.to(device), [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)
                loss = lossd['loss']
                if batch_idx % 100 == 0:
                    logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                results['loss'] += loss.item()
                for k,v in lossd['losses'].items():
                    results['losses'][k] = results['losses'].get(k,0) + v.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_idx += 1
            except Exception as e:
                logging.warning(f"Skip batch: {e}")
                continue
    for k in results: results[k] = (results[k]/batch_idx) if isinstance(results[k],float) else {kk:vv/batch_idx for kk,vv in results[k].items()}
    return results

def run():
    args = parse_args()
    save_folder = os.path.join(args.logdir, datetime.datetime.now().strftime('%y%m%d_%H%M')+'_'+args.description)
    os.makedirs(save_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s',
                        handlers=[logging.FileHandler(f'{save_folder}/log'), logging.StreamHandler()])
    device = get_device(False)
    logging.info('Loading dataset...')
    ds = get_dataset(args.dataset)(args.dataset_path, output_size=args.input_size, random_rotate=True, random_zoom=True,
                                   include_depth=args.use_depth, include_rgb=args.use_rgb)
    logging.info(f'Dataset size: {len(ds)}')
    indices = list(range(len(ds)))
    split = int(args.split * len(ds))
    np.random.shuffle(indices)
    train_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=torch.utils.data.SubsetRandomSampler(indices[:split]), drop_last=True)
    logging.info('Loading network...')
    net = get_network(args.network)(input_channels=args.use_depth+3*args.use_rgb, dropout=args.use_dropout,
                                    prob=args.dropout_prob, channel_size=args.channel_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    logging.info('Start training')
    for epoch in range(args.epochs):
        logging.info(f'Beginning Epoch {epoch:02d}')
        train(epoch, net, device, train_loader, optimizer, args.batches_per_epoch)
        torch.save(net, f'{save_folder}/epoch_{epoch:02d}_iou_0.00')
        logging.info(f'Model saved at epoch {epoch}')
    logging.info('Training completed.')

if __name__ == '__main__':
    run()
