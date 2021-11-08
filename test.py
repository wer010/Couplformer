#!/usr/bin/env python

import argparse
from time import time
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import src as models
from utils.losses import LabelSmoothingCrossEntropy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("_")
                     and callable(models.__dict__[name]))

best_acc1 = 0

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    }
}

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR quick training script')

    # Data args
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')
    parser.add_argument('--model-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to model (default: model.pth)')
    # Optimization hyperparams
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-2, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', default=0., type=float,
                        help='gradient norm clipping (default: 0 (disabled))')

    parser.add_argument('-m', '--model',
                        type=str.lower,
                        choices=model_names,
                        default='cct_2', dest='model')

    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument('--conv-layers', default=2, type=int,
                        help='number of convolutional layers (cct only)')

    parser.add_argument('--conv-size', default=3, type=int,
                        help='convolution kernel size (cct only)')

    parser.add_argument('--patch-size', default=4, type=int,
                        help='image patch size (vit and cvt only)')

    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')

    parser.add_argument('--disable-aug', action='store_true',
                        help='disable augmentation policies for training')

    parser.add_argument('--gpu-id', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable cuda')

    return parser



def main():

    parser = init_parser()
    args = parser.parse_args()
    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']

    model = models.__dict__[args.model](img_size=img_size,
                                        num_classes=num_classes,
                                        positional_embedding=args.positional_embedding,
                                        n_conv_layers=args.conv_layers,
                                        kernel_size=args.conv_size,
                                        patch_size=args.patch_size)

    model.load_state_dict(torch.load(args.model_path))

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)



    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    augmentations = []
    if not args.disable_aug:
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
        ]
    augmentations += [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]

    augmentations = transforms.Compose(augmentations)


    val_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    print("Beginning testing")
    time_begin = time()


    acc = cls_validate(val_loader, model,  args)

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'final top-1: {acc:.2f}%')


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cls_validate(val_loader, model, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(val_loader)):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu_id, non_blocking=True)
                target = target.cuda(args.gpu_id, non_blocking=True)

            output = model(images)


            # since we're not training, we don't need to calculate the gradients for our outputs
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return (100 * correct / total)


if __name__ == '__main__':
    main()
