#!/usr/bin/env python

import argparse
from time import time
from datetime import datetime
import math
import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import src as models
from utils.losses import LabelSmoothingCrossEntropy
from thop import profile

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


    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=200, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')

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
    global best_acc1

    parser = init_parser()
    args = parser.parse_args()
    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']


    for n in model_names:
        if n in ['cct_sd', 'kct_sd']:
            continue
        model = models.__dict__[n](img_size=224,
                                            num_classes=1000,
                                            positional_embedding=args.positional_embedding,
                                            n_conv_layers=args.conv_layers,
                                            kernel_size=args.conv_size,
                                            patch_size=args.patch_size)
        criterion = LabelSmoothingCrossEntropy()

        input = torch.randn(1, 3, 224, 224)
        macs, params = profile(model, inputs=(input,))
        print(f'{n} model Macs is {macs}, params is {params}')





if __name__ == '__main__':
    main()
