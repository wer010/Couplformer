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


def init_arg():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="imagenet",
                        help="Which downstream task.")
    parser.add_argument("--model_type",
                        default="cct_sd",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    parser.add_argument("--num_workers", default=4, type=int,
                        help="number of workers")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")

    parser.add_argument("--num_layers", default=4, type=int,
                        help="num_layers")
    parser.add_argument("--num_heads", default=2, type=int,
                        help="num_heads")
    parser.add_argument("--mlp_ratio", default=1, type=int,
                        help="mlp_ratio")
    parser.add_argument("--embedding_dim", default=128, type=int,
                        help="embedding_dim")
    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument("--train_batch_size", default=100, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=20, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1602, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--pretrain", default=None, type=str,
                        help="path of pretrain model")


    parser.add_argument("--learning_rate", default=0.0005, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument('--clip-grad-norm', default=0., type=float,
                        help='gradient norm clipping (default: 0 (disabled))')

    parser.add_argument("--epoch", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epoch", default= 10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--warmup_lr", default=0.001, type=float,
                        help="lr of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--disable-aug', action='store_true',
                        help='disable augmentation policies for training')

    args = parser.parse_args()
    return args


def main():
    global best_acc1

    args = init_arg()
    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']
    img_size = 224
    model = models.__dict__[args.model_type](img_size=img_size,
                                        num_classes=num_classes,
                                        positional_embedding=args.positional_embedding,
                                        num_layers=args.num_layers,
                                        num_heads=args.num_heads,
                                        mlp_ratio=args.mlp_ratio,
                                        embedding_dim=args.embedding_dim,
                                        n_conv_layers=2,
                                        kernel_size=7)
    criterion = LabelSmoothingCrossEntropy()

    if args.pretrain is not None:
        model_dict = model.state_dict()
        model_dict.pop('classifier.fc.weight')
        model_dict.pop('classifier.fc.bias')
        pretrained_dict = {k: v for k, v in torch.load(args.pretrain).items() if k in model_dict}

        model.load_state_dict(pretrained_dict,strict=False)

    ctime = datetime.now()
    run_path = './runs/' + ctime.strftime("%Y-%b-%d_%H:%M:%S") + '_' + args.model_type+'_' + args.dataset
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    writer = SummaryWriter(run_path)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model.cuda(0)
        criterion = criterion.cuda(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    augmentations = []
    if not args.disable_aug:
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
        ]
    augmentations += [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        *normalize,
    ]



    augmentations = transforms.Compose(augmentations)
    train_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=True, download=True, transform=augmentations)

    val_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.num_workers)

    print("Beginning training")
    total_iter = 0

    for epoch in range(args.epoch):
        time_begin = time()
        if args.pretrain is None:
            adjust_learning_rate(optimizer, epoch, args)
        model.train()
        loss_val, acc1_val = 0, 0
        n = 0
        for i, (images, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            writer.add_scalar('Loss/train', loss, total_iter)
            writer.add_scalar('Acc/train', acc1[0], total_iter)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], total_iter)

            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

            optimizer.step()
            total_iter+=1
            if i % 100 == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                print(f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

        acc1 = cls_validate(val_loader, model, criterion, epoch)
        best_acc1 = max(acc1, best_acc1)
        total_mins = (time() - time_begin) / 60
        print(f'Epoch finished in {total_mins:.2f} minutes, with best acc {best_acc1:.2f} and final acc{acc1:.2f}')

    torch.save(model.state_dict(), run_path+'/'+'checkpoint.pth')


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif args.decay_type=='cosine':
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epoch) / (args.epoch - args.warmup_epoch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_train(train_loader, model, criterion, optimizer, epoch, args, writer):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu_id, non_blocking=True)
            target = target.cuda(args.gpu_id, non_blocking=True)
        output = model(images)

        loss = criterion(output, target)

        acc1 = accuracy(output, target)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        optimizer.step()

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            print(f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def cls_validate(val_loader, model, criterion,  epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    time_begin = time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc1:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc1


if __name__ == '__main__':
    main()
