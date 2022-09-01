import argparse
import os
import torch
import numpy as np
import random

import torch.utils.data as data
from time import strftime, localtime
import models.wrn as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn import manifold

from utils import mkdir_p

parser = argparse.ArgumentParser(description='PyTorch t-SNE Training')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--num_max', type=int, default=1500,
                    help='Number of labeled data')
parser.add_argument('--imb_ratio_l', type=int, default=100,
                    help='Number of labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100,
                    help='Number of labeled data')

parser.add_argument('--out', default='result',
                    help='Directory to output the result')

# Dataset options
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Dataset')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset

    num_class = 10
    args.num_max = 1500
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset

    num_class = 100
    args.num_max = 150

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)


def plot_tsne(all_feature, all_targets):
    #print('check feature vector:', all_feature[:10, : ])没有null值

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    all_feature_tsne = tsne.fit_transform(all_feature)


    #all_targets = np.array(all_targets)
    #嵌入空间可视化
    x_min, x_max = all_feature_tsne.min(0), all_feature_tsne.max(0)

    X_norm = (all_feature_tsne - x_min) / (x_max - x_min)
    figure1 = plt.figure(figsize=(10,10))
    figure2 = plt.figure(figsize=(10,10))
    ax1 = figure1.subplots()  # type(ax1) = axes
    ax2 = figure2.subplots()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    for i in range(X_norm.shape[0]):
        colors = plt.cm.Set3(all_targets[i])

        ax1.text(X_norm[i, 0], X_norm[i, 1], str(all_targets[i]), color=colors, fontdict={'weight': 'bold', 'size':9})
        ax2.scatter(X_norm[i, 0], X_norm[i, 1], color=colors, s=4)

    figure1.savefig('tsne_class.png', dpi=600)
    figure2.savefig('tsne_only.png',dpi=600)



def main():
    global best_acc


    # Data
    print(f'==> Preparing cifar10')
    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, 10, args.imb_ratio_l)  # 每个类带标签数据的数量，num_max是cifar10第一个类的数量
    U_SAMPLES_PER_CLASS = make_imb_data(2 * args.num_max, 10, args.imb_ratio_u)  # 每个类不带标签数据的数量。这里设置的是带标签数据的2倍
    _, _, test_set = dataset.get_cifar('../datasets', N_SAMPLES_PER_CLASS,
                                       U_SAMPLES_PER_CLASS)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(2, num_class)
        if use_cuda:
            model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    if use_cuda:
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    pretrained_dict = checkpoint['ema_state_dict']
    load_dict = {k: v for k, v in pretrained_dict.items()}
    model.load_state_dict(load_dict, strict=False)
    ema_model.load_state_dict(load_dict, strict=False)

    # Evaluation part
    all_feature = validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

    all_feature = all_feature.cpu().numpy()

    plot_tsne(all_feature, test_set.targets)


def validate(valloader, model, criterion, use_cuda, mode):
    all_feature = []
    all_target = []
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _, feature = model(inputs, return_feature=True)  # 这只是一个batch 的 feature，要收集所有的batch的feature后才能做tsne
            all_feature.append(feature)

    all_feature = torch.cat(all_feature, dim=0)

    return all_feature.squeeze()


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):  # [0,9)
        if i == (class_num - 1):  #
            class_num_list.append(int(max_num / gamma))  # 最后一个类
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)


if __name__ == '__main__':
    main()
