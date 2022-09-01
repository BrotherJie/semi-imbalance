from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from time import strftime, localtime

import models.wrn as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


parser = argparse.ArgumentParser(description='PyTorch FixMatch post-hoc logit adjustment')
# Optimization options

parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', required=True,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--num_max', type=int, default=1500,
                    help='Number of samples in the maximal class')
parser.add_argument('--ratio', type=float, default=2.0,
                    help='Relative size between labeled and unlabeled data')
parser.add_argument('--imb_ratio', type=int, default=100,
                    help='Imbalance ratio for data')
parser.add_argument('--imb_ratio_u', type=int, default=100, help='Imbalance ratio for unlabeled data')
# Dataset options
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Dataset')


# logit adjustment with post-hoc way options
parser.add_argument('--logit_adj_post', help='adjust logits post hoc', type=int, default=1, choices=[0, 1])
parser.add_argument('--tro_post_range', help='check diffrent val of tro in post hoc', type=list,
                    default=[2, 2.25, 2.5, 2.75, 3, 3.25, 3.5])


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset
    num_class = 10
    args.num_max = 1500
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    num_class = 100
    args.num_max = 150





def main():
    current = strftime('%Y-%m-%d-%H-%M-%S', localtime())
    args.out = args.out + '/' + current
    mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced ', args.dataset)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio_u)

    train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar('../datasets', N_SAMPLES_PER_CLASS,
                                                                         U_SAMPLES_PER_CLASS)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=4, drop_last=True)
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

    if use_cuda:
        cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()

    # Resume
    title = 'fix-cifar-10-post-hoc-logit-adjustment'
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    pretrained_dict = checkpoint['ema_state_dict']
    #load_dict = {k: v for k, v in pretrained_dict.items() if 'output' not in k}
    model.load_state_dict(pretrained_dict, strict=False)


    logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    logger.set_names(['Value of tro', 'Val Loss', 'Val Accuracy'])


    print('===> start post-hoc logit adjustment')

    best_acc = 0.0
    best_tro = 0.0
    precision = []
    recall = []
    for tro in args.tro_post_range:
        #args.logit_adjustments = compute_adjustment(unlabeled_trainloader, model, tro, args)
        #args.logit_adjustments = compute_adjustment_all(labeled_trainloader,unlabeled_trainloader, model, tro, args)
        args.logit_adjustments = compute_adjustment_label(N_SAMPLES_PER_CLASS, tro, args)
        val_loss, val_acc, val_section_acc, val_classwise_precision, val_classwise_recall = validate(test_loader, model, criterion, use_cuda, 'Test state')
        print('Classwise Precision: ', val_classwise_precision)
        print('Classwise Recall: ', val_classwise_recall)
        logger.append([tro, val_loss, val_acc])
        if val_acc > best_acc:
            best_acc = val_acc
            best_tro = tro
            precision = val_classwise_precision
            recall = val_classwise_recall
    logger.close()

    print('====================')
    print()
    print('Imbalanced ratio of labeled data: ', args.imb_ratio)
    print('Best accuracy:', best_acc)
    print('Best tro: ', best_tro)
    print('Precision: ', precision)
    print('Recall: ', recall)




def validate(valloader, model, criterion, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    classwise_TP = torch.zeros(num_class)
    classwise_FP = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    if use_cuda:
        classwise_correct = classwise_correct.cuda()
        classwise_num = classwise_num.cuda()
        classwise_TP = classwise_TP.cuda()
        classwise_FP = classwise_FP.cuda()
        section_acc = section_acc.cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            #print('logits: ', outputs)
            loss = criterion(outputs, targets)
            if args.logit_adj_post:
                outputs = outputs - args.logit_adjustments

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()
                classwise_TP[i] += (class_mask * pred_mask).sum()
                classwise_FP[i] += ((1 - class_mask) * ((pred_label == i).float())).sum()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    classwise_precision = (classwise_TP / (classwise_TP + classwise_FP))
    classwise_recall = (classwise_TP / classwise_num)

    if use_cuda:
        classwise_precision = classwise_precision.cpu()
        classwise_recall = classwise_recall.cpu()
        section_acc = section_acc.cpu()

    return (losses.avg, top1.avg, section_acc.numpy(), classwise_precision.numpy(), classwise_recall.numpy())


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)


'''
参数：unlabeled loader, stage1训练好的模型，tro, args
返回值：adjustment

经过stage1训练完后的模型能有效缓解long-tail问题，所以此时的imbalanced ratio变小了
可以通过对整个unlabeled dataloader的数据进行预测，获得各个类别数据的数量作为先验来改变模型
'''
def compute_adjustment(unlabeled_loader, model, tro, args):
    """compute the base probabilities"""
    #model.eval()
    label_freq = {}
    with torch.no_grad():
        for _, ((inputs, inputs_s_1, inputs_s_2), _, index) in enumerate(unlabeled_loader):
            inputs = inputs.cuda()
            logits, _ = model(inputs)
            # torch.max返回指定维度数据的最大值和对应的index
            # 返回的index就代表对应的类
            _, predicted = torch.max(logits, 1)
            for j in predicted:
                key = int(j.item())
                label_freq[key] = label_freq.get(key, 0) + 1

        label_freq = dict(sorted(label_freq.items()))
        label_freq_array = np.array(list(label_freq.values()))

        for i in range(num_class - label_freq_array.size):
            label_freq_array = np.append(label_freq_array,[1])
        print('====================================')
        print('predicted labels of unlabeled data: ', label_freq_array)
        imbalanced_ratio = label_freq_array[0]/label_freq_array[-1]
        print('imbalanced ratio of predicted label data', imbalanced_ratio)
        label_freq_array = label_freq_array / label_freq_array.sum()
        print('label frequence:', label_freq_array)
        adjustments = np.log(label_freq_array ** tro + 1e-12)
        print('value of tro: ', tro)
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.cuda()
        return adjustments

def compute_adjustment_label(label_freq, tro, args):

    label_freq_array = np.array(label_freq)
    print('====================================')
    print('labels of data: ', label_freq_array)
    label_freq_array = label_freq_array / label_freq_array.sum()
    print('label frequence:', label_freq_array)
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    print('value of tro: ', tro)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments

def compute_adjustment_all(labeled_trainloader, unlabeled_trainloader, model, tro, args):
    label_freq = {}
    with torch.no_grad():
        for _, (inputs_label, _, index) in enumerate(labeled_trainloader):
            inputs_label = inputs_label.cuda()
            logits_label, _ = model(inputs_label)
            # torch.max返回指定维度数据的最大值和对应的index
            # 返回的index就代表对应的类
            _, predicted = torch.max(logits_label, 1)
            for j in predicted:
                key = int(j.item())
                label_freq[key] = label_freq.get(key, 0) + 1

        for _, ((inputs, inputs_s_1, inputs_s_2), _, index) in enumerate(unlabeled_trainloader):
            inputs = inputs.cuda()
            logits, _ = model(inputs)
            # torch.max返回指定维度数据的最大值和对应的index
            # 返回的index就代表对应的类
            _, predicted = torch.max(logits, 1)
            for j in predicted:
                key = int(j.item())
                label_freq[key] = label_freq.get(key, 0) + 1

        label_freq = dict(sorted(label_freq.items()))
        label_freq_array = np.array(list(label_freq.values()))

        for i in range(num_class - label_freq_array.size):
            label_freq_array = np.append(label_freq_array, [1])
        print('====================================')
        print('predicted labels of all data: ', label_freq_array)
        imbalanced_ratio = label_freq_array[0] / label_freq_array[-1]
        print('imbalanced ratio of predicted label data', imbalanced_ratio)
        label_freq_array = label_freq_array / label_freq_array.sum()
        print('label frequence:', label_freq_array)
        adjustments = np.log(label_freq_array ** tro + 1e-12)
        print('value of tro: ', tro)
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.cuda()
        return adjustments

if __name__ == '__main__':
    main()
