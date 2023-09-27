from __future__ import print_function

import os
import argparse
import socket
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_fine_tune as train, validate
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import CE_KL_PWCE_loss
from centroid import Centroid

import matplotlib.pyplot as plt
import seaborn as sns

CIFARCLASS = ['Airplane', 'Automobile', 'Bird',
    'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    # breakpoint()
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict['resnet20'](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('confusion matrix')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--title', type=str, default=None, help='title')
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10'], help='dataset')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    opt.model = get_teacher_name(opt.path_t)
    
    # opt.model_name = '{}_{}_FT_lr_{}_decay_{}_CMI_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
    #                                                         opt.weight_decay,opt.param[1], opt.trial)

    return opt



def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        _, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        _, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # model
    # model = model_dict[opt.model](num_classes=n_cls)
    model = load_teacher(opt.path_t, n_cls).cuda()
    model = model.cuda()

    # tensorboard
    # logger = tb_z.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = None

    # routine
    model.eval()

    confusion_mat = np.zeros((n_cls, n_cls))

    with torch.no_grad():
        for img, tar in val_loader:
            img, tar = img.cuda(), tar.cuda()
            output = model(img)
            pred = torch.argmax(output, 1)
            pairs = torch.cat((tar.view(-1, 1), pred.view(-1, 1)), 1)
            pair, cnt = torch.unique(pairs, dim = 0, return_counts = True)
            pair, cnt = pair.cpu(), cnt.cpu()
            for idx, (gt, p) in enumerate(pair.numpy()):
                confusion_mat[gt][p] += cnt[idx].item()
        confusion_mat /= 1000

        # plt.imshow(confusion_mat)
        # plt.colorbar(ticks = [0, 1])

        # for i in range(n_cls):
        #     for j in range(n_cls):
        #         plt.annotate(str(confusion_mat[i][j]), xy=(j+0.5, i+0.5),
        #              ha='center', va='center', color='white')
        sns.set(rc={'figure.figsize':(11.7*1.6,8.27*1.6)})
        hm = sns.heatmap(data=confusion_mat, cmap="YlGnBu", annot=True, 
                         xticklabels=CIFARCLASS, yticklabels=CIFARCLASS)
        # plt.show()
        plt.title(opt.title.replace('_', ' '), fontsize = 20)
        plt.savefig("{}.pdf".format(opt.title),
                    bbox_inches ="tight",
                    transparent = True,
                    orientation ='landscape')



if __name__ == '__main__':
    main()
