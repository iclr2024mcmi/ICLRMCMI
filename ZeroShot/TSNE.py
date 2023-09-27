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
from copy import deepcopy
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_fine_tune as train, validate
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import CE_KL_PWCE_loss
from centroid import Centroid

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

CIFARCLASS = ['Airplane', 'Automobile', 'Bird',
    'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
cifar_class_dict = {i + 1: CIFARCLASS[i] for i in range(len(CIFARCLASS))}

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
    if '20' in model_path:
        model = model_dict['resnet20'](num_classes=n_cls)
    elif '56' in model_path:
        model = model_dict['resnet56'](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--title', type=str, default=None, help='title')
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10'], help='dataset')
    parser.add_argument('--dci', '--drop_class_idx', nargs='*', type=float, default=[0, 3])
    opt = parser.parse_args()
    opt.dci = np.array(opt.dci)
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
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
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

    Probs = (torch.ones([10, 1000, 10])*(-1))
    pointer = torch.zeros([10])
    targets = torch.cumsum(torch.ones(10,1).expand(-1,1000),0).reshape(-1)

    with torch.no_grad():
        for img, tar in tqdm(val_loader):
            img, tar = img.cuda(), tar.cuda()
            output = model(img)
            Prob = output
            Classes, pointer_INC =  tar.cpu().unique(return_counts=True)
            for idx, Class in enumerate(Classes):
                Probs[Class, int(pointer[Class]):int(pointer[Class]) + pointer_INC[idx]] = Prob[tar.cpu() == Class]
                pointer[Class] += pointer_INC[idx]
    tsne = TSNE(random_state=0, perplexity=15)
    tsne_output = tsne.fit_transform(Probs.reshape(-1,10))
    plt.rcParams['figure.figsize'] = 10, 10
    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['Class'] = targets
    dfcpy = deepcopy(df)
    # breakpoint()
    mask = pd.core.series.Series(np.array(((np.array(dfcpy['Class'])-opt.dci[:,None])==0).sum(0),dtype=bool))
    dfcpy.loc[mask, ['x', 'y']] = -1000000
    dfcpy.loc[mask, ['x', 'y']] = -1000000
    # breakpoint()
    sns.scatterplot(
        x='x', y='y',
        hue='Class',
        palette=sns.color_palette("colorblind", 10),
        data=dfcpy,
        marker='o',
        legend=False,
        alpha=0.6
    )

    
    dfcpy = deepcopy(df)
    mask = pd.core.series.Series(~np.array(((np.array(dfcpy['Class'])-opt.dci[:,None])==0).sum(0),dtype=bool))
    dfcpy.loc[mask, ['x', 'y']] = -1000000
    dfcpy.loc[mask, ['x', 'y']] = -1000000
    dfcpy['Class'] = dfcpy['Class'].replace(cifar_class_dict)
    # breakpoint()
    sns.scatterplot(
        x='x', y='y',
        hue='Class',
        palette=sns.color_palette("colorblind", 10),
        data=dfcpy,
        marker='o',
        legend="brief",
        alpha=0.6
    )
    plt.xlim(-125, 125)
    plt.ylim(-125, 125)

    # plt.show()
    plt.savefig("{}.pdf".format(opt.title),
                bbox_inches ="tight",
                transparent = True,
                orientation ='landscape')

if __name__ == '__main__':
    main()
