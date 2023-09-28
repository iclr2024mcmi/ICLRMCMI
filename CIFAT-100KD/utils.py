""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

EPS = 1e-25
INF = 1e25

def accuracy(output, target, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        # batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0))
        return res
    
def get_network(args):
    if args.net == 'vgg11_bn':
        print("hi")
        from models.resnet_cifar import vgg11_bn
        net = vgg11_bn()
    if args.net == 'resnet56':
        from models.resnet_cifar import resnet56_cifar100
        net = resnet56_cifar100(pretrained=False)
    elif args.net == 'resnet110':
        from models.resnet_cifar import resnet110_cifar100
        net = resnet110_cifar100(pretrained=False)
    else:
        print('the network name you have entered is not supported yet')
    net = net.cuda()
    return net

class pair_wise_CE(nn.Module):
    def __init__(self,):
        super(pair_wise_CE, self).__init__()
    def forward(self, output, labels):
        rtval = 0 
        labels = labels[:, None]  # extend dim
        device = output.device
        output = F.softmax(output, dim=1)
        mask = torch.eq(labels, labels.t()).bool().to(device)
        mask_neg = (~mask).float()
        # breakpoint()
        LogOutput = torch.nan_to_num(torch.log(torch.clamp(output, EPS, INF)),0)
        CrossEntropy = -torch.matmul(output, LogOutput.t().detach())
        pair_wise_CE = (mask_neg*CrossEntropy).reshape(-1)
        rtval = torch.mean(pair_wise_CE[pair_wise_CE!=0])
        return rtval

class pair_wise_CE_W_Anchor(nn.Module):
    def __init__(self,n_classes):
        super(pair_wise_CE_W_Anchor, self).__init__()
        self.n_classes = n_classes
    def forward(self, output, labels, anchor):
        # breakpoint()
        device = output.device
        rtval = 0
        output = F.softmax(output, dim=1)
        Loganchor = torch.nan_to_num(torch.log(torch.clamp(anchor, EPS, INF)),0)
        OneHotMask = 1-F.one_hot(labels, self.n_classes).to(device)
        pair_wise_CE_W_Anchor = (OneHotMask*torch.matmul(output, Loganchor.t())).reshape(-1)
        rtval = torch.mean(pair_wise_CE_W_Anchor[pair_wise_CE_W_Anchor!=0])
        return rtval
    
