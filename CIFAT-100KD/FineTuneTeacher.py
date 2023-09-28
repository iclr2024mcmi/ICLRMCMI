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

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_fine_tune as train, validate
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import CE_KL_PWCE_loss
from centroid import Centroid

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
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--CentroidSampleSize', type=int, default=8, help='num of workers to use')
    parser.add_argument('--param', nargs='*', type=float, default=[1], help='coefficients apply on loss func')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    opt.model = get_teacher_name(opt.path_t)
    # breakpoint()
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    
    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.learning_rate = opt.learning_rate/2
    opt.model_name = '{}_{}_FT_lr_{}_decay_{}_CMI_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay,opt.param[1], opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt



def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    # model = model_dict[opt.model](num_classes=n_cls)
    model = load_teacher(opt.path_t, n_cls).cuda()
    samples_data, samples_tar = [], []
    train_label_list = torch.tensor(train_loader.dataset.targets)
    CMIs = []
    CEs = []
    for Class in tqdm(range(100)):
        idx = (train_label_list == Class).nonzero().squeeze().numpy()
        sampler = SubsetRandomSampler(idx)
        class_loader = DataLoader(train_loader.dataset, batch_size = 500, sampler = sampler, pin_memory = True)
        img, tar = next(iter(class_loader))
        samples_data.append(img.numpy())
        samples_tar.append(tar)
        # data = training_loader.dataset.data[idx]
        # tar = [Class] * len(data)
        # class_dataset = dataset(data, tar, transform)
        # class_loaders[Class] = DataLoader(class_dataset, batch_size = 32, shuffle = True, pin_memory = True)

    # samples_data = torch.cat(samples_data, 0)
    samples_data = np.array(samples_data)
    samples_tar = torch.cat(samples_tar, 0)

    centroids = Centroid(100, samples_data, samples_tar, 1, CdecayFactor=0.9999)
    centroids.update_epoch(model, train_loader)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()
    valcriterion = nn.CrossEntropyLoss()
    if len(opt.param)>1:
        criterion = CE_KL_PWCE_loss(n_classes=n_cls)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    # logger = tb_z.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = None

    # routine
    LrScheduler = CosineAnnealingLR(optimizer,opt.epochs)
    for epoch in range(1, opt.epochs + 1):

        # adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        #test_acc, test_acc_top5, test_loss = validate(train_loader, model, valcriterion, opt)
        #test_acc, test_acc_top5, test_loss = validate(val_loader, model, valcriterion, opt)
        
        train_acc, train_loss = train(epoch, train_loader, centroids, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, valcriterion, opt)
        LrScheduler.step()
        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        # logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        # if epoch % 1 == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model.state_dict(),
        #         'accuracy': test_acc,
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    # print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, 'ckpt_last.pth')
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
