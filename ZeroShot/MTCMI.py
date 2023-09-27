import numpy as np
import os
import torch
from models import wrn_40_2
from tqdm import tqdm
from dataset.cifar100 import get_cifar100_dataloaders
from utils import accuracy
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
DIR = "./save/models"

model = wrn_40_2(num_classes=100).cuda()
train_loader, val_loader = get_cifar100_dataloaders(batch_size=128, num_workers=32)
ce_loss = nn.CrossEntropyLoss()

for filename in os.listdir(DIR):
    if "FT" in filename:
        # breakpoint()
        CEs=[]
        CMIs=[]
        ACCs = []
        for i in range(1):
            
            # ckpt = torch.load(os.path.join(DIR, filename,"ckpt_epoch_{}.pth".format(i)))
            ckpt = torch.load(os.path.join(DIR, "wrn_40_2_vanilla","ckpt_epoch_240.pth"))
            model.load_state_dict(ckpt['model'])
            cnt = 0
            ce=0
            pointer = torch.zeros([100])
            logits = (torch.ones([100, 500, 100])*(-10000)).cuda()
            Ncorrect = 0
            model.eval()
            for image, target in tqdm(train_loader):
                image,target = image.cuda(), target.cuda()
                logit = model(image).detach()
                Ncorrect += accuracy(logit, target, [1])[0]
                cnt += target.shape[0]
                Classes, pointer_INC =  target.cpu().unique(return_counts=True)
                logit = logit
                ce+=ce_loss(logit, target)
                for idx, Class in enumerate(Classes):
                    logits[Class, int(pointer[Class]):int(pointer[Class]) + pointer_INC[idx]] = logit[target.cpu() == Class]
                    pointer[Class] += pointer_INC[idx]
                    Prob = F.softmax(logits, 2)
            Centroids = torch.mean(Prob, 1)
            LogCentroids = Centroids.log()[:,None,:].expand(-1,500,-1).cpu()
            CMI = F.kl_div(LogCentroids.reshape(-1,100), F.log_softmax(logits,2).reshape(-1,100).cpu(), \
                        reduction="batchmean", log_target=True)
            CMIs.append(float(CMI))
            CEs.append(float(ce/cnt))
            ACCs.append(float(Ncorrect/cnt))
        plt.plot(CMIs, label="CMI")
        plt.plot(CEs, label="CE")
        plt.title(filename)
        plt.legend()
        plt.show()
        breakpoint()

