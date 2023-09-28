import json

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-25
INF = 1e25

class CE_KL_loss(nn.Module):
    def __init__(self):
        super(CE_KL_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.C = None

    def forward(self, output, target, centroids):
        loss = self.ce_loss(output, target)
        surrogate_loss = F.kl_div(centroids.log(), F.log_softmax(output,1), reduction="batchmean", log_target=True)
        return loss, surrogate_loss


class CE_KL_ENP_loss(nn.Module):
    def __init__(self):
        super(CE_KL_ENP_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.C = None

    def __entropy(self, Prob):
        Prob1 = Prob.clone()
        Prob2 = Prob.clone()
        Prob1[Prob<1e-10] = 1.
        Prob2[Prob<1e-10] = 0.
        return -(Prob2*torch.log(Prob1)).sum(1)
    
    def forward(self, output, target, centroids):
        rttensor=torch.zeros(3).to(target.device)
        loss = self.ce_loss(output, target)
        surrogate_loss = F.kl_div(centroids.log(), F.log_softmax(output,1), reduction="batchmean", log_target = True)
        Prob = F.softmax(output,1)
        entropy = self.__entropy(Prob)
        rttensor[0], rttensor[1], rttensor[2] = loss, surrogate_loss, -torch.mean(entropy)
        # breakpoint()
        return rttensor
    
class CE_KL_LS_loss(nn.Module):
    def __init__(self, n_classes):
        super(CE_KL_LS_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.C = n_classes

    def smooth(self, target, eps):
        out = (eps / (self.C - 1)) * torch.ones((target.shape[0], self.C))
        for row, col in enumerate(target.cpu()):
            out[row, col] = 1 - eps
        out = out.to(target.device)
        return out

    def forward(self, output, target, centroids, eps):
        rttensor=torch.zeros(3).to(target.device)
        loss = self.ce_loss(output, target)
        surrogate_loss1 = F.kl_div(centroids.log(), F.log_softmax(output,1), reduction="batchmean", log_target=True)
        smooth_label = self.smooth(target, eps)
        surrogate_loss2 = F.kl_div(smooth_label.log(), F.log_softmax(output, 1), reduction = 'batchmean', log_target=True)
        rttensor[0], rttensor[1], rttensor[2] = loss, surrogate_loss1, surrogate_loss2
        return rttensor

class pair_wise_KL_Loss(nn.Module):
    def __init__(self,):
        super(pair_wise_KL_Loss, self).__init__()

    def forward(self, output, labels):
        rtval = 0 
        labels = labels[:, None]  # extend dim
        device = output.device
        output = F.softmax(output, dim=1)
        mask = torch.eq(labels, labels.t()).bool().to(device)
        mask_neg = (~mask).float()
        
        LogOutput = torch.nan_to_num(torch.log(output),0)
        NEntropy = torch.sum(output*LogOutput, dim=1, keepdim=True)#.expand(BatchSize)
        CrossEntropy = torch.matmul(output, LogOutput.t())
        PariWiseKlDiv = NEntropy-CrossEntropy
        rtval = torch.mean(mask_neg*PariWiseKlDiv)
        return rtval

class CE_KL_PWKL_loss(nn.Module):
    def __init__(self, n_classes):
        super(CE_KL_PWKL_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.C = n_classes
        self.PairWiseKlLoss = pair_wise_KL_Loss()
    def forward(self, output, target, centroids, eps):
        rttensor=torch.zeros(3).to(target.device)
        loss = self.ce_loss(output, target)
        surrogate_loss1 = F.kl_div(centroids.log(), F.log_softmax(output,1), reduction="batchmean", log_target=True)
        surrogate_loss2 = self.PairWiseKlLoss(output, target)
        rttensor[0], rttensor[1], rttensor[2] = loss, surrogate_loss1, surrogate_loss2
        return rttensor


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
        # breakpoint()
        Loganchor = torch.nan_to_num(torch.log(torch.clamp(anchor, EPS, INF)),0)
        OneHotMask = 1-F.one_hot(labels, self.n_classes).to(device)
        pair_wise_CE_W_Anchor = -(OneHotMask*torch.matmul(output, Loganchor.t())).reshape(-1)
        rtval = torch.mean(pair_wise_CE_W_Anchor[pair_wise_CE_W_Anchor!=0])
        return rtval
    

