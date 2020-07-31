import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from encoder_net import ResNet_BYOL, MLPHead
from utils import distribute_over_GPUs

class BYOL(nn.Module):
    def __init__(self, online, target, predictor, args):
        super(BYOL, self).__init__()
        self.online = online
        self.target = target
        self.predictor = predictor
        self.m = args.m

        for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return -2 * (x * y).sum(dim=-1)


    def forward(self, x1, x2):
        pred_1 = self.predictor(self.online(x1))
        pred_2 = self.predictor(self.online(x2))

        with torch.no_grad():
            for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            target_2 = self.target(x1)
            target_1 = self.target(x2)

        loss = self.regression_loss(pred_1, target_1)
        loss += self.regression_loss(pred_2, target_2)

        return loss.mean()