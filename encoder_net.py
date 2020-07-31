import torch
import torch.nn as nn
import torchvision.models as models


class ResNet_BYOL(torch.nn.Module):
    def __init__(self, network, mlp_size, pro_size):
        super(ResNet_BYOL, self).__init__()
        if network== 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif network == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(resnet.fc.in_features, mlp_size, pro_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_size, pro_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_size),
            nn.BatchNorm1d(mlp_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_size, pro_size)
        )

    def forward(self, x):
        return self.net(x)
