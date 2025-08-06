import torch.nn as nn
from mmseg.models.backbones.resnet import ResNet

from opencd.registry import MODELS


@MODELS.register_module()
class ResNet_BT(ResNet):
    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = [x]
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class LGAE(nn.Module):
    def __init__(self, channels=128, r=4):
        super(LGAE, self).__init__()
        inner_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lattn_feats = self.local_att(x)
        gattn_feats = self.global_att(x)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return x * w


class LGAA(LGAE):
    def forward(self, x1, x2):
        addition_feats = x1 + x2
        lattn_feats = self.local_att(addition_feats)
        gattn_feats = self.global_att(addition_feats)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return 2 * x1 * w + 2 * x2 * (1 - w)


# Semantic Fusion Module
class SFM(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], out_channels=128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.lgaa1 = LGAA(channels=256)
        self.lgaa2 = LGAA(channels=128)

    def forward(self, x):
        x2, x3, x4 = x
        x4 = self.block1(x4)
        x4 = self.lgaa1(x4, x3)
        x4 = self.block2(x4)
        x4 = self.lgaa2(x4, x2)
        return x4


@MODELS.register_module()
class LSAFHead(nn.Module):
    def __init__(self, **kwargs):
        super(LSAFHead, self).__init__()
        self.binary_cd_head = None
        self.semantic_cd_head = None
        self.semantic_cd_head_aux = None
        self.lgaa = LGAA()
        self.lgae = LGAE()
