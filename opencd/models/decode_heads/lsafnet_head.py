import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.backbones.resnet import BasicBlock, ResNet
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from opencd.registry import MODELS

from .multi_head import MultiHeadDecoder


@MODELS.register_module()
class ResNet_BT(ResNet):
    """Output 5-level features from ResNet backbone."""

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
        x0, x1, x2, x3, x4 = x
        x4 = self.block1(x4)
        x4 = self.lgaa1(x4, x3)
        x4 = self.block2(x4)
        x4 = self.lgaa2(x4, x2)
        return x0, x1, x4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers(in_channels, out_channels)

    def _make_layers(self, in_channels, out_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, downsample))
        layers.append(BasicBlock(out_channels, out_channels))
        layers.append(
            nn.Conv2d(out_channels, out_channels, 3, 1, 2, dilation=2, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class SSDecoder(nn.Module):
    def __init__(self):
        super(SSDecoder, self).__init__()
        self.block1 = DecoderBlock(128 + 64, 128)
        self.block2 = DecoderBlock(128 + 64, 128)
        self.lgae = LGAE(channels=128)

    def forward(self, x0, x1, x4):
        x = F.interpolate(x4, x1.shape[2:], mode="bilinear")
        x = torch.cat([x, x1], 1)
        x1 = self.block1(x)
        x = F.interpolate(x1, x0.shape[2:], mode="bilinear")
        x = torch.cat([x, x0], 1)
        x0 = self.block2(x)
        x4 = self.lgae(x4)
        return x0, x1, x4


class CRDecoder(nn.Module):
    def __init__(self):
        super(CRDecoder, self).__init__()
        self.block1 = DecoderBlock(256, 128)
        self.block2 = DecoderBlock(256, 128)
        self.block3 = DecoderBlock(256, 128)
        self.ctfa = CTFA(in_channels=128)

    def forward(self, x0, x1, x1_c, x2_c):
        change_feats = self.ctfa(x1_c, x2_c)
        x = self.block1(change_feats)
        x = F.interpolate(x, x1.shape[2:], mode="bilinear")
        x1 = torch.cat([x, x1], 1)
        x = self.block2(x1)
        x = F.interpolate(x, x0.shape[2:], mode="bilinear")
        x = torch.cat([x, x0], 1)
        x = self.block3(x)
        return x


class CTFA(nn.Module):
    def __init__(self, in_channels):
        super(CTFA, self).__init__()
        self.height = 2
        inner_channels = max(int(in_channels / 4), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(
                nn.Conv2d(
                    inner_channels, in_channels, kernel_size=1, stride=1, bias=False
                )
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        abs_feats = torch.abs(x1 - x2)
        batch_size = x1.shape[0]
        channels = x1.shape[1]

        inp_feats_ = torch.cat([x1, x2], dim=1)
        inp_feats_ = inp_feats_.view(
            batch_size, self.height, channels, inp_feats_.shape[2], inp_feats_.shape[3]
        )

        feats_U = torch.sum(inp_feats_, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(
            batch_size, self.height, channels, 1, 1
        )
        attention_vectors = self.softmax(attention_vectors)
        attn_feats = torch.sum(inp_feats_ * attention_vectors, dim=1)

        return torch.cat([attn_feats, abs_feats], dim=1)


@MODELS.register_module()
class LSAFHead(MultiHeadDecoder):
    def __init__(self, **kwargs):
        super(LSAFHead, self).__init__()
        self.binary_cd_head = None  # FCNHead(...)
        self.semantic_cd_head = None
        self.semantic_cd_head_aux = None
        self.lgaa = LGAA()
        self.lgae = LGAE()

        #
        self.neck = SFM()
        self.ss_decoder = SSDecoder()
        self.cr_decoder = CRDecoder()

    def forward(self, inputs):
        x1, x2 = inputs

        x1_0, x1_1, x1_4 = self.neck(x1)
        x2_0, x2_1, x2_4 = self.neck(x2)

        x1_0, x1_1, x1_4 = self.ss_decoder(x1_0, x1_1, x1_4)
        x2_0, x2_1, x2_4 = self.ss_decoder(x2_0, x2_1, x2_4)

        change_feats = self.cr_decoder(
            torch.abs(x1_0 - x2_0), torch.abs(x1_1 - x2_1), x1_4, x2_4
        )


@MODELS.register_module()
class SSHead(BaseDecodeHead):
    def __init__(self, in_channels=256, channels=64, **kwargs):
        super(SSHead, self).__init__(dropout_ratio=0.0, **kwargs)

        self.decoder = nn.Sequential(
            DecoderBlock(self.in_channels, 128),
            ConvModule(128, self.channels, 3, padding=1, norm_cfg=self.norm_cfg),
        )

    def forward(self, inputs):
        pass
