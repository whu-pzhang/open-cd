import torch
from mmcv.cnn import ConvModule
from mmseg.models.backbones.resnet import BasicBlock
from torch import nn

from opencd.registry import MODELS

from .multi_head import MultiHeadDecoder


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cba(x)


class CBA3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cba(x)


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel
            ),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )

    def forward(self, x):
        return self.dwconv(x)


@MODELS.register_module()
class Multi_Level_Feature_Aggreagation(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)

        # self.cat_conv = CBA1x1(384, 128)
        self.cat_conv = ConvModule(
            384,
            128,
            kernel_size=1,
            padding=1,
            dilation=0,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )

    def forward(self, x1, x2, x3):
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.cat_conv(x)
        return x


class ECA(nn.Module):
    def __init__(self, kernal=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernal, padding=(kernal - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# Change_Specific_Transfer
class BCFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = CBA1x1(in_channels * 2, in_channels)
        self.eca = ECA()
        self.resblock = self._make_layer(BasicBlock, 256, 128, 6, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        xc1 = self.conv(torch.cat([x1, x2], dim=1))
        xc2 = self.conv(torch.cat([x2, x1], dim=1))
        change = self.eca(xc1 + xc2)
        diff = torch.abs(x1 - x2)
        change = torch.cat([change, diff], dim=1)
        change = self.resblock(change)
        return change


class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # upsample
        self.upconv = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=7,
            stride=2,
            padding=3,
            output_padding=1,
        )
        # self.catconv = CBA3x3(out_channel * 2, out_channel)
        self.catconv = ConvModule(
            out_channel * 2,
            out_channel,
            kernel_size=3,
            padding=1,
            dilation=1,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, up, skip):
        # upsample
        up = self.upconv(up)
        up = torch.cat([up, skip], dim=1)
        up = self.catconv(up)
        return up


class TaskInteractionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")

    def forward(self, old_sem1, old_sem2, old_change, change_result):
        sem_max_out, _ = torch.max(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_avg_out = torch.mean(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_out = self.sigmoid(
            self.Sem2Change(torch.cat([sem_max_out, sem_avg_out], dim=1))
        )
        new_change = old_change * sem_out

        b, c, h, w = old_sem1.size()
        fea_sem1 = torch.reshape(old_sem1.permute(0, 2, 3, 1), [b * h * w, c])
        fea_sem2 = torch.reshape(old_sem2.permute(0, 2, 3, 1), [b * h * w, c])

        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        target = unchange_mask.float()
        target = target - change_mask.float()
        target = torch.reshape(target, [b * h * w])
        similarity_loss = self.loss_f(fea_sem1, fea_sem2, target)
        return new_change, similarity_loss


@MODELS.register_module()
class BTSCDHEad(MultiHeadDecoder):
    def __init__(self, **kwargs):
        super().__init__()

        self.binary_cd_head = None
        self.semantic_cd_head = None
        self.semantic_cd_head_aux = None
