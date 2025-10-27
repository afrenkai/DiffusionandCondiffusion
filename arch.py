import torch
import torch.nn as nn
import numpy as np
from consts import TIME_EMBED_DIM, T

def sinusoidal_positional_encoding(tid: torch.Tensor) -> torch.tensor:
    embed_min_freq = 1.0
    freq = torch.exp(torch.linspace(np.log(embed_min_freq), np.log(T), TIME_EMBED_DIM //2)).view(1 ,-1).to(tid.device)
    angular_speeds = 2.0 * torch.pi * freq
    tid = tid.view(-1, 1).float()
    emb = torch.cat([torch.sin(tid.matmul(angular_speeds) / T), torch.cos(tid.matmul(angular_speeds)/T)], dim = 1)
    return emb


def double_conv(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dconv_down1 = double_conv(in_channels + TIME_EMBED_DIM, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
        align_corners=True)
        #self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up3 = double_conv(832, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x, time_index):
        time_embedding = sinusoidal_positional_encoding(time_index)
        x = torch.cat([x,
        time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2),
        x.size(3))], dim=1)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = torch.cat([x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out
