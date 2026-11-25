"""Spatial coupling blocks. Derived from the published MSDDN implementation at 834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19. Parameter names and active operations are retained."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixing import InvertibleConv1x1

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out
