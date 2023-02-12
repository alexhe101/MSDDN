#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:47:04
LastEditTime: 2020-12-09 23:12:31
Description: PanNet: A deep network architecture for pan-sharpening (VDSR-based)
2000 epoch, decay 1000 x0.1, batch_size = 128, learning_rate = 1e-2, patch_size = 33, MSE
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F


import scipy
import scipy.signal
import numpy as np
import math


class FRFT(nn.Module):
    def __init__(self, in_channels, out_channels, channel_reduction, stride=1, groups=1):
        super(FRFT, self).__init__()
        C0 = int(in_channels/channel_reduction/3)
        C1 = int(in_channels/channel_reduction) - 2*C0
        self.mag_s = nn.Conv2d(C1, C1, kernel_size=3, padding=1)
        self.mag_f = nn.Conv2d(C1, C1, kernel_size=1, padding=0)
        self.mag = nn.Conv2d(C1, C1, kernel_size=1, padding=0)
        self.pha = nn.Conv2d(C1, C1, kernel_size=1, padding=0)
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=1, padding=0)
        self.conv_1 = nn.Conv2d(C0, C0, kernel_size=1, padding=0)

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels // channel_reduction, kernel_size=1, groups=groups, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels // channel_reduction, out_channels, kernel_size=1, groups=groups, bias=False)

    def frft2d(self, matrix, a=[0.7, 0.7]):
        temp = torch.zeros([matrix.shape[0], matrix.shape[1]], dtype=torch.complex64)
        for k in range(0, matrix.shape[0]):
            temp[k, :] = self.frft(matrix[k, :], a[0])
        out = torch.zeros((temp.shape[0], temp.shape[1]), dtype=torch.complex64)
        for m in range(0, temp.shape[1]):
            out[:, m] = self.frft(temp[:, m], a[1])

        return out

    def ifrft2d(self, matrix, a=[0.7, 0.7]):
        temp = torch.zeros((matrix.shape[0], matrix.shape[1]), dtype=torch.complex64)
        for k in range(0, matrix.shape[0]):
            temp[k, :] = self.ifrft(matrix[k, :], a[0])
        out = torch.zeros((temp.shape[0], temp.shape[1]), dtype=torch.complex64)
        for m in range(0, temp.shape[1]):
            out[:, m] = self.ifrft(temp[:, m], a[1])

        return out

    def frft(self, f, a):
        ret = torch.zeros(f.shape[0], dtype=torch.complex64)
        f = torch.tensor(f, dtype=torch.complex64)
        N1 = torch.tensor(len(f))
        N = len(f)
        shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
        sN = torch.sqrt(N1)

        # Special cases
        if a == 0.0:
            return f
        if a == 2.0:
            return torch.flipud(f)
        if a == 1.0:
            ret[shft] = torch.fft.fft(f[shft]) / sN
            return ret
        if a == 3.0:
            ret[shft] = torch.fft.ifft(f[shft]) * sN
            return ret

        # reduce to interval 0.5 < a < 1.5
        if a > 2.0:
            a = a - 2.0
            f = torch.flipud(f)
        if a > 1.5:
            a = a - 1
            f[shft] = torch.fft.fft(f[shft]) / sN
        if a < 0.5:
            a = a + 1
            f[shft] = torch.fft.ifft(f[shft]) * sN

        # the general case for 0.5 < a < 1.5
        alpha = a * math.pi / 2
        tana2 = torch.tan(alpha / 2)
        sina = torch.sin(alpha)
        f = torch.hstack((torch.zeros(N - 1), self.sincinterp(f), torch.zeros(N - 1))).T

        # chirp premultiplication
        chrp = torch.exp(-1j * math.pi / N * tana2 / 4 * torch.arange(-2 * N + 2, 2 * N - 1).T ** 2)
        f = chrp * f

        # chirp convolution
        c = math.pi / N / sina / 4
        ret = torch.tensor(scipy.signal.fftconvolve(torch.exp(1j * c * torch.arange(-(4 * N - 4), 4 * N - 3).T ** 2), f))
        ret = ret[4 * N - 4:8 * N - 7] * torch.sqrt(c / math.pi)

        # chirp post multiplication
        ret = chrp * ret
        # normalizing constant
        ret = torch.exp(-1j * (1 - a) * math.pi / 4) * ret[N - 1:-N + 1:2]
        return ret

    def ifrft(self, f, a):
        return self.frft(f, -a)

    def sincinterp(self, x):
        N = len(x)
        y = torch.zeros(2 * N - 1, dtype=x.dtype)
        y[:2 * N:2] = x
        xint = torch.tensor(scipy.signal.fftconvolve(y[:2 * N], torch.sinc(torch.arange(-(2 * N - 3), (2 * N - 2)).T / 2), ))
        return xint[2 * N - 3: -2 * N + 3]

    def convolve(self,a ,b):
        r = torch.zeros(len(a) + len(b), dtype=torch.complex64).cuda()
        a = dict(enumerate(a))
        v = dict(enumerate(b))
        for i in range(len(a) + len(v) - 1):
            s = 0
            for j in range(len(a)):
                s += a[j] * v.get(i - j, 0)  # 卷积核下标不存在时返回0
            r[i] = s
        return r

    def IFRFT2D(self, matrix, angle=1):
        inter = []
        out = []
        for n in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                matrix1 = matrix[n, c, :, :]
                y = self.ifrft2d(matrix1, [angle, angle])
                inter.append(y)
            inter = torch.stack(inter, dim=0)
            out.append(inter)
            inter = []
        out = torch.stack(out, dim=0)
        return out

    def FRFT2D(self, matrix, angle=1):
        inter = []
        out = []
        for n in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                matrix1 = matrix[n, c, :, :]
                y = self.frft2d(matrix1, [angle, angle])
                inter.append(y)
            inter = torch.stack(inter, dim=0)
            out.append(inter)
            inter = []
        out = torch.stack(out, dim=0)
        return out

    def forward(self, x, channel_reduction=4, angle=0.5):
        angle = torch.tensor(angle)
        alpha = torch.cos(angle * math.pi / 2)
        N, C, H, W = x.shape
        H1 = int(H * alpha)
        W1 = int(W * alpha)

        x1 = self.downsample(x)
        x1 = self.conv1(x1)
        C1 = int(C/channel_reduction)
        C0 = int(C/channel_reduction/3)
        x_0 = x1[:, 0:C0, :, :]
        x_05 = x1[:, C0:C1-C0, :, :]
        x_1 = x1[:, C1-C0:C1, :, :]

        Mask1 = torch.zeros((N, C1-2*C0, H, W)).cuda()
        mask1 = torch.ones((N, C1-2*C0, H1, W1))
        Mask1[:, :, int((H - H1) / 2):int((H - H1) / 2) + H1, int((W - W1) / 2):int((W - W1) / 2) + W1] = mask1
        Mask2 = 1 - Mask1

        Fre = self.FRFT2D(x_05, angle=angle)
        mag = torch.abs(Fre).cuda()
        pha = torch.angle(Fre).cuda()

        mag_s = mag * Mask1
        mag_f = mag * Mask2
        Mag_s = self.mag_s(mag_s)
        Mag_f = self.mag_f(mag_f)
        mag_out = Mag_s * Mask1 + Mag_f * Mask2
        mag_out = self.mag(mag_out)
        pha_out = self.pha(pha)

        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        Fre_out = torch.complex(real, imag)
        IFRFT = self.IFRFT2D(Fre_out, angle=angle)
        IFRFT = torch.abs(IFRFT).cuda()

        fre = torch.fft.rfft2(x_1, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.conv_1(mag)
        pha_out = self.conv_1(pha)
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        x_1 = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        x_0 = self.conv_0(x_0)

        output = torch.cat([x_0, IFRFT, x_1], dim=1)
        output = self.conv2(output)
        return output



class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        base_filter = 64
        num_channels = 7
        out_channels = 4
        self.args = args

        #self.FRFT = FRFT(48, 32, 4)

        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias=False)

        #self.body = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias=False)

        self.body = FRFT(48, 32, 4)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation=None, norm=None, bias=False)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms, b_ms, x_pan):

        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :])).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :])).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f, channel_reduction=4, angle=0.5)
        x_f = self.output_conv(x_f)
        x_f = torch.add(x_f, b_ms)

        return x_f

