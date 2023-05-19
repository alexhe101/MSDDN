import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init
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
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)

def downsample(x,h,w):
    pass
def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)
class Net(nn.Module):
    def __init__(self, num_channels=4,channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        channels = base_filter
        self.fuse1 = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.fuse2 = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.fuse3 = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.fuse4 = nn.Sequential(nn.Conv2d(2*channels,channels,3,1,1),nn.ReLU(),nn.Conv2d(channels,channels,1,1,0))
        self.fuse5 = nn.Sequential(nn.Conv2d(2*channels,channels,3,1,1),nn.ReLU(),nn.Conv2d(channels,channels,1,1,0))
        self.fuse6 = nn.Sequential(nn.Conv2d(2*channels,channels,3,1,1),nn.ReLU(),nn.Conv2d(channels,channels,1,1,0))
        self.msconv = nn.Conv2d(4,channels,3,1,1)# conv for ms
        self.panconv = nn.Conv2d(1,channels,3,1,1)
        self.conv0 = HinResBlock(channels,channels)
        self.conv1 = HinResBlock(channels,channels)
        self.conv2 = HinResBlock(channels,channels)
        self.conv_ada = nn.Conv2d(2*channels,channels,3,1,1)
        self.fft1 = Freprocess(channels)
        self.fft2 = Freprocess(channels)
        self.fft3 = Freprocess(channels)
        self.conv_out = Refine(channels,4)
        self.down_pan1 = nn.Conv2d(channels,channels,3,2,1)
        self.down_spa1 = nn.Conv2d(channels,channels,3,2,1)
        self.down_spa2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_pan2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_pan3 = nn.Conv2d(channels, channels, 3, 2, 1)
        # self.process = FeatureProcess(channels)
        # self.cdc = nn.Sequential(nn.Conv2d(1, 4, 1, 1, 0), cdcconv(4, 4), cdcconv(4, 4))
        # self.refine = Refine(channels, 4)

    def forward(self, ms,_,pan):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        seq = []
        mHR = upsample(ms, M, N) # size 4
        ms = mHR
        msf = self.msconv(ms) #(4->channels) # size 4
        panf = self.panconv(pan) #(1->channels)
        seq.append(panf) # for fft size 4
        panf_2  =self.conv1(panf) #(panf channels->channels)
        seq.append(panf_2) # for fft size 4
        spa_fuse = self.fuse1(torch.cat([msf, panf], 1)) # concat msf&panf to inn block (2*channels -> channels)

        '''downsample spa_fuse and panf2'''
        M = M//2
        N = N//2
        d_spa = self.down_spa1(spa_fuse)#downsample(spa_fuse,M,N) # size 2
        d_panf = self.down_pan1(panf_2) #downsample(panf_2,M,N) # size 2
        spa_fuse = self.fuse2(torch.cat([d_spa, d_panf], 1)) # downsampled features into inn block (2*channels-> channels)

        ''' downsample spa_fuse and d_panf'''
        M = M//2
        N = N//2
        d_spa = self.down_spa2(spa_fuse)
        panf_3 = self.conv0(d_panf) # size 2
        seq.append(panf_3)
        d_panf = self.down_pan2(panf_3)#downsample(panf_3, M, N)
        spa_fuse = self.fuse3(torch.cat([d_spa, d_panf], 1))

        '''upsample and do fft'''
        M*=2
        N*=2
        spa_fuse = upsample(spa_fuse,M,N)
        fft_out = self.fft1(spa_fuse,seq[-1]) #(channels)
        t = self.fuse4(torch.cat([fft_out,spa_fuse],1)) #(2*channels -> channels)
        t+=spa_fuse


        spa_fuse = t
        M*=2
        N*=2
        spa_fuse = upsample(spa_fuse, M, N)
        fft_out = self.fft2(spa_fuse,seq[-2])
        t = self.fuse5(torch.cat([fft_out,spa_fuse],1))
        t += spa_fuse

        spa_fuse = t
        fft_out = self.fft3(spa_fuse, seq[-3])
        t = self.fuse6(torch.cat([fft_out, spa_fuse], 1))
        t= self.conv_out(t) #channels to 4
        HR = t+ms
        return HR

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3





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
