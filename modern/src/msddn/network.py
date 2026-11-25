"""Checkpoint-compatible MSDDN. Derived from the published MSDDN implementation at 834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19. Parameter names and active operations are retained."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import HinResBlock, InvBlock
from .frequency import Freprocess
from .refine import Refine

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

    def forward(self, ms, _, pan):
        h, w = pan.shape[-2:]
        ms = upsample(ms, h, w)
        msf = self.msconv(ms)
        pan_full = self.panconv(pan)
        pan_full_guidance = self.conv1(pan_full)
        spatial = self.fuse1(torch.cat([msf, pan_full], 1))

        half_h, half_w = h // 2, w // 2
        pan_half = self.down_pan1(pan_full_guidance)
        spatial = self.fuse2(torch.cat([self.down_spa1(spatial), pan_half], 1))
        # Preserve evaluation order: spatial downsample precedes PAN conv0.
        spatial_down = self.down_spa2(spatial)
        pan_half_guidance = self.conv0(pan_half)
        spatial = self.fuse3(torch.cat([spatial_down, self.down_pan2(pan_half_guidance)], 1))

        # Use the same floor / multiply size arithmetic as the released code.
        h, w = (half_h // 2) * 2, (half_w // 2) * 2
        spatial = upsample(spatial, h, w)
        fft_out = self.fft1(spatial, pan_half_guidance)
        t = self.fuse4(torch.cat([fft_out, spatial], 1))
        t += spatial
        spatial = upsample(t, h * 2, w * 2)
        fft_out = self.fft2(spatial, pan_full_guidance)
        t = self.fuse5(torch.cat([fft_out, spatial], 1))
        t += spatial
        spatial = t
        fft_out = self.fft3(spatial, pan_full)
        t = self.fuse6(torch.cat([fft_out, spatial], 1))
        return self.conv_out(t) + ms
