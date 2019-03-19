# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license, 
# visit https://opensource.org/licenses/MIT.

"""Models of Spatial Attention Generative Adversarial Network"""

import torch
import torch.nn as nn


def get_norm(name, nc):
    if name == 'batchnorm':
        return nn.BatchNorm2d(nc)
    if name == 'instancenorm':
        return nn.InstanceNorm2d(nc)
    raise ValueError('Unsupported normalization layer: {:s}'.format(name))

def get_nonlinear(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'lrelu':
        return nn.LeakyReLU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError('Unsupported activation layer: {:s}'.format(name))

class ResBlk(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResBlk, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, 3, 1, 1),
            get_norm('batchnorm', n_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 3, 1, 1),
            get_norm('batchnorm', n_out),
        )
    
    def forward(self, x):
        return self.layers(x)

class _Generator(nn.Module):
    def __init__(self, input_channels, output_channels, last_nonlinear):
        super(_Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, 1, 3),  # n_in, n_out, kernel_size, stride, padding
            get_norm('instancenorm', 32),
            get_nonlinear('relu'),
            nn.Conv2d(32, 64, 4, 2, 1),
            get_norm('instancenorm', 64),
            get_nonlinear('relu'),
            nn.Conv2d(64, 128, 4, 2, 1),
            get_norm('instancenorm', 128),
            get_nonlinear('relu'),
            nn.Conv2d(128, 256, 4, 2, 1),
            get_norm('instancenorm', 256),
            get_nonlinear('relu'),
        )
        self.resblk = nn.Sequential(
            ResBlk(256, 256),
            ResBlk(256, 256),
            ResBlk(256, 256),
            ResBlk(256, 256),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            get_norm('instancenorm', 128),
            get_nonlinear('relu'),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            get_norm('instancenorm', 64),
            get_nonlinear('relu'),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            get_norm('instancenorm', 32),
            get_nonlinear('relu'),
            nn.ConvTranspose2d(32, output_channels, 7, 1, 3),
            get_nonlinear(last_nonlinear),
        )
    
    def forward(self, x, a=None):
        if a is not None:
            assert len(a.size()) == 2 and x.size(0) == a.size(0)
            a = a.type(x.dtype)
            a = a.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat((x, a), dim=1)
        h = self.conv(x)
        h = self.resblk(h)
        y = self.deconv(h)
        return y

class Generator(nn.Module):
    def __init__(self, input_channels):
        super(Generator, self).__init__()
        self.AMN = _Generator(input_channels + 1, input_channels, 'tanh')
        self.SAN = _Generator(input_channels, 1, 'sigmoid')
    def forward(self, x, a):
        y = self.AMN(x, a)
        m = self.SAN(x)
        y_ = y * m + x * (1-m)
        return y_, m

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(32, 64, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(64, 128, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(128, 256, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(256, 512, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(512, 1024, 4, 2, 1),
            get_nonlinear('lrelu'),
        )
        self.src = nn.Conv2d(1024, 1, 3, 1, 1)
        self.cls = nn.Sequential(
            nn.Conv2d(1024, 1, 2, 1, 0),
            get_nonlinear('sigmoid'),
        )
    
    def forward(self, x):
        h = self.conv(x)
        return self.src(h), self.cls(h).squeeze().unsqueeze(1)

if __name__ == '__main__':
    from torchsummary import summary
    AMN = Generator(4, 3, 'tanh')
    summary(AMN, (4, 128, 128), device='cpu')
    SAN = Generator(3, 1, 'sigmoid')
    summary(SAN, (3, 128, 128), device='cpu')
    D = Discriminator(3)
    summary(D, (3, 128, 128), device='cpu')