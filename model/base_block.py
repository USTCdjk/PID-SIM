import math
import torch
import torchvision
import torch.nn as nn
import torch.fft as fft
import PIL.Image as Image
import torch.nn.functional as F
import numpy as np
#对输入图像（通常是 [B, C, H, W]）进行二维快速傅里叶变换（FFT）
def fft2d(input):
    fft_out = fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out

#将低频信息从图像的角落移动到中心（频谱居中），用于频域可视化或进一步处理
def fftshift2d(input):
    output = fft.fftshift(input, dim=(2,3))
    '''b, c, h, w = input.shape
    fs11 = input[:, :, -h // 2:h, -w // 2:w]
    fs12 = input[:, :, -h // 2:h, 0:w // 2]
    fs21 = input[:, :, 0:h // 2, -w // 2:w]
    fs22 = input[:, :, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    output = torchvision.transforms.Resize((128, 128), interpolation=Image.BICUBIC)(output)'''
    return output
 #用于封装 nn.Conv2d，自动设置合适的 padding 以保持特征图的尺寸   
def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

#残差模块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding= (kernel_size // 2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

#非残差的双卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding= (kernel_size // 2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


#将图像空间信息重新排列到通道维度（channel）
def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k

    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)
#子像素下采样
class Downsampler(nn.Module):
    def __init__(self, factor):
        super(Downsampler, self).__init__()
        self.factor = factor
    def forward(self, input):
     return pixel_unshuffle(input, self.factor)


#子像素上采样
class Upsampler(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(Upsampler, self).__init__()
        self.conv = nn.Conv2d(filter_in, filter_out*4 , (1,1), 1, 0, groups=groups, bias=True)
        self.rule = nn.ReLU(inplace=True)

     def forward(self, input, shape):
        N,C,H,W=shape
        out = self.rule(self.conv(input))
        out = F.pixel_shuffle(out,2)
        return out



