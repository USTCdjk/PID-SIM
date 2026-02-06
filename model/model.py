import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_block import conv, fft2d, fftshift2d, Upsampler
import numpy as np
from numpy import pi

# 通道注意力机制模块
class CALayer(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# 改进的融合频谱特征的通道注意力模块，仅保留振幅注意力并添加高频掩码
class ACALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU(), high_freq_ratio=0.1):
        super().__init__()
        self.amplitude_conv = conv(n_feat, n_feat, kernel_size)
        self.act = act
        self.global_average_pooling2d = nn.AdaptiveAvgPool2d(1)
        self.global_max_pooling2d = nn.AdaptiveMaxPool2d(1)
        self.amplitude_attention = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Sigmoid()
        )
        self.high_freq_ratio = high_freq_ratio  # 低频区域比例，中心区域被掩码掉
        
    def create_high_freq_mask(self, x):
        """创建频域高频掩码，过滤掉中心低频区域"""
        b, c, h, w = x.shape
        mask = torch.ones_like(x)
        
        # 计算低频区域半径
        h_radius = int(h * self.high_freq_ratio / 2)
        w_radius = int(w * self.high_freq_ratio / 2)
        
        # 频域中心坐标
        h_center = h // 2
        w_center = w // 2
        
        # 掩码掉中心低频区域
        mask[..., h_center - h_radius : h_center + h_radius, 
             w_center - w_radius : w_center + w_radius] = 0.0
        return mask

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = fft2d(x)
        x_amplitude = torch.abs(x_fft)
        x_amplitude = torch.pow(x_amplitude + 1e-8, 0.8)
        amplitude1 = fftshift2d(x_amplitude)
        
        # 应用高频掩码，只关注高频区域
        high_freq_mask = self.create_high_freq_mask(amplitude1)
        amplitude1 = amplitude1 * high_freq_mask
        
        amplitude2 = self.act(self.amplitude_conv(amplitude1))
        y_avg = self.global_average_pooling2d(amplitude2).view(b, c, 1, 1)
        y_max = self.global_max_pooling2d(amplitude2).view(b, c, 1, 1)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.amplitude_attention(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y

# 改进的融合振幅注意力和轻量级相位补偿的模块
class APCALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU(), high_freq_ratio=0.1):
        super().__init__()
        self.act = act
        self.conv = conv(n_feat, n_feat, 1)
        
        # 仅保留振幅注意力
        self.amplitude_attention = ACALayer(conv, n_feat, kernel_size, reduction, high_freq_ratio=high_freq_ratio)
        
        # 轻量级相位补偿模块（替代原有的PCALayer）
        # 修复：调整卷积层输入通道数为n_feat
        self.phase_compensation = nn.Sequential(
            conv(n_feat, n_feat // reduction, kernel_size=1),  # 输入通道数改为n_feat
            nn.ReLU(),
            conv(n_feat // reduction, n_feat, kernel_size=1),
            nn.Tanh()  # 相位补偿通常在[-1,1]范围内
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            conv(n_feat, n_feat // reduction, kernel_size=1),
            nn.ReLU(),
            conv(n_feat // reduction, 2, kernel_size=1)  # 2个权重，分别对应振幅注意力和相位补偿
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 振幅注意力
        amp_att = self.amplitude_attention(x)
        
        # 轻量级相位补偿 - 修复维度问题
        x_fft = fft2d(x)
        x_phase = torch.atan2(x_fft.imag + 1e-8, x_fft.real + 1e-8)  # 形状: [b, c, h, w]
        phase1 = fftshift2d(x_phase)  # 形状: [b, c, h, w]
        
        # 移除不必要的维度扩展，直接使用相位信息
        phase_comp = self.phase_compensation(phase1)  # 现在输入是4D: [b, c, h, w]
        phase_att = x * (1 + phase_comp)  # 应用相位补偿
        
        # 学习融合权重
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.weight(y)
        weights = F.softmax(y, dim=1)
        
        # 融合振幅注意力和相位补偿
        output = amp_att * weights[:, 0:1, ...] + phase_att * weights[:, 1:2, ...]
        
        output = self.act(self.conv(output))
        output = output + x  # 残差连接
        return output

# 残差增强模块
class APCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), high_freq_ratio=0.1):
        super(APCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        # 使用改进的APCALayer
        modules_body.append(APCALayer(conv, n_feat, kernel_size, reduction, act=act, high_freq_ratio=high_freq_ratio))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

# 残差组模块
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks, high_freq_ratio=0.1):
        super(ResidualGroup, self).__init__()
        modules_body = [
            APCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, high_freq_ratio=high_freq_ratio) 
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

# 超分辨率重建模块
class SuperResolvedImage_prediction(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, reduction=16, act=nn.ReLU(), n_resblocks=3):
        super(SuperResolvedImage_prediction, self).__init__()

        self.body = ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks)
        self.Upsampler = Upsampler(n_feats, n_feats)
        self.pred = conv(in_channels=n_feats, out_channels=1, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        res = self.body(x)
        res = res + x
        shape = res.shape
        res = self.Upsampler(res, shape)
        recon_image = self.pred(res)
        return recon_image

# 噪声预测模块
class Noise_prediction(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, reduction=16, act=nn.ReLU(), n_resblocks=2):
        super(Noise_prediction, self).__init__()

        self.body = ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks)
        self.pred = conv(in_channels=n_feats, out_channels=9, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        res = self.body(x)
        res = res + x
        noisy_image = self.pred(res)
        return noisy_image

# 条纹伪影预测模块
class Stripe_prediction(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, reduction=16, act=nn.ReLU(), n_resblocks=2):
        super(Stripe_prediction, self).__init__()
        self.ps = torch.zeros(3,3)  # 0pt.Nangles, opt.Nshifts
 
        self.body = ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks)
        
        self.conv1 = conv(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=False)
        self.norm1 = nn.InstanceNorm2d(n_feats)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = conv(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=False)
        self.norm2 = nn.InstanceNorm2d(n_feats)
        self.act2 = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(n_feats, 128)
        self.pred_map = conv(in_channels=n_feats, out_channels=9, kernel_size=kernel_size, bias=False)
        self.pred_wave_vector = nn.Linear(128, 3*2)  # shape:(opt.Nangles,2)
        self.pred_phase_shifts = nn.Linear(128, 3*3)  # shape:(opt.Nangles,opt.Nshifts)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 输入与特征提取
        res = self.body(x)
        res = res + x
        
        # 条纹图预测分支
        stripe_image = self.pred_map(res)
        
        # 条纹频率与相位预测分支
        fea = self.pool1(self.act1(self.norm1(self.conv1(res))))
        fea = self.global_avg_pool(self.act2(self.norm2(self.conv2(fea))))
        fea = self.fc1(fea.view(fea.size(0), -1))
        
        # 波矢预测
        wave_vector = self.pred_wave_vector(fea)
        wave_vector = (2 * self.sigmoid(wave_vector) - 1) * 180 / 512
        
        # 相位预测
        phase_shifts = self.pred_phase_shifts(fea)
        phase_shifts = (self.sigmoid(phase_shifts) * 2 - 0.075) * torch.tensor(3.141592653589793, dtype=torch.float32)
        
        return stripe_image, wave_vector, phase_shifts

# 调制因子预测模块
class ModFac_prediction(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, reduction=16, act=nn.ReLU(), n_resblocks=2):
        super(ModFac_prediction, self).__init__()

        self.body = ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks)
        self.conv1 = conv(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=False)
        self.norm1 = nn.InstanceNorm2d(n_feats)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = conv(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=False)
        self.norm2 = nn.InstanceNorm2d(n_feats)
        self.act2 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(n_feats, 128)
        self.pred_ModFac = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        res = self.body(x)
        res = res + x
        
        fea = self.pool1(self.act1(self.norm1(self.conv1(res))))
        fea = self.global_avg_pool(self.act2(self.norm2(self.conv2(fea))))
        fea = self.fc1(fea.view(fea.size(0), -1))
        
        ModFac = self.pred_ModFac(fea)
        ModFac = (2 * self.sigmoid(ModFac) - 1) * 0.15 + 0.5
        
        return ModFac

# PSF预测模块
class PSF_prediction(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, reduction=16, act=nn.ReLU(), n_resblocks=2):
        super(PSF_prediction, self).__init__()

        self.body = ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks)
        self.conv1 = conv(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=False)
        self.norm1 = nn.InstanceNorm2d(n_feats)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = conv(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=False)
        self.norm2 = nn.InstanceNorm2d(n_feats)
        self.act2 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(n_feats, 128)
        self.pred_PSF_scale = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        res = self.body(x)
        res = res + x
        
        fea = self.pool1(self.act1(self.norm1(self.conv1(res))))
        fea = self.global_avg_pool(self.act2(self.norm2(self.conv2(fea))))
        fea = self.fc1(fea.view(fea.size(0), -1))
        
        PSF_scale = self.pred_PSF_scale(fea)
        PSF_scale = (2 * self.sigmoid(PSF_scale) - 1) * 0.2 + 1
        
        return PSF_scale

# 主模型
class PID(nn.Module):
    def __init__(self):
        super(PID, self).__init__()
        n_resgroups = 4
        n_resblocks = 4
        n_feats = 64
        kernel_size = 3
        reduction = 16
        act = nn.ReLU()
        high_freq_ratio = 0.1  # 低频区域比例，中心10%的区域被掩码
        
        modules_head = [conv(9, n_feats, kernel_size)]
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, 
                n_resblocks=n_resblocks, high_freq_ratio=high_freq_ratio) 
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        self.pred_recon_image = SuperResolvedImage_prediction(n_feats)
        self.pred_noisy_image = Noise_prediction(n_feats)
        self.pred_stripe_image = Stripe_prediction(n_feats)
        self.pred_ModFac = ModFac_prediction(n_feats)
        self.pred_PSF = PSF_prediction(n_feats)

    def forward(self, x):
        x = self.head(x[:, 0:9, :, :])
        res = self.body(x)
        res = res + x
        
        recon_image = self.pred_recon_image(res)
        noisy_image = self.pred_noisy_image(res)
        stripe_image, wave_vector, phase_shifts = self.pred_stripe_image(res)
        ModFac = self.pred_ModFac(res)
        PSFscale = self.pred_PSF(res)

        return recon_image, noisy_image, stripe_image, wave_vector, phase_shifts, ModFac, PSFscale
    