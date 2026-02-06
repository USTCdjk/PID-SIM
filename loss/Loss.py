import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np
import torch.fft as fft
from scipy.special import jv
from torchvision import transforms
from torch.autograd import Function

###############Definition for RegLoss#################
#结合了图像重建质量、条纹图、噪声图以及光学参数之间的差异，常用于训练**结构光照明显微成像（SIM）**的图像重建网络，尤其适合联合学习 SIM 图像和光学参数。
class RegLoss(nn.Module):
    def __init__(self, Pred_w=1., Stripe_w=1., Noisy_w=1., Wave_w=1., Phase_w=1., ModFac_w=1.,PSF_w=1.): #设置了7个损失项的权重超参数
        super(RegLoss, self).__init__()
        self.Pred_w=Pred_w #预测图像与 GT 图像的损失权重
        self.Stripe_w=Stripe_w #条纹图像（照明图）与 GT 的差异
        self.Noisy_w=Noisy_w #噪声图像的相似性损失
        self.Wave_w=Wave_w #波矢向量（wave vector）的回归损失
        self.Phase_w=Phase_w #相位（phase shift）损失
        self.ModFac_w=ModFac_w # 调制度（modulation factor）损失
        self.PSF_w=PSF_w #PSF缩放参数的回归损失
        #使用的基础损失函数
        self.Fidloss1 = nn.L1Loss() #图像的绝对误差
        self.SSIMLoss = SSIM() #结构相似性损失（衡量图像结构保真度）
        self.Fidloss2 = nn.MSELoss() #用于参数的回归（波矢、相位、PSF 等）
        self.DistLoss = CosineSimilarityLoss() #用于噪声图像间的角度相似度
    #PredImage: 网络预测的去噪图,GTimg: Ground Truth 图像,StripeImage: 网络预测的条纹图像,gt_StripeImage: Ground Truth 条纹图,NoisyImage, gt_NoisyImage: 模拟图像与 GT 之间的噪声差异
    def forward(self, PredImage, GTimg, StripeImage, gt_StripeImage, NoisyImage, gt_NoisyImage, Optical_para, wave_vector, phase_shifts, ModFac, PSFscale):
        shape = np.shape(PredImage)
        #对应的 Ground Truth 从 Optical_para 中提取并 .cuda() 到 GPU
        gt_wave_vector= Optical_para["wave_vector"].cuda()
        gt_phase_shifts= Optical_para["phase_shifts"].cuda()
        gt_ModFac= Optical_para["ModFac"].cuda()
        gt_PSFscale= Optical_para["PSFscale"].cuda()
        ModFac = ModFac.view(shape[0])
        PSFscale = PSFscale.view(shape[0])
        #print(PSFscale)
        #print(gt_PSFscale)
        #各项损失计算
        loss_wave_vector=self.Fidloss2(wave_vector, gt_wave_vector)
        loss_phase_shifts=self.Fidloss2(phase_shifts, gt_phase_shifts)
        
        loss_PSFscale=self.Fidloss2(PSFscale, gt_PSFscale)
        loss_ModFac=self.Fidloss2(ModFac, gt_ModFac)
        
        loss_pred=self.Fidloss1(PredImage, GTimg)+0.1*torch.abs(1-self.SSIMLoss(PredImage, GTimg)) 

        loss_Stripe=self.Fidloss1(StripeImage, gt_StripeImage)
        loss_Noisy=self.DistLoss(NoisyImage, gt_NoisyImage)
        #loss_Noisy=self.Fidloss1(NoisyImage, gt_NoisyImage)
        #最终损失
        loss= self.Pred_w*loss_pred+self.Stripe_w*loss_Stripe+self.Noisy_w*loss_Noisy+self.Wave_w*loss_wave_vector+self.Phase_w*loss_phase_shifts+self.ModFac_w*loss_ModFac+self.PSF_w*loss_PSFscale
        return loss

###############Definition for RealRegLoss#################
#用于比较预测图像 PredImage 与真实图像 GTimg 的相似性，主要适用于图像重建、去噪或超分辨率任务。
class RealRegLoss(nn.Module):
    #初始化函数 __init__
    def __init__(self, weight=0.1):
        super(RealRegLoss, self).__init__()
        self.weight = weight
        self.Fidloss1 = nn.L1Loss()
        self.SSIMLoss = SSIM()
        
    def forward(self, PredImage, GTimg):
   
        loss = (1-self.weight)*self.Fidloss1(PredImage, GTimg)+self.weight*torch.abs(1-self.SSIMLoss(PredImage, GTimg)) 

        return loss

###############Definition for ReconLoss#################
#让神经网络预测出的图像，在合成结构光照明图像后，与真实输入图像尽可能一致。该损失函数直接将物理成像模型引入到损失中，从而强制网络学到的是“真实结构光照明下”的物理合理图像。
class ReconLoss(nn.Module):
    def __init__(self, NoiseFrac=1.):
        super(ReconLoss, self).__init__()
        self.Fidloss = nn.L1Loss()
        self.NoiseFrac=NoiseFrac
        self.PI=  torch.tensor(3.141592653589793, dtype=torch.float32)
    def forward(self, inputs, pred_img, NoisyImage, Optical_para, wave_vector, phase_shifts, ModFac_w, PSFscale):
        shape = np.shape(inputs)
        # 图像尺寸调整
        resize_transform = transforms.Resize((shape[2], shape[3]))  # 调整大小到原来的一半
        pred_img = resize_transform(pred_img)

        w = shape[2]
        wo = w / 2
        X, Y = Get_X_Y_MeshGrids(w) #获取空间网格
        Nangles= Optical_para["Nangles"]
        Nshifts= Optical_para["Nshifts"]
        meanInten= Optical_para["meanInten"].cuda()
        ampInten= meanInten*ModFac_w
        k2mat = wave_vector
        ps = phase_shifts
        #OTFo= Optical_para["OTF"].cuda()
        #OTFo=OTFo.unsqueeze(1)
        #OTFo=resize_transform(OTFo)

        _, OTFo = PsfOtf(inputs, PSFscale) # 光学传递函数模拟（PSF 卷积）
        sig = pred_img
        for batch_idx in range(shape[0]):
            for i_a in range(Nangles[0]):
                for i_s in range(Nshifts[0]):
                    #生成调制条纹信号（sig）
                    sig[batch_idx,0,:,:] = meanInten[batch_idx,i_a] + ampInten[batch_idx,i_a] * cos_wave(2*self.PI * (k2mat[batch_idx,i_a, 0] * (X - wo) + k2mat[batch_idx,i_a, 1] * (Y - wo))+ ps[batch_idx,i_a, i_s])
        sup_sig = pred_img*sig #叠加调制图案，生成模拟结构光信号
        ST = np.real(ifft2d(fft2d(sup_sig) * fftshift2d(OTFo)))
        #添加噪声，计算损失
        STnoisy = ST + self.NoiseFrac * NoisyImage
        loss = self.Fidloss(inputs, STnoisy)

        return loss


 #将空间域图像转换到频率域，方便进行频域操作      
def fft2d(input):
    fft_out = fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out   

 #对频域图像执行二维傅里叶反变换，恢复到空间域   
def ifft2d(input):
    ifft_out = fft.ifftn(input, dim=(2, 3), norm='ortho')
    return ifft_out

#将傅里叶频谱的零频分量（DC 分量）移动到中心区域。
def fftshift2d(input):
    output = fft.fftshift(input, dim=(2,3))
    return output
#生成结构光条纹图案       
def cos_wave(x):
    return torch.clip(torch.cos(x), 0, 1)
#生成坐标网格
def Get_X_Y_MeshGrids(w):
    x = torch.linspace(0, w - 1, w)
    y = torch.linspace(0, w - 1, w)
    X, Y = torch.meshgrid(x, y)
    return X.cuda(), Y.cuda()


#为一批图像生成与之对应的 点扩散函数（PSF） 和 光学传递函数（OTF），用于模拟成像系统的光学模糊效果
def PsfOtf(x, PSFOTFscale):

        device = x.device
        dtype = x.dtype

        # 获取批次、通道、高度和宽度
        batch_size, channels, height, width = x.shape

        # 生成网格
        Y, X = torch.meshgrid(torch.linspace(0, height - 1, height, device=device, dtype=dtype),
                              torch.linspace(0, width - 1, width, device=device, dtype=dtype))
        R = torch.sqrt(X**2 + Y**2)  # 不扩展维度，因为我们需要在后面进行广播

        # 初始化 PSF 和 OTF 张量
        psf = torch.zeros((batch_size, channels, height, width), device=device, dtype=dtype)
        otf = torch.zeros((batch_size, channels, height, width), device=device, dtype=dtype)

        eps = torch.finfo(dtype).eps

        # 逐批次处理
        for i in range(batch_size):
            scale = PSFOTFscale[i].item()  # 获取该批次的 scale 值
            psf_single = torch.abs(2 * BesselJv.apply(scale * R + eps, 1) / (scale * R + eps))**2 #用一阶贝塞尔函数 J₁(x) 近似构造 Airy Disk（模拟点光源衍射模糊）
            psf_single = fft.fftshift(psf_single)
            psf[i] = psf_single.expand(channels, height, width)

            otf_single = fft.fftn(psf_single, dim=(-2, -1))
            otf_single = otf_single / torch.max(torch.abs(otf_single))
            otf_single = torch.abs(fft.fftshift(otf_single, dim=(-2, -1)))
            otf[i] = otf_single.expand(channels, height, width)

        return psf, otf


#计算 贝塞尔函数（Bessel function of the first kind） 的值及其梯度，以便可以在神经网络训练中正确反向传播
class BesselJv(Function):
    @staticmethod
    def forward(ctx, input, order):
        input_cpu = input.detach().cpu().numpy()
        result = jv(order, input_cpu)
        result = torch.tensor(result, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(input, result)
        ctx.order = order
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, result = ctx.saved_tensors
        order = ctx.order

        epsilon = 1e-6
        input_cpu = input.detach().cpu().numpy()
        grad_input = (jv(order, input_cpu + epsilon) - jv(order, input_cpu - epsilon)) / (2 * epsilon)
        grad_input = torch.tensor(grad_input, dtype=input.dtype, device=input.device)
        grad_input = grad_input * grad_output
        return grad_input, None

################Definition for CosineSimilarityLoss#################
#衡量两个张量之间的相似度差异
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
    def forward(self, x1, x2):

        loss = (1 - F.cosine_similarity(x1, x2))  

        return loss.mean()


################Definition for SpectrumLoss###################
#基于频谱差异的自定义损失函数 SpectrumLoss，用于衡量模型输出与目标在频域（傅里叶变换后）上的差异
class SpectrumLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(SpectrumLoss, self).__init__()
        self.alpha = alpha #alpha 是权重系数，控制实部损失和虚部损失的相对重要性

    def forward(self, output, target):
        # 对模型输出和目标进行二维傅里叶变换
        output_fft = torch.fft.fftn(output, dim=(2, 3), norm='ortho')
        target_fft = torch.fft.fftn(target, dim=(2, 3), norm='ortho')
        output_fft_shifted = torch.fft.fftshift(output_fft,dim=(2, 3))
        target_fft_shifted = torch.fft.fftshift(target_fft,dim=(2, 3))
        #分别提取频谱的实部和虚部。
        output_fft_shifted_real=torch.real(output_fft_shifted)
        output_fft_shifted_imag=torch.imag(output_fft_shifted)
        target_fft_shifted_real=torch.real(target_fft_shifted)
        target_fft_shifted_imag=torch.imag(target_fft_shifted)
        # 分别计算实部和虚部之间的差异作为损失，并加入约束
        real_loss = torch.mean(torch.abs(output_fft_shifted_real - target_fft_shifted_real))
        imag_loss = torch.mean(torch.abs(output_fft_shifted_imag - target_fft_shifted_imag))
        loss = real_loss + self.alpha * imag_loss  # 可以根据需要设置不同的权重
        
        return loss






################Definition for SSIM Loss###################
#衡量两张图像结构相似度的常用指标
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


#生成一维高斯权重向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

#生成一个二维高斯加权窗口，用于多通道图像的SSIM计算
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

#基于滑动高斯窗口的结构相似性指数（SSIM）计算
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




#用于自动生成窗口（卷积核），并将其放到正确的设备和类型上，调用 _ssim 进行计算
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
