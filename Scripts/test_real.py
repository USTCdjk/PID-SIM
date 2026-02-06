import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import cv2
import tifffile
from torch.utils.data import DataLoader

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse

from argparse import ArgumentParser
from model.model import PID
from model.APCAN_1 import APCAN
from model.DFCAN16 import DFCAN
from util.util import img_comp

def test(args, test_loader):
    if args.model_name == 'APCAN':
        print("Training model: APCAN")
        model = APCAN().cuda()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {num_params:,} 个 ({num_params / 1e6:.2f} M)")
    elif args.model_name == 'DFCAN':
        print("Training model: DFCAN")
        model = DFCAN().cuda()
    elif args.model_name == 'PID':
        print("Training model: PID")
        model = PID().cuda()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {num_params:,} 个 ({num_params / 1e6:.2f} M)")
    else:
        raise ValueError(f"Error: the model is not implemented!!!!")
    
    checkpoint = torch.load(args.chinkpoint_for_test)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("start testing>>>>>>>>>") 
    # 创建Output文件夹（确保存在）
    if not os.path.exists('Output'):
        os.makedirs('Output')
    # 初始化指标日志文件
    metrics_log_path = os.path.join('Output_DT', 'image_metrics.log')
    with open(metrics_log_path, 'w') as f:
        f.write("图像序号\tPSNR\tSSIM\tNRMSE\tMSE\n")
        f.write("-" * 50 + "\n")
    
    # 初始化参数输出文件（清空原有内容）
    params_log_path = os.path.join('Output', 'model_params.log')
    with open(params_log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("模型输出参数日志 (wave_vector/phase_shifts/PSFscale/ModFac)\n")
        f.write("=" * 80 + "\n\n")
    
    psnr_val, ssim_val, mse_val, nrmses_val = evaluation(args, test_loader, model, metrics_log_path, params_log_path)
    print(f'\n最终平均 - Eval_PSNR: {psnr_val:.4f}, Eval_SSIM: {ssim_val:.4f}, Eval_mse: {mse_val:.4f}, Eval_nrmses: {nrmses_val:.4f}')
    # 在日志文件末尾添加平均值
    with open(metrics_log_path, 'a') as f:
        f.write("-" * 50 + "\n")
        f.write(f"平均值\t{psnr_val:.4f}\t{ssim_val:.4f}\t{nrmses_val:.4f}\t{mse_val:.4f}\n")
                
def evaluation(args, test_loader, model, metrics_log_path, params_log_path):
    model.eval()
    eval_loss = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    nrmses_sum = 0.0
    
    for eval_idx, (input_images, gt_images, norm_inputs, norm_gt) in enumerate(test_loader):
        val_input = norm_inputs.cuda()
        val_gt = norm_gt.cuda()

        with torch.no_grad():
            recon_image, noisy_image, stripe_image, wave_vector, phase_shifts, ModFac, PSFscale = model(val_input)
        
        val_output = recon_image
        val_output = val_output.cpu().detach().numpy().transpose(0, 2, 3, 1)
        val_output = np.clip(val_output, 0, 1)
        val_gt = val_gt.cpu().numpy().transpose(0, 2, 3, 1)
        val_input = val_input.cpu().numpy().transpose(0, 2, 3, 1)
        stripe_image = stripe_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        noisy_image = noisy_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        
        # 核心修改：将Tensor转为numpy数组（彻底脱离Tensor类型）
        wave_vector = wave_vector.cpu().detach().numpy()  # 转numpy
        phase_shifts = phase_shifts.cpu().detach().numpy()  # 转numpy
        ModFac = ModFac.cpu().detach().numpy()  # 转numpy
        PSFscale = PSFscale.cpu().detach().numpy()  # 转numpy
        
        # 调整参数维度（匹配batch维度）
        shape = recon_image.shape  # [batch, channel, H, W]
        wave_vector = wave_vector.reshape(shape[0], 3, 2)  # [batch, 3, 2]
        phase_shifts = phase_shifts.reshape(shape[0], 3, 3)  # [batch, 3, 3]
        # PSFscale = PSFscale.reshape(shape[0])  # [batch] 替换view，用numpy的reshape
        ModFac = ModFac.reshape(shape[0])  # [batch] 替换view，用numpy的reshape
        
        # 逐张计算指标并保存
        for batch_idx, (inp, out) in enumerate(zip(val_output, val_gt)):
            # 计算指标（注意：img_comp可能返回单个值或元组，这里按原逻辑）
            psnrs, ssims, mses, nrmses = img_comp(inp, out)
            
            # 累加用于计算平均值
            psnr_sum += psnrs
            ssim_sum += ssims
            mse_sum += mses
            nrmses_sum += nrmses
            
            # 图像全局序号（处理batch_size>1的情况）
            global_img_idx = eval_idx * test_loader.batch_size + batch_idx + 1
            
            # 打印当前图像的指标
            print(f"\n===== 图像 {global_img_idx:03d} 指标 =====")
            print(f"PSNR: {psnrs:.4f}, SSIM: {ssims:.4f}, NRMSE: {nrmses:.4f}, MSE: {mses:.4f}")
            
            # 打印当前图像的模型参数（现在是numpy标量，可正常格式化）
            print(f"\n===== 图像 {global_img_idx:03d} 模型参数 =====")
            print(f"wave_vector (3x2):")
            for i in range(3):
                print(f"  [{wave_vector[batch_idx, i, 0]:.6f}, {wave_vector[batch_idx, i, 1]:.6f}]")
            
            print(f"\nphase_shifts (3x3):")
            for i in range(3):
                print(f"  [{phase_shifts[batch_idx, i, 0]:.6f}, {phase_shifts[batch_idx, i, 1]:.6f}, {phase_shifts[batch_idx, i, 2]:.6f}]")
            
            # print(f"\nPSFscale: {PSFscale[batch_idx]:.6f}")
            print(f"ModFac: {ModFac[batch_idx]:.6f}")
            print("-" * 50)
            
            # 保存指标到日志文件
            with open(metrics_log_path, 'a') as f:
                f.write(f"{global_img_idx:03d}\t{psnrs:.4f}\t{ssims:.4f}\t{nrmses:.4f}\t{mses:.4f}\n")
            
            # 保存参数到日志文件
            with open(params_log_path, 'a') as f:
                f.write(f"===== 图像 {global_img_idx:03d} 参数 =====\n")
                f.write("wave_vector (3x2):\n")
                for i in range(3):
                    f.write(f"  {wave_vector[batch_idx, i, 0]:.6f}\t{wave_vector[batch_idx, i, 1]:.6f}\n")
                
                f.write("\nphase_shifts (3x3):\n")
                for i in range(3):
                    f.write(f"  {phase_shifts[batch_idx, i, 0]:.6f}\t{phase_shifts[batch_idx, i, 1]:.6f}\t{phase_shifts[batch_idx, i, 2]:.6f}\n")
                
                # f.write(f"\nPSFscale: {PSFscale[batch_idx]:.6f}\n")
                f.write(f"ModFac: {ModFac[batch_idx]:.6f}\n")
                f.write("-" * 80 + "\n\n")
        
        # 保存图像（保持原逻辑，若batch_size>1会覆盖，可根据需要修改）
        save_images(val_gt, eval_idx, 'Output_DT/gt/') 
        save_images(val_input[:, :, :, 0:1], eval_idx, 'Output_DT/input/') 
        save_images(val_output, eval_idx, 'Output_DT/SR/')  
        save_images(noisy_image[:,:,:,0:1], eval_idx,'Output/NoisyImage/')   
        save_images(stripe_image[:,:,:,0:1], eval_idx,'Output/StripeImage/')
 
    # 计算平均值（注意：len(test_loader)是batch数，需乘以batch_size得到总图像数，若最后一个batch不足则自动适配）
    total_img_num = len(test_loader.dataset)
    psnr_val = psnr_sum / total_img_num
    ssim_val = ssim_sum / total_img_num
    mse_val = mse_sum / total_img_num
    nrmses_val = nrmses_sum / total_img_num
    
    return psnr_val, ssim_val, mse_val, nrmses_val

def save_images(output_image, eval_idx, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 若batch_size>1，这里只保存batch中第一张图，可修改为保存所有图
    output_image = output_image[0, :, :, 0] * 65535
    output_image = output_image.astype('uint16')
    formatted_number = "{:03d}".format(eval_idx)
    tifffile.imsave(os.path.join(output_path, f"{formatted_number}.tif"), output_image)