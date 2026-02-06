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
from util.util import img_comp

def test(args, test_loader):
    model = PID().cuda() #创建模型并移动到GPU
    checkpoint = torch.load(args.chinkpoint_for_test) #加载训练好的模型权重
    model.load_state_dict(checkpoint['model_state_dict']) #加载模型参数
    print("start testing>>>>>>>>>") 
    psnr_val, ssim_val, mse_val, nrmses_val=evaluation(args, test_loader, model) #调用evaluation进行评估
    print(f'Eval_PSNR: {psnr_val:.4f}, Eval_SSIM: {ssim_val:.4f}, Eval_mse: {mse_val:.4f}, Eval_nrmses: {nrmses_val:.4f}') #打印评估指标：PSNR、SSIM、MSE、NRMSE

#遍历测试数据，调用模型进行预测，计算指标，保存图像，并将预测参数及误差写到output.txt                
def evaluation(args, test_loader, model):
    model.eval() #模型切换到评估模式
    eval_loss = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum =0.0
    nrmses_sum = 0.0
    #遍历test_loader中的数据
    for eval_idx,(gt_img, Syn_input, Stripes_sig, Noise_sig, Optical_para) in enumerate(test_loader):
        #将输入和gt转到GPU
        val_input=Syn_input.cuda()
        val_gt=gt_img.cuda()
        Stripes_sig=Stripes_sig.cuda()
        #with torch.no_grad()下调用模型，得到预测结果和其他输出
        with torch.no_grad():
             recon_image, noisy_image, stripe_image, wave_vector, phase_shifts, ModFac, PSFscale=model(val_input)
        val_output = recon_image
        #将tensor转成numpy数组，并做维度变换（一般是NCHW -> NHWC）
        val_output = val_output.cpu().detach().numpy().transpose(0, 2, 3, 1)
        val_output = np.clip(val_output,0,1)
        val_gt = val_gt.cpu().numpy().transpose(0, 2, 3, 1)
        val_input = val_input.cpu().numpy().transpose(0, 2, 3, 1)
        stripe_image = stripe_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        noisy_image = noisy_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        #norm_input= norm_input.numpy().transpose(0, 2, 3, 1)
        #使用自定义的img_comp函数计算每张图的PSNR、SSIM、MSE、NRMSE指标，并累加
        for inp, out in zip(val_output, val_gt):
            psnrs, ssims, mses, nrmses = img_comp(inp, out)
            psnr_sum += psnrs
            ssim_sum += ssims
            mse_sum += mses
            nrmses_sum += nrmses
        #保存多种结果图像（GT，输入，输出，噪声图，条纹图）
        save_images(val_gt, eval_idx,'Output/gt/') 
        save_images(val_input[:,:,:,0:1], eval_idx,'Output/input/') 
        save_images(val_output, eval_idx,args.results_path)  
        save_images(noisy_image[:,:,:,0:1], eval_idx,'Output/NoisyImage/')   
        save_images(stripe_image[:,:,:,0:1], eval_idx,'Output/StripeImage/')
        
        #处理光学参数和对应的误差，写入output.txt
        gt_wave_vector = Optical_para["wave_vector"]
        gt_phase_shifts= Optical_para["phase_shifts"]
        gt_PSFscale= Optical_para["PSFscale"]
        gt_ModFac= Optical_para["ModFac"]
        
        shape=np.shape(recon_image)
        wave_vector = torch.reshape(wave_vector, (shape[0], 3,2))
        phase_shifts = torch.reshape(phase_shifts, (shape[0], 3,3))
        ModFac = ModFac.view(shape[0])
        PSFscale = PSFscale.view(shape[0])

        
        ErrorMap_wave_vector=torch.abs(gt_wave_vector.cuda()-wave_vector)/(torch.abs(gt_wave_vector.cuda())+1e-6)
        ErrorMap_phase_shifts=torch.abs(gt_phase_shifts.cuda()-phase_shifts)/(torch.abs(gt_phase_shifts.cuda())+1e-6)
        ErrorMap_PSFscale=torch.abs(gt_PSFscale.cuda()-PSFscale)/(torch.abs(gt_PSFscale.cuda())+1e-6)
        ErrorMap_ModFac=torch.abs(gt_ModFac.cuda()-ModFac)/(torch.abs(gt_ModFac.cuda())+1e-6)
        with open('output.txt', 'a') as f:
             f.write("gt_wave_vector:\n")
             for i in range(gt_wave_vector.shape[0]):
                 #f.write(f"Slice {i}:\n")
                 for j in range(gt_wave_vector.shape[1]):
                     line = ' '.join(f"{gt_wave_vector[i, j, k].item()}" for k in range(gt_wave_vector.shape[2]))
                     f.write(line + "\n")
                     
             f.write("wave_vector:\n")        
             for i in range(wave_vector.shape[0]):
                 #f.write(f"Slice {i}:\n")
                 for j in range(wave_vector.shape[1]):
                     line = ' '.join(f"{wave_vector[i, j, k].item()}" for k in range(wave_vector.shape[2]))
                     f.write(line + "\n")

             f.write("ErrorMap_wave_vector:\n")        
             for i in range(ErrorMap_wave_vector.shape[0]):
                 #f.write(f"Slice {i}:\n")
                 for j in range(ErrorMap_wave_vector.shape[1]):
                     line = ' '.join(f"{ErrorMap_wave_vector[i, j, k].item()}" for k in range(ErrorMap_wave_vector.shape[2]))
                     f.write(line + "\n")
     
             f.write("gt_phase_shifts:\n")         
             for i in range(gt_phase_shifts.shape[0]):
                 #f.write(f"Slice {i}:\n")
                 for j in range(gt_phase_shifts.shape[1]):
                     line = ' '.join(f"{gt_phase_shifts[i, j, k].item()}" for k in range(gt_phase_shifts.shape[2]))
                     f.write(line + "\n")
                     
             f.write("phase_shifts:\n")        
             for i in range(phase_shifts.shape[0]):
                 #f.write(f"Slice {i}:\n")
                 for j in range(phase_shifts.shape[1]):
                     line = ' '.join(f"{phase_shifts[i, j, k].item()}" for k in range(phase_shifts.shape[2]))
                     f.write(line + "\n")

             f.write("ErrorMap_phase_shifts:\n")        
             for i in range(ErrorMap_phase_shifts.shape[0]):
                 #f.write(f"Slice {i}:\n")
                 for j in range(ErrorMap_phase_shifts.shape[1]):
                     line = ' '.join(f"{ErrorMap_phase_shifts[i, j, k].item()}" for k in range(ErrorMap_phase_shifts.shape[2]))
                     f.write(line + "\n")
      
             f.write(f"gt_PSFscale: {gt_PSFscale.item()}\n")

             f.write(f"PSFscale: {PSFscale.item()}\n")
             
             f.write(f"ErrorMap_PSFscale: {ErrorMap_PSFscale.item()}\n")
             
             f.write(f"gt_ModFac: {gt_ModFac.item()}\n")

             f.write(f"ModFac: {ModFac.item()}\n")
             
             f.write(f"ErrorMap_ModFac: {ErrorMap_ModFac.item()}\n")
             
             f.write("\n")
        
        #save_images(val_input[:,:,:,0:1], eval_idx,'Output/results2/')  
        '''save_images(val_input[:,:,:,1:2], eval_idx+100,'Output/results2/')  
        save_images(val_input[:,:,:,2:3], eval_idx+200,'Output/results2/')  
        save_images(val_input[:,:,:,3:4], eval_idx+300,'Output/results2/')  
        save_images(val_input[:,:,:,4:5], eval_idx+400,'Output/results2/')  
        save_images(val_input[:,:,:,5:6], eval_idx+500,'Output/results2/')  
        save_images(val_input[:,:,:,6:7], eval_idx+600,'Output/results2/')  
        save_images(val_input[:,:,:,7:8], eval_idx+700,'Output/results2/')  
        save_images(val_input[:,:,:,8:9], eval_idx+800,'Output/results2/')  '''
        #print(val_output[0,0,:,:])
        #exit()
    #计算整体平均指标
    psnr_val = psnr_sum /len(test_loader) 
    ssim_val = ssim_sum /len(test_loader) 
    mse_val = mse_sum /len(test_loader) 
    nrmses_val = nrmses_sum /len(test_loader) 
    f.close() 
    return psnr_val, ssim_val, mse_val, nrmses_val
#保存单张输出图像到指定文件夹，文件名自动按数字编号
def save_images(output_image, eval_idx,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_image=output_image[0,:,:,0]*255
    output_image=output_image.astype('uint8')
    formatted_number = "{:03d}".format(eval_idx)
    tifffile.imsave(output_path+str(formatted_number)+'.tif', output_image)  #

