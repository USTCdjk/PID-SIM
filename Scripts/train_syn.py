import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tifffile
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.model import PID
from model.APCAN_1 import APCAN
from model.DFCAN16 import DFCAN
from util.util import img_comp
from loss.Loss import RegLoss, ReconLoss
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")    

def train(args, train_loader, eval_loader):
    # 初始化模型
    if args.model_name == 'APCAN':
        print("Training model: APCAN")
        model = APCAN().cuda()
    elif args.model_name == 'DFCAN':
        print("Training model: DFCAN")
        model = DFCAN().cuda()
    elif args.model_name == 'PID':
        print("Training model: PID")
        model = PID().cuda()
    else:
        raise ValueError(f"Error: the model {args.model_name} is not implemented!")

    # 优化器与调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.nEpochs)

    # TensorBoard
    writer = SummaryWriter()

    # 损失函数
    L1_loss = nn.L1Loss()
    Recon_Loss = ReconLoss()
    lamda = 1
    Reg_Loss = RegLoss(Pred_w=1., Stripe_w=lamda, Noisy_w=0.4, Wave_w=lamda, Phase_w=lamda, ModFac_w=lamda,PSF_w=lamda)

    # 断点续训
    start_epoch = 0
    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")

    print("Start training >>>>>>>>") 

    # 训练主循环（每个 Epoch 一个进度条）
    for epoch in range(start_epoch, args.nEpochs):
        # 初始化 Epoch 级 Loss 累计
        epoch_total_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_rcon_loss = 0.0

        # 单个 Epoch 的进度条（tqdm 包裹整个 train_loader 迭代）
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch [{epoch + 1}/{args.nEpochs}]", ncols=100)
        
        for i, (gt_img, Syn_input, Stripes_sig, Noise_sig, Optical_para) in pbar:
            # 数据入 GPU
            input_images = Syn_input.cuda()
            gt_images = gt_img.cuda()
            Stripes_sig = Stripes_sig.cuda()
            Noise_sig = Noise_sig.cuda()

            # 模型前向
            model.train()
            recon_image, noisy_image, stripe_image, wave_vector, phase_shifts, ModFac, PSFscale = model(input_images)

            # 调整张量形状
            shape = np.shape(recon_image)
            wave_vector = torch.reshape(wave_vector, (shape[0], 3, 2))
            phase_shifts = torch.reshape(phase_shifts, (shape[0], 3, 3))

            # 计算损失
            output = recon_image
            loss_reg = Reg_Loss(output, gt_images, stripe_image, Stripes_sig, noisy_image, Noise_sig, Optical_para, wave_vector, phase_shifts, ModFac, PSFscale)
            loss_rcon = Recon_Loss(input_images, output, noisy_image, Optical_para, wave_vector, phase_shifts, ModFac, PSFscale)
            batch_loss = 0.9 * loss_reg + 0.1 * loss_rcon

            # 累计 Epoch 级 Loss
            epoch_total_loss += batch_loss.item()
            epoch_reg_loss += loss_reg.item()
            epoch_rcon_loss += loss_rcon.item()

            # 反向传播与优化
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 验证集评估（保持原逻辑，每 args.test_frequency 步执行）
            if (i + 1) % args.test_frequency == 0:
                with torch.no_grad():
                    model.eval()
                    psnr_val, ssim_val, mse_val, nrmses_val = evaluation(eval_loader, model)
                    writer.add_scalar('psnr_val', psnr_val, epoch * len(train_loader) + i)
                    writer.add_scalar('ssim_val', ssim_val, epoch * len(train_loader) + i)
                    writer.add_scalar('mse_val', mse_val, epoch * len(train_loader) + i)
                    writer.add_scalar('nrmses_val', nrmses_val, epoch * len(train_loader) + i)
                writer.add_scalar('train_batch_loss', batch_loss.item(), epoch * len(train_loader) + i)
                pbar.set_postfix_str(f'Eval PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, MSE: {mse_val:.4f}, NRMSE: {nrmses_val:.4f}')

        # Epoch 结束：计算平均 Loss 并打印
        epoch_avg_loss = epoch_total_loss / len(train_loader)
        epoch_avg_reg_loss = epoch_reg_loss / len(train_loader)
        epoch_avg_rcon_loss = epoch_rcon_loss / len(train_loader)

        print("=" * 80)
        print(f"Epoch [{epoch + 1}/{args.nEpochs}] | Avg Total Loss: {epoch_avg_loss:.6f} | Avg Reg Loss: {epoch_avg_reg_loss:.6f} | Avg Recon Loss: {epoch_avg_rcon_loss:.6f}")
        print("=" * 80)

        # TensorBoard 记录 Epoch 级 Loss
        writer.add_scalar('Avg_Train_Loss_per_Epoch', epoch_avg_loss, epoch)
        writer.add_scalar('Avg_Reg_Loss_per_Epoch', epoch_avg_reg_loss, epoch)
        writer.add_scalar('Avg_Recon_Loss_per_Epoch', epoch_avg_rcon_loss, epoch)

        # 学习率调度
        scheduler.step()

        # 保存断点
        if (epoch + 1) % args.model_save_frequency == 0:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            filename = f'checkpoint_epoch{epoch + 1}.pth.tar'
            save_path = os.path.join(args.checkpoint_path, filename)               
            print(f"Saving checkpoint to: {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_avg_loss': epoch_avg_loss
            }, save_path)
            print(f"Checkpoint saved for epoch {epoch + 1}")

    writer.close()

def evaluation(eval_loader, model):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    nrmses_sum = 0.0
    total_samples = 0

    for eval_idx, (gt_img, Syn_input, _, _, _) in enumerate(eval_loader):
        val_input = Syn_input.cuda()
        val_gt = gt_img.cuda()
        batch_size = val_input.shape[0]
        total_samples += batch_size

        with torch.no_grad():
            recon_image, _, _, _, _, _, _ = model(val_input)  # 仅取重建图像
        val_output = recon_image.cpu().numpy().transpose(0, 2, 3, 1)
        val_gt = val_gt.cpu().numpy().transpose(0, 2, 3, 1)

        for inp, out in zip(val_output, val_gt):
            psnrs, ssims, mses, nrmses = img_comp(inp, out)
            psnr_sum += psnrs
            ssim_sum += ssims
            mse_sum += mses
            nrmses_sum += nrmses

    psnr_val = psnr_sum / total_samples if total_samples > 0 else 0
    ssim_val = ssim_sum / total_samples if total_samples > 0 else 0
    mse_val = mse_sum / total_samples if total_samples > 0 else 0
    nrmses_val = nrmses_sum / total_samples if total_samples > 0 else 0
        
    return psnr_val, ssim_val, mse_val, nrmses_val

def save_images(output_image, eval_idx, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_image = output_image[0, :, :, 0] * 255
    output_image = output_image.astype('uint8')
    formatted_number = f"{eval_idx:03d}"
    tifffile.imsave(os.path.join(output_path, f"{formatted_number}.tif"), output_image)