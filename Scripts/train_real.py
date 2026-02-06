import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import cv2
import tifffile
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from model.model_ap.APCAN_1 import SIM_net
# from model.model import SIM_net
from model.model import PID
from model.APCAN_1 import APCAN
from model.DFCAN16 import DFCAN
from util.util import img_comp
from loss.Loss import RegLoss, ReconLoss, RealRegLoss
from loss.Spectrum_Loss import SpectrumLoss
from loss.Loss import SSIM
# import pyiqa
# os.environ['CUDA_VISIBLE_DEVICES']= '0'
import warnings

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")
from tqdm import tqdm

def train(args, train_loader, test_loader):
    # 初始化模型、损失函数和优化器
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 设置 TensorBoard
    writer = SummaryWriter()
    # 设置断点继续训练的参数
    start_epoch = 0
    L1_loss = nn.L1Loss()
    Psf_loss = nn.MSELoss()
    Recon_Loss = ReconLoss()
    ssim_loss = SSIM()
    lamda = 1
    Reg_Loss = RealRegLoss(weight=args.weight)
    RealReg_Loss = RealRegLoss(args.weight)
    Spectrum_Loss = SpectrumLoss()
    # 处理权重冻结和断点续训
    if args.resume:
        # 断点续训时，加载已有权重，使用已冻结的状态
        print(f"从断点 {args.resume} 继续训练")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint['epoch']
        print(f"从第 {start_epoch}  epoch 继续训练")

        # 断点续训也需要重新设置冻结状态
        # if args.freeze_percent > 0:
        #     print(f"断点续训，保持冻结 {args.freeze_percent}% 的权重")
        #     model = set_parameter_requires_grad(model, args.freeze_percent)

    Optical_para = {"Nangles": np.array([3]), "Nshifts": np.array([3]),
                    "meanInten": torch.tensor(np.ones((args.batch_size, 3)))}

    # 设置学习率调整策略
    scheduler = CosineAnnealingLR(optimizer, T_max=args.nEpochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)

    best_psnr_dict = {}  # 用于记录每个测试集的最佳PSNR和对应的epoch

    print("start training>>>>>>>>>")
    print(
        f"use_data: {args.data_percent}% | dataset use_nums: {len(train_loader) * args.batch_size} | dataset total: {len(train_loader) * args.batch_size / (args.data_percent / 100)}")
    print(
        f"将一部分数据集进行旋转: rotate_data: {args.rotate_data_percent}% | "
    )
    for epoch in range(start_epoch, args.nEpochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", ncols=100)
        for i, (input_images, gt_images, norm_inputs, norm_gt) in pbar:
            input_images = norm_inputs.cuda()
            gt_images = norm_gt.cuda()

            model.train()
            recon_image = model(input_images)
            # print(recon_image.shape)
            # print("psf:",psf_pred.shape)
            # shape=np.shape(recon_image)
            # wave_vector = torch.reshape(wave_vector, (shape[0], 3,2))
            # phase_shifts = torch.reshape(phase_shifts, (shape[0], 3,3))
            # PSFscale = PSFscale.view(shape[0])

            output = recon_image[0]
            # print(L1_loss(output, gt_images),ssim_loss(output, gt_images))
            '''ggtt=gt_images.cpu().detach().numpy().transpose(0, 2, 3, 1)
            ppdd=input_images.cpu().detach().numpy().transpose(0, 2, 3, 1)
            print(np.shape(ggtt))
            print(np.shape(ppdd))
            save_images(ggtt[0:1,:,:,:], i, 'Output/results/')
            save_images(ppdd[0:1,:,:,0:1], i, 'Output/results2/')'''
            # print("outshape:",output.shape)
            # print("gt_images_shape",gt_images.shape)
            loss_reg = Reg_Loss(output, gt_images)
            loss_Spec = Spectrum_Loss(output, gt_images)
            # gt_PSF产生
            # gt_psf = generator_psf_batch(batch_size=input_images.size(0), device=device)
            # gt_psf = gt_psf.unsqueeze(1)                 # shape: [B, 1, H, W]
            # # 5. 计算损失
            # loss_psf = Psf_loss(psf_pred, gt_psf)
            # loss = loss_reg + 0.1*loss_Spec
            # loss_rcon=Recon_Loss(input_images, output, noisy_image, Optical_para, wave_vector, phase_shifts, ModFac, PSFscale)
            # loss = 0.9*loss_reg + 0.1*loss_psf #+ loss_recon#L1_loss(output, gt_images) #loss_reg#+ 0.25*loss_rcon
            # loss = args.weight*L1_loss(output, gt_images)#+(1-args.weight)*(1-ssim_loss(output, gt_images))
            loss = args.weight * L1_loss(output, gt_images) + (1 - args.weight) * torch.abs(
                1 - ssim_loss(output, gt_images))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            '''if (i+1) % (args.test_frequency//10) ==0:
               print(f'Epoch [{epoch + 1}/{args.nEpochs}], Step [{i + 1}/{len(train_loader)}], '
                     f'Train Loss: {loss:.4f}')'''
            # if (i+1) % args.test_frequency == 0:

            #     with torch.no_grad():
            #          model.eval()
            #          psnr_val, ssim_val, mse_val, nrmses_val=evaluation(eval_loader, model)
            #          writer.add_scalar('psnr_val', psnr_val, epoch * len(train_loader) + i)
            #          writer.add_scalar('ssim_val', ssim_val, epoch * len(train_loader) + i)
            #          writer.add_scalar('mse_val', ssim_val, epoch * len(train_loader) + i)
            #          writer.add_scalar('nrmses_val', ssim_val, epoch * len(train_loader) + i)
            #     writer.add_scalar('train_loss', running_loss / args.test_frequency, epoch)
            #     print(f'Epoch [{epoch + 1}/{args.nEpochs}], Step [{i + 1}/{len(train_loader)}], '
            #           f'Eval Loss: {running_loss / args.test_frequency:.4f},'
            #           f'Eval_PSNR: {psnr_val:.4f}, Eval_SSIM: {ssim_val:.4f}, Eval_mse: {mse_val:.4f}, Eval_nrmses: {nrmses_val:.4f}')

            #     running_loss = 0.0
            # 👉 在此添加打印每轮loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.nEpochs}], Avg Train Loss: {avg_loss:.6f}")
        writer.add_scalar('Avg_Train_Loss_per_Epoch', avg_loss, epoch)

        scheduler.step()
        # 保存断点
        if (epoch + 1) % args.model_save_frequency == 0:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            filename = 'chinkpoint' + '_epoch' + str(epoch + 1) + '.pth.tar'
            save_path = os.path.join(args.checkpoint_path, filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f'Saved checkpoint for epoch {epoch}:', save_path)

            # 每隔args.train_span_test》测试这12个测试集》由于12个测试集指标的结果不用来微调模型，所以不影响最终的结果，不同于验证集。
            if (epoch + 1) % args.train_span_test == 0:
                # 创建日志文件夹
                if not os.path.exists(args.checkpoint_path):
                    os.makedirs(args.checkpoint_path)
                # log_path = f"/data3/ddt/PID-SIM_syn/checkpoint_PSF/FinalModel/TEST_results_epoch{epoch+1}.txt"
                log_path = os.path.join(args.checkpoint_path, "TEST_results.txt")
                with open(log_path, 'a') as log_file:
                    log_file.write(f"Epoch {epoch + 1} Evaluation Results:\n")
                    log_file.write("Dataset\tPSNR\tSSIM\tMSE\tNRMSE\n")

                    for name, loader in test_loader:
                        with torch.no_grad():
                            model.eval()
                            psnr_val, ssim_val, mse_val, nrmses_val = evaluation(args, loader, model)
                            line = f"{name}\t{psnr_val:.4f}\t{ssim_val:.4f}\t{mse_val:.4e}\t{nrmses_val:.4f}\n"
                            print(line.strip())
                            log_file.write(line)

                            # 保存 PSNR 最好的记录，同时记录其他指标（以 PSNR 为准）
                            if name not in best_psnr_dict or psnr_val > best_psnr_dict[name]['psnr']:
                                best_psnr_dict[name] = {
                                    'psnr': psnr_val,
                                    'ssim': ssim_val,
                                    'mse': mse_val,
                                    'nrmses': nrmses_val,
                                    'epoch': epoch + 1
                                }

                # 保存最佳测试集指标到TXT文件
                best_metrics_path = os.path.join(args.checkpoint_path, "best_metrics.txt")
                with open(best_metrics_path, 'w') as f:
                    f.write("Dataset\tBest_PSNR\tSSIM\tMSE\tNRMSE\tEpoch\n")
                    for name, metrics in best_psnr_dict.items():
                        line = f"{name}\t{metrics['psnr']:.4f}\t{metrics['ssim']:.4f}\t{metrics['mse']:.4e}\t{metrics['nrmses']:.4f}\t{metrics['epoch']}\n"
                        f.write(line)
                print(f"Saved best metrics to {best_metrics_path}")

    writer.close()


def evaluation(args, test_loader, model):
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
            recon_image = model(val_input)
        val_output = recon_image[0]
        val_output = val_output.cpu().detach().numpy().transpose(0, 2, 3, 1)
        val_output = np.clip(val_output, 0, 1)
        val_gt = val_gt.cpu().numpy().transpose(0, 2, 3, 1)
        val_input = val_input.cpu().numpy().transpose(0, 2, 3, 1)
        # stripe_image = stripe_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        # noisy_image = noisy_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        # norm_input= norm_input.numpy().transpose(0, 2, 3, 1)
        for inp, out in zip(val_output, val_gt):
            psnrs, ssims, mses, nrmses = img_comp(inp, out)
            psnr_sum += psnrs
            ssim_sum += ssims
            mse_sum += mses
            nrmses_sum += nrmses
        save_images(val_gt, eval_idx, 'Output/gt/')
        save_images(val_input[:, :, :, 0:1], eval_idx, 'Output/input/')
        save_images(val_output, eval_idx, args.results_path)
        # save_images(noisy_image[:,:,:,0:1], eval_idx,'Output/NoisyImage/')
        # save_images(stripe_image[:,:,:,0:1], eval_idx,'Output/StripeImage/')

        # shape=np.shape(recon_image)
        # wave_vector = torch.reshape(wave_vector, (shape[0], 3,2))
        # phase_shifts = torch.reshape(phase_shifts, (shape[0], 3,3))
        # PSFscale = PSFscale.view(shape[0])
        # ModFac = ModFac.view(shape[0])

        with open('output.txt', 'a') as f:

            #  f.write("wave_vector:\n")
            #  for i in range(wave_vector.shape[0]):
            #      #f.write(f"Slice {i}:\n")
            #      for j in range(wave_vector.shape[1]):
            #          line = ' '.join(f"{wave_vector[i, j, k].item()}" for k in range(wave_vector.shape[2]))
            #          f.write(line + "\n")

            #  f.write("phase_shifts:\n")
            #  for i in range(phase_shifts.shape[0]):
            #      #f.write(f"Slice {i}:\n")
            #      for j in range(phase_shifts.shape[1]):
            #          line = ' '.join(f"{phase_shifts[i, j, k].item()}" for k in range(phase_shifts.shape[2]))
            #          f.write(line + "\n")

            # f.write(f"PSFscale: {PSFscale.item()}\n")

            #  f.write(f"ModFac: {ModFac.item()}\n")

            f.write("\n")

        # save_images(val_input[:,:,:,0:1], eval_idx,'Output/results2/')
        '''save_images(val_input[:,:,:,1:2], eval_idx+100,'Output/results2/')  
        save_images(val_input[:,:,:,2:3], eval_idx+200,'Output/results2/')  
        save_images(val_input[:,:,:,3:4], eval_idx+300,'Output/results2/')  
        save_images(val_input[:,:,:,4:5], eval_idx+400,'Output/results2/')  
        save_images(val_input[:,:,:,5:6], eval_idx+500,'Output/results2/')  
        save_images(val_input[:,:,:,6:7], eval_idx+600,'Output/results2/')  
        save_images(val_input[:,:,:,7:8], eval_idx+700,'Output/results2/')  
        save_images(val_input[:,:,:,8:9], eval_idx+800,'Output/results2/')  '''
        # print(val_output[0,0,:,:])
        # exit()
    psnr_val = psnr_sum / len(test_loader)
    ssim_val = ssim_sum / len(test_loader)
    mse_val = mse_sum / len(test_loader)
    nrmses_val = nrmses_sum / len(test_loader)
    f.close()
    return psnr_val, ssim_val, mse_val, nrmses_val


def save_images(output_image, eval_idx, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_image = output_image[0, :, :, 0] * 65535
    output_image = output_image.astype('uint16')
    formatted_number = "{:03d}".format(eval_idx)
    tifffile.imsave(output_path + str(formatted_number) + '.tif', output_image)  #
