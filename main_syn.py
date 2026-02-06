###
#训练模拟数据集代码
###
import os
import torch
import tifffile
from torch.utils.data import DataLoader
from dataset.Bioimage_dataset import Dataset,Test_Dataset
from argparse import ArgumentParser
#from model.model import SIM_net
from Scripts.train_syn import train
from Scripts.test_syn import test
import scipy
os.environ['CUDA_VISIBLE_DEVICES']= '0'


def build_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="PID", help="selecting the training model")
    parser.add_argument("--resume", type=str, default="", help="Resume path (default: none)")                            #断点继续训练需要加载的模型权重
    parser.add_argument("--chinkpoint_for_test", type=str, default="", help="Resume path (default: none)")  #测试集加载的权重
    parser.add_argument("--datapath", type=str, default='./../../../', help="data path")                                                                            #根路径  
    
    #数据集路径设置：训练集、验证集、测试集. 需要先读取图像的列表。
    parser.add_argument("--datalist", type=str, default='/data_prepare/datalist_ddt/HeLa/train_HeLa10Class_paths.txt', help="train list") #train_CCP_list.txt  #test_CCP_level_05_list.txt
    parser.add_argument("--evallist", type=str, default='/data_prepare/datalist_ddt/HeLa/valid_HeLa10Class_paths.txt', help="eval list")
    parser.add_argument("--testlist", type=str, default='', help="test list")
    
    #模型训练时的参数配置：lr、patchSize 、batchsize 、Epochs
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train")

    #gama参数，权重？模拟数据集?
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--weight', type=float, default=0.85, help='weight')

    #训练过程中，保存的训练log、结果路径、权重路径    
    # parser.add_argument("--log_pth", type=str, default='./logs/log_test.txt')
    # parser.add_argument("--results_path", type=str, default='Output_DT_syn/results/')
    # parser.add_argument("--checkpoint_path", type=str, default='./checkpoint_HeLa_SFE', help="data path")

    parser.add_argument("--log_pth", type=str, default='./logs/log_test.txt')
    parser.add_argument("--results_path", type=str, default='Output_hcp_CCPs/results/')
    parser.add_argument("--checkpoint_path", type=str, default='/checkpoint', help="data path")
    
    #训练过程中保存的间隔和测试的间隔
    parser.add_argument("--test_frequency", type=int, default=1000)
    parser.add_argument("--model_save_frequency", type=int, default=2)
    
    #训练模式还是测试模式
    parser.add_argument("--train_or_test", type=str, default='train')


    args = parser.parse_args()
    return args

#主函数入口
def main(args):
    # print(f"实际调用的 scipy 版本：{scipy.__version__}")
    # print(f"scipy 安装路径：{scipy.__file__}")
    if args.train_or_test == 'train':
       #print("生成sim中...请等待！")
       train_data=Dataset(args.datapath,args.datalist,'train', args.patch_size)                 #对A数据集添加操作，附加SIM信息，得到B数据集；B数据集用来预训练模型，得到微调模型
       #print(train_data.count)
       eval_data=Dataset(args.datapath,args.evallist,'eval', args.patch_size)
       train_loader=DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=True)
       #print(train_loader.count)
       eval_loader=DataLoader(eval_data, batch_size=1, num_workers=4, shuffle=False)
       #print("生成sim结束...开始训练！")
       train(args, train_loader, eval_loader)                                                       #训练函数
    elif args.train_or_test == 'test':
       #test_data=Test_Dataset(args.datapath,args.testlist) 
       test_data=Dataset(args.datapath,args.testlist,'test', args.patch_size) 
       test_loader=DataLoader(test_data, batch_size=1, shuffle=False)
       test(args, test_loader)
    else:
       print("Error: wrong args for train_or_test!")
       exit()
if __name__ == "__main__":
    args=build_args()
    main(args)
    print("over!")


'''for batch_idx, (input_images, gt_images) in enumerate(data_loader):

    gt_images = gt_images*65535
    input_images = input_images*65535
    gt_images = gt_images.numpy().astype('uint16')
    input_images = input_images.numpy().astype('uint16')
    #print(gt_images)
    # 使用 OpenCV 保存图像
    tifffile.imsave('gt_image.tif', gt_images[0,0,:,:])
    tifffile.imsave('input_images1.tif', input_images[0,0,:,:])
    tifffile.imsave('input_images2.tif', input_images[0,1,:,:])
    tifffile.imsave('input_images3.tif', input_images[0,2,:,:])
    tifffile.imsave('input_images4.tif', input_images[0,3,:,:])
    tifffile.imsave('input_images5.tif', input_images[0,4,:,:])
    tifffile.imsave('input_images6.tif', input_images[0,5,:,:])
    tifffile.imsave('input_images7.tif', input_images[0,6,:,:])
    tifffile.imsave('input_images8.tif', input_images[0,7,:,:])
    tifffile.imsave('input_images9.tif', input_images[0,8,:,:])
    exit()'''



