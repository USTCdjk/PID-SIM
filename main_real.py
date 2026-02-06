import os
import torch
import tifffile
import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from dataset.BioSR_dataset import Dataset   #这个也需要改
from argparse import ArgumentParser
# from model.model import SIM_net
from Scripts.train_real import train #需要改这个train函数装着它的头文件路径
from Scripts.test_real import test
os.environ['CUDA_VISIBLE_DEVICES']= '1'


def build_args():
    parser = ArgumentParser()
    # 原有参数保持不变...
    parser.add_argument("--model_name", type=str, default="PID", help="selecting the training model")#模型修改选择APCAN
    parser.add_argument("--resume", type=str, default="", help="Resume path (default: none)")
    parser.add_argument("--chinkpoint_for_test", type=str, default="", help="Resume path (default: none)")
    parser.add_argument("--datapath", type=str, default='', help="data path")
    parser.add_argument("--datalist", type=str, default='/data_prepare/datalist_ddt/MT/train_MT_list.txt', help="train list")
    
    # 添加数据集百分比参数
    parser.add_argument("--data_percent", type=float, default=100.0, help="Percentage of data to use (0-100)") #使用单个数据集 的10%
    
    # 新增旋转相关参数
    parser.add_argument("--rotate_data_percent", type=float, default=20.0, 
                      help="Percentage of selected data to rotate and add to training set (0-100)")
    parser.add_argument("--rotate_angles", type=int, nargs='+', default=[90, 180, 270], 
                      help="List of rotation angles in degrees (default: 90 180 270)")
    
    # 冻结参数
    # parser.add_argument("--freeze_percent", type=float, default=60.0, 
    #                   help="Percentage of pretrained weights to freeze (0-100, 0=no freeze)")
    
    # 其他原有参数...
    parser.add_argument("--testlist", type=str, default='', help="test list")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--weight', type=float, default=0.1, help='weight')
    parser.add_argument("--log_pth", type=str, default='')
    parser.add_argument("--results_path", type=str, default='')
    parser.add_argument("--checkpoint_path", type=str, default='', help="data path")
    parser.add_argument("--test_frequency", type=int, default=10)
    parser.add_argument("--model_save_frequency", type=int, default=5)
    parser.add_argument("--train_or_test", type=str, default='train')
    parser.add_argument("--train_span_test", type=int, default='5')
    
    args = parser.parse_args()
    
    # 添加参数验证
    assert 0 <= args.rotate_data_percent <= 100, "rotate_data_percent must be between 0 and 100"
    for angle in args.rotate_angles:
        assert angle in [0, 90, 180, 270], "rotate_angles must be 0, 90, 180, or 270"
    
    return args


def main(args):
    if args.train_or_test == 'train':
       # 加载原始训练集
       train_data = Dataset(
           args.datapath, 
           args.datalist, 
           'train', 
           patch_size=args.patch_size,
           data_percent=args.data_percent,
           rotate_data_percent=args.rotate_data_percent,
           rotate_angles=args.rotate_angles
       )
    
       # 创建数据加载器
       train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
       
       # 测试集加载部分保持不变
       datalist_root='/data4/ddt/PID-SIM_syn/data_prepare/'
       test_lists = [
            'datalist_ddt/F-actin/test_F-actin_high.txt',
             'datalist_ddt/F-actin/test_F-actin_low.txt',
            'datalist_ddt/F-actin/test_F-actin_medium.txt',
        ]

       test_loaders = []
       for tlist in test_lists:
            full_path = os.path.join(datalist_root, tlist)
            test_data = Dataset(args.datapath, full_path, 'test', args.patch_size)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
            test_loaders.append((os.path.basename(tlist), test_loader))

       train(args, train_loader, test_loaders) 
    elif args.train_or_test == 'test':
       print(f"读取数据list：{args.testlist}")
       print(f"读取的模型地址为：{args.chinkpoint_for_test}")
       test_data=Dataset(args.datapath, args.testlist, 'test', args.patch_size) 
       test_loader=DataLoader(test_data, batch_size=1, shuffle=False)
       test(args, test_loader)
    else:
       print("Error: wrong args for train_or_test!")
       exit()

if __name__ == "__main__":
    args=build_args()
    main(args)
    print("over!")
