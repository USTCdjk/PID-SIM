import os
import random
from torch.utils.data import Dataset
import numpy as np
import tifffile
from util.util import prctile_norm

class Dataset(Dataset):
    def __init__(self, datapath, datalist, mode='train', norm=True, input_num=9, 
                 patch_size=None, data_percent=100.0, rotate_data_percent=0, rotate_angles=[90, 180, 270]):  
        # 新增2个旋转相关参数：
        # rotate_indices：需要旋转的样本索引列表（从基础集中筛选的索引）
        # rotate_angles：可选旋转角度列表（默认90/180/270度）
        self.datapath = datapath
        self.datalist = datalist
        self.mode = mode
        self.norm = norm
        self.input_num = input_num
        self.patch_size = patch_size
        self.data_percent = data_percent  # 原有：数据集百分比筛选
        self.rotate_data_percent = rotate_data_percent  # 新增：需旋转的样本索引（如[0,2,5,...]）
        self.rotate_angles = rotate_angles    # 新增：可选旋转角度
        self.rotate_indices = []
        
        # 读取并筛选数据列表（原有逻辑不变）
        self.in_path, self.gt_path = self._read_datalist()
        #百分比后筛选出的数据路径
        self.in_path, self.gt_path = self._filter_data_by_percent(self.in_path, self.gt_path)
        
        if  self.rotate_data_percent > 0:
           # 计算需要旋转的样本数量
           rotate_num = int(len(self.in_path) * (self.rotate_data_percent / 100))
           rotate_num = max(1, rotate_num)  # 至少旋转1个样本
           
           # 随机选择要旋转的样本索引
           np.random.seed(43)  # 固定种子，确保结果可复现
           self.rotate_indices = np.random.choice(len(self.in_path), rotate_num, replace=False).tolist()
        #print("rotate_indices",self.rotate_indices)

    def __len__(self):
        return len(self.in_path)

    def __getitem__(self, idx):
        # 原有实现保持不变...
        in_path = self.in_path[idx]
        gt_path = self.gt_path[idx]
        # 2. 判断是否需要旋转（当前样本索引在rotate_indices中）
        need_rotate = idx in self.rotate_indices
        rotate_angle = random.choice(self.rotate_angles) if need_rotate else 0  # 随机选一个旋转角度
        
        if self.mode == 'train':
           input_images = []
           norm_inputs = []
           for i in range(self.input_num):
               input_image = tifffile.imread(in_path + '/' + str(i + 1) + '.tif')
               input_image = np.array(input_image, dtype=np.float32)
               if need_rotate:
                    input_image = self._rotate_image(input_image, rotate_angle)
               input_image = input_image / 65535.
               if self.norm:
                  norm_input = prctile_norm(input_image)
               input_images.append(input_image)
               norm_inputs.append(norm_input)
           input_images = np.stack(input_images, 0)  
           norm_inputs = np.stack(norm_inputs, 0)    
        elif self.mode == 'eval':
           input_images = []
           norm_inputs = []
           with tifffile.TiffFile(in_path) as tif:
                for i in range(self.input_num):
                    input_image = tif.pages[i].asarray()
                    input_image = np.array(input_image, dtype=np.float32)
                    if need_rotate:
                        input_image = self._rotate_image(input_image, rotate_angle)
                    input_image = input_image / 65535.
                    if self.norm:
                       norm_input = prctile_norm(input_image)
                    input_images.append(input_image)
                    norm_inputs.append(norm_input)
                input_images = np.stack(input_images, 0) 
                norm_inputs = np.stack(norm_inputs, 0)   
        elif self.mode == 'test':
           input_images = []
           norm_inputs = []
           with tifffile.TiffFile(in_path) as tif:
                for i in range(self.input_num):
                    input_image = tif.pages[i].asarray()
                    input_image = np.array(input_image, dtype=np.float32)
                    if need_rotate:
                        input_image = self._rotate_image(input_image, rotate_angle)
                    input_image = input_image / 65535.
                    if self.norm:
                       norm_input = prctile_norm(input_image)
                    input_images.append(input_image)
                    norm_inputs.append(norm_input)
                input_images = np.stack(input_images, 0) 
                norm_inputs = np.stack(norm_inputs, 0) 
        else:
           print("Error! seleting the proper mode: train or test.")
           exit()
        
        gt_images = tifffile.imread(gt_path)
        gt_images = np.array(gt_images, dtype=np.float32)
        if need_rotate:  
            gt_images = self._rotate_image(gt_images, rotate_angle)
        gt_images = gt_images / 65535.
        if self.norm:
           norm_gt = prctile_norm(gt_images)
        gt_images = np.expand_dims(gt_images, axis=0)
        norm_gt = np.expand_dims(norm_gt, axis=0)

        return input_images, gt_images, norm_inputs, norm_gt

    
    def _read_datalist(self):
        f = open(self.datalist, 'r')
        in_path = []
        gt_path = []
        for line in f:
            try:
                in_img, gt_img = line.strip("\n").split(' ')
            except ValueError:
                in_img = gt_img = line.strip("\n")
            in_path.append(self.datapath + in_img)
            gt_path.append(self.datapath + gt_img)
        return in_path, gt_path  
    
    def _filter_data_by_percent(self, in_path, gt_path):
        """根据百分比筛选数据，并保证每次筛选结果一致"""
        if self.data_percent >= 100.0:
            return in_path, gt_path
            
        # 计算需要保留的样本数量
        total = len(in_path)
        keep_num = int(total * (self.data_percent / 100.0))
        keep_num = max(1, keep_num)  # 至少保留一个样本
        
        # 设置固定随机种子，确保每次筛选结果相同
        random.seed(42)  # 使用固定种子，保证结果可复现
        
        # 生成随机索引
        indices = list(range(total))
        random.shuffle(indices)
        selected_indices = indices[:keep_num]
        
        # 根据索引筛选数据
        selected_in = [in_path[i] for i in selected_indices]
        selected_gt = [gt_path[i] for i in selected_indices]
        
        return selected_in, selected_gt

    def random_crop(self, input_image, patch_size=512): 
        _, width, height = input_image.shape
        width_start = random.randint(5, width - patch_size - 5)
        width_end = width_start + patch_size
        height_start = random.randint(5, height - patch_size - 5)
        height_end = height_start + patch_size
        croped_image = input_image[:, width_start:width_end, height_start:height_end]
        return croped_image

    def _rotate_image(self, img, angle):
        """
        图像旋转（仅支持90/180/270度，无插值模糊）
        :param img: 输入图像（形状：[H, W] 单通道 / [C, H, W] 多通道）
        :param angle: 旋转角度（0/90/180/270）
        :return: 旋转后的图像（形状与输入一致）
        """
        if angle == 0:
            return img
        elif angle == 90:
            # 逆时针旋转90度（对最后两个维度：H,W）
            return np.rot90(img, k=1, axes=(-2, -1))
        elif angle == 180:
            # 旋转180度
            return np.rot90(img, k=2, axes=(-2, -1))
        elif angle == 270:
            # 顺时针旋转90度（等价于逆时针旋转270度）
            return np.rot90(img, k=3, axes=(-2, -1))
        else:
            raise ValueError(f"Unsupported rotation angle: {angle}! Only support 0/90/180/270.")