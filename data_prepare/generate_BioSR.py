import os
import glob

# 指定文件夹路径
root_folder = "/data3/DT/DDL/dataset/train/F-actin/training"
images_folder = "/data3/DT/DDL/dataset/train/F-actin/training_gt"

# 获取所有子文件夹
subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

# 获取所有图片文件
image_files = glob.glob(os.path.join(images_folder, "*.tif")) 

subfolders.sort()
image_files.sort()

# 创建并写入到输出文件中
print("Generate datalist ......")
with open("/data4/ddt/PGD-SIM_syn/data_prepare/datalist_ddt/F-actin/train_F-actin_list.txt", "a+") as f:
    for idex in range(len(subfolders)):
        print("subfolders:", len(subfolders))
        print("image_files:", len(image_files))
        #print(idex)
        #f.write(subfolders[idex][8:]+' '+image_files[idex][8:]+'\n')
        f.write(subfolders[idex] + ' ' + image_files[idex] + '\n')
f.close()
print("Finished datalist generation!")
       

