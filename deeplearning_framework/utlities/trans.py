import os
import numpy as np
from PIL import Image


def png_to_npy(source_folder, target_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            # 构建完整的文件路径
            file_path = os.path.join(source_folder, filename)

            # 使用PIL打开图像
            with Image.open(file_path) as img:
                # 将图像转换为NumPy数组
                img_array = np.array(img)

                # 构建目标文件路径
                target_path = os.path.join(target_folder, filename.replace('.png', '.npy'))

                # 保存NumPy数组为.npy文件
                np.save(target_path, img_array)
                print(f'Converted {filename} to {target_path}')


# 指定源文件夹和目标文件夹
source_folder = './datasets/results/test'
target_folder = './self_net_predictions'

if __name__ == '__main__':
    # 调用函数
    png_to_npy(source_folder, target_folder)