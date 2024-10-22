import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class my_dataset(Dataset):
    def __init__(self, image_path, annotation_path, transform):
        # 读取数据集文件
        self.image_filenames = os.listdir(image_path)
        self.annotation_filenames = os.listdir(annotation_path)
        # 初始化 transform
        self.transform = transform()
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index) :
        # 读取图像和标签
        image_filename = self.image_filenames[index]
        anntation_filename = self.annotation_filenames[index]
        # 使用os.path.join 来组合路径和文件
        image_path = os.path.join('datasets','images', image_filename)
        annotation_path = os.path.join('datasets', 'annotations', anntation_filename)
        # 读取文件
        image = open(image_path)
        annotation = open(annotation_path)
        # 对数据进行增强
        images = self.transform(image)

        return (image,annotation)

class my_transform(transforms):
    def __init__(self, image, input_size):
        self.images = image
        self.input_size = input_size
    def data_augmentation(self):
        data_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomRotation(degrees=45),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), # 亮度，对比度，饱和度，色相
            transforms.RandomGrayscale(p = 0.025),
            transforms.ToTensor(),
            transforms.Normalize([0,0,0],[0,0,0])# 均值和标准差


        ])
# 使用示例
if __name__ == "__main__":
    img_path = ''
    annotation_pth = ''
    batch = 10
    transform = my_transform()
    datasets = my_dataset(image_path=img_path, annotation_path=annotation_pth, transform=transform)
    dataloader = DataLoader(datasets,batch_size=batch,shuffle=True, num_workers=8)# num_workers线程数


    

