import numpy as np
import os
import torch
from PIL import Image
# 用于对图片进行iou计算
def calculate_iou(outputs, masks, calc_miou=False):
    # outputs: [batch_size, num_classes, 200, 200]
    # masks: [batch_size, 200, 200]
    # for every class calc each class iou on each image and then average over all images
    # return iou list of length num_classes
    outputs = torch.argmax(outputs, dim=1)
    iou_list = []
    for output in outputs:
        iou = []
        for i in range(1, 4):
            intersection = ((output == i) & (masks == i)).sum()
            union = ((output == i) | (masks == i)).sum()
            if union == 0:
                iou.append(np.nan)
            else:
                iou.append((intersection / union).cpu().numpy())
        if calc_miou:
            intersection = ((output != 0) & (masks != 0)).sum()
            union = ((output != 0) | (masks != 0)).sum()
            if union == 0:
                iou.append(np.nan)
            else:
                iou.append((intersection / union).cpu().numpy())
        iou_list.append(iou)
    return np.nanmean(iou_list, axis=0)

def calculate_iou_for_files(folder1, folder2, num_classes):
    files1 = [f for f in os.listdir(folder1) if f.endswith('.jpg')]
    files2 = [f for f in os.listdir(folder2) if f.endswith('.png')]
    total_iou = [[] for _ in range(num_classes)]  # 初始化为二维列表
    total_files = 0

    for file1,file2 in files1,files2:
        file_path1 = os.path.join(folder1, file1)
        file_path2 = os.path.join(folder2, file2)

        if not os.path.exists(file_path2):
            print(f"Warning: File {file_path2} does not exist.")
            continue

        pred_mask = Image.open(file_path1)
        true_mask = Image.open(file_path2)

        iou_results = calculate_iou(pred_mask, true_mask, num_classes)

        # 累加每个类别的 IoU 结果
        for i in range(num_classes):
            if iou_results[i]:
                total_iou[i].extend(iou_results[i])
        total_files += 1

    mean_iou = [np.mean(iou) if iou else float('nan') for iou in total_iou]
    mean_iou.append(np.nanmean(mean_iou) if total_files > 0 else float('nan'))  # 计算 mIoU

    return mean_iou

def main():
    folder2 = 'E:/ai/DNet/1_ self_net_gt_predictions'
    folder1 = 'E:/ai/v2/test_predictions'
    num_classes = 5  # 假设有4个类别 + 1背景

    iou_results = calculate_iou_for_files(folder1, folder2, num_classes)


    for i, iou in enumerate(iou_results[:-1], start=1):
        if(i==5):
            print(f'mIoU: {iou:.4f}')
        else:print(f'Class {i} IoU: {iou:.4f}')
