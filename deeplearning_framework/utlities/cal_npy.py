
import numpy as np
import os

# 用于对npy文件计算iou
def calculate_iou(prediction, ground_truth, num_classes):
    prediction = np.array(prediction)
    ground_truth = np.array(ground_truth)

    # 初始化 iou_list 为适当大小的列表
    iou_list = [[] for _ in range(num_classes)]

    classes = np.unique(np.concatenate([prediction.flatten(), ground_truth.flatten()]))
    classes = list(classes)
    classes.append(4)
    flag = 1
    for class_id in classes:
        if class_id == 0:
            continue

        gt_mask = (ground_truth == class_id).astype(np.uint8)
        pred_mask = (prediction == class_id).astype(np.uint8)
#miou
        if(class_id==4):
            gt_mask1  =(ground_truth >0).astype(np.uint8)
            pred_mask1 = (prediction >0).astype(np.uint8)
            intersection1 = np.sum(gt_mask1 & pred_mask1)
            union1 = np.sum(gt_mask1 | pred_mask1)
            if union1 == 0:
                iou1 = 0
            else:
                iou1 = intersection1 / union1
            iou_list[4].append(iou1)
            continue


        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)


        if union == 0:
            iou =  0 # 防止除以零

        else:
            iou = intersection / union


        iou_list[int(class_id)].append(iou)


    return iou_list


def calculate_iou_for_files(folder1, folder2, num_classes):
    files = [f for f in os.listdir(folder1) if f.endswith('.npy')]
    total_iou = [[] for _ in range(num_classes)]  # 初始化为二维列表
    total_files = 0

    for file in files:
        file_path1 = os.path.join(folder1, file)
        file_path2 = os.path.join(folder2, file)

        if not os.path.exists(file_path2):
            print(f"Warning: File {file_path2} does not exist.")
            continue

        pred_mask = np.load(file_path1)
        true_mask = np.load(file_path2)

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
    num_classes = 5  # 假设有4个类别

    iou_results = calculate_iou_for_files(folder1, folder2, num_classes)


    for i, iou in enumerate(iou_results[:-1], start=1):
        if(i==5):
            print(f'mIoU: {iou:.4f}')
        else:print(f'Class {i} IoU: {iou:.4f}')


if __name__ == '__main__':
    main()