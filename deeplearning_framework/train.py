
# 导入所需的库
from torch.utils.data import DataLoader, Dataset  # 用于数据加载
from PIL import Image  # 用于图像处理
import numpy as np  # 用于数值计算
import torch  # PyTorch库
import torch.nn as nn  # 神经网络模块
from dataloader.dataloader import datasets, transform  # 自定义的数据集和变换
from model import model  # 自定义模型
import tqdm  # 进度条显示


# 创建训练数据加载器
def train_loader(transform, image_dir, mask_dir, batch_size):
    # 使用给定的变换和路径创建数据集
    dataset = datasets(image_dir, mask_dir, transform)
    # 创建数据加载器，支持批处理和洗牌
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# 创建测试数据加载器
def test_loader(transform, image_dir, mask_dir, batch_size):
    dataset = datasets(image_dir, mask_dir, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


# 定义损失函数
def loss_fun(output, target):
    # 使用交叉熵损失
    return nn.CrossEntropyLoss()(output, target)


# 训练模型的函数
def train_model(model, dataloader, criterion, optimizer, num_epochs, device, test_data_loader, scheduler=None):
    model = model.to(device)  # 将模型移动到指定设备（GPU或CPU）

    for epoch in range(num_epochs):  # 遍历每个训练周期
        model.train()  # 设置模型为训练模式

        loss1_plot = []  # 存储训练损失
        loss2_plot = []  # 存储验证损失

        for images, masks in tqdm(dataloader):  # 遍历训练数据加载器
            images = images.to(device)  # 移动图像到设备
            masks = masks.to(device)  # 移动掩膜到设备

            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, masks)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            loss1_plot.append(loss.item())  # 记录训练损失

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度
            for images, masks in tqdm(test_data_loader):  # 遍历测试数据加载器
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, masks)  # 计算损失
                loss2_plot.append(loss.item())  # 记录验证损失

        # 输出每个周期的损失信息
        print(
            f"Epoch: {epoch + 1}, Loss: {sum(loss1_plot) / len(loss1_plot)}, Validation Loss: {sum(loss2_plot) / len(loss2_plot)}")

        if scheduler is not None:  # 如果有学习率调度器
            print(f"Lr: {scheduler.get_last_lr()}")
            scheduler.step()  # 更新学习率

        if epoch % 10 == 0:  # 每10个周期保存一次模型
            torch.save(model.state_dict(), "model_epoch_{}.pth".format(epoch))
            print("Model saved to model_epoch_{}.pth".format(epoch))

    return model  # 返回训练好的模型


# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择计算设备

    train_transform = transform()  # 获取训练数据的变换
    test_transform = transform()  # 获取测试数据的变换

    # 设置图像和掩膜的路径及批量大小
    train_img_dir = ""
    train_mask_dir = ""
    train_batch_size = 4
    test_img_dir = ""
    test_mask_dir = ""
    test_batch_size = 1

    # 创建数据加载器
    train_data_loader = train_loader(train_transform, train_img_dir, train_mask_dir, train_batch_size)
    test_data_loader = test_loader(test_transform, test_img_dir, test_mask_dir, test_batch_size)

    # 初始化模型并加载预训练权重
    model = model.self_net(input_shape=(1, 200, 200), num_classes=4).to(device)
    model.load_state_dict(torch.load("model_nano.pth"))

    criterion = loss_fun  # 设置损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 设置优化器

    num_epochs = 10  # 训练周期数
    model = train_model(model, train_data_loader, criterion, optimizer, num_epochs, device, test_data_loader)  # 训练模型
    torch.save(model.state_dict(), "model_nano.pth")  # 保存最终模型
    print("Model saved to model.pth")
