# 基线方法实现代码
# 基本配置：
#   dataset_config：指定数据集路径和文件名匹配方法
#   gpu_ranges：指定训练的freq_index区间 和该区间训练的gpu 例如 如果我只有一块8Gb显存的gpu 我可以指定 两个gpu_ranges： [0,16,0]和 [16,32,0] 即代表在 0号gpu上同时训练两个线程
#       第一个线程依次训练 freq_index 0-15 第二个线程依次训练16-31 每个模型约占4Gb显存
# 其它配置：
#   诸如学习率配置等、命名很标准在 train_model 方法中初始那一块的定义中，默认给出即可

import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import ShDataset
import torch.nn as nn
import time

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.elu2 = nn.ELU(inplace=True)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.elu3 = nn.ELU(inplace=True)
        self.multiple = int(out_channels/ in_channels)

    def forward(self, x):
        identity = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # print(out.shape)
        # print(identity.shape)
        # print(self.multiple)
        identity = identity.repeat_interleave(self.multiple, dim=1)
        out += identity  # Adding the residual connection
        out = torch.nn.functional.max_pool3d(out, kernel_size=2, stride=2)
        return out

class CNNModel(nn.Module):
    def __init__(self, num_classes=360):  # num_classes represents the number of frequency bins
        super(CNNModel, self).__init__()
        self.block1 = Block(1, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 512)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)  # Output for one frequency bin

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def subsample_points(feature, coord, num_points):
    """下采样点云到指定的点数"""
    current_num_points = coord.size(2)
    if current_num_points > num_points:
        indices = torch.randperm(current_num_points)[:num_points]
        return feature[:, indices, :], coord[:, :, indices]
    return feature, coord

def pad_points(feature, coord, num_points):
    """填充点云到指定的点数"""
    current_num_points = coord.size(2)
    if current_num_points < num_points:
        padding_size = num_points - current_num_points
        padding_feature = torch.zeros(1, padding_size, 1, device=feature.device)
        padding_coord = torch.zeros(1, 3, padding_size, device=coord.device)
        return torch.cat([feature, padding_feature], dim=1), torch.cat([coord, padding_coord], dim=2)
    return feature, coord


def evaluate_test_set(model, freq_index, test_loader, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():  # 在评估阶段不计算梯度
        for voxel, hrtf, sh in test_loader:
            voxel = voxel.unsqueeze(1).float().to(device)
            # print(hrtf.shape)
            hrtf = hrtf[:, :, 1,:].squeeze().float().to(device)
            hrtf = hrtf[:, :, freq_index].squeeze().float().to(device)
            outputs = model(voxel)
            loss = criterion(outputs, hrtf)
            total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    model.train()  # 将模型设置回训练模式
    return average_loss

def train_model(freq_index, gpu_id):
    # 定义输出目录的基础路径
    OUTPUT_DIR_BASE = "/data/fyw/onlinux/output_chedar_CNN_erb_log/"

    # 初始化模型
    model = CNNModel(2522)
    batchsize = 30
    # 假设输入数据形状为[batch_size, channels, depth, height, width]
    # input_size = (1, n_input_channels, 32, 32, 32)  # 示例输入

    # 打印模型每一层的输出形状
    # print_model_shape(model, input_size)

    criterion = nn.MSELoss()  # 或根据任务选择其他损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.2)  # Start with a learning rate of 0.1
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # Decay the learning rate

    log_interval = 10  # 每10个epoch记录一次
    log_file = "training_log.txt"
    print(" torch.cuda.is_available()")
    print(torch.cuda.is_available())
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"[GPU {gpu_id}] Training model for freq_index {freq_index} on {device}")
    model.to(device)
    dataset_config = {
        "chedar": {
            "ply_path": "/data/fyw/dataset/dataset/chedar/cropped_chader/",
            "sofa_path": "/data/fyw/dataset/dataset/chedar/hrtfs_erb64_log/",
            "sh_path": "/data/fyw/dataset/dataset/chedar/hrtfs_erb64_log_sh/",
            "ply_pattern": "chedar_*_centered.npy",
            "sofa_pattern": "chedar_*_UV1m.npy",
            "sh_pattern": "chedar_*_UV1m.npy",
            "ply_file_name_format": "chedar_{}_centered.npy",
            "sofa_file_name_format": "chedar_{}_UV1m.npy",
            "sh_file_name_format": "chedar_{}_UV1m.npy",
            "hrtf_index": freq_index
        },
        # ?????????????...
        "hutubs": {
            "ply_path": r"C:\Users\freeway\Desktop\HRTF_dataset\HUBTUB\3D ear meshes",
            "sofa_path": r"D:\hrtf\chader\\",
            "ply_pattern": "pp*_3DheadMesh.ply",
            "sofa_pattern": "pp*_HRIRs_measured.sofa",
            "ply_file_name_format": "pp{}__3DheadMesh.ply",
            "sofa_file_name_format": "pp{}_HRIRs_measured.sofa",
            "hrtf_index": freq_index
        },
        "sonicom": {
            "ply_path": "/data/fyw/dataset/dataset/SONICOM/npy/",
            "sofa_path": "/data/fyw/dataset/dataset/SONICOM/hrtfs_mel_selected_npy_64/",
            "sh_path": "/data/fyw/dataset/dataset/SONICOM/hrtfs_7sh_mel64/",
            "ply_pattern": "adjusted_P*_watertight_centered.npy",
            "sofa_pattern": "P*_FreeFieldComp_48kHz.npy",
            "sh_pattern": "P*_FreeFieldComp_48kHz.npy",
            "ply_file_name_format": "adjusted_P{}_watertight_centered.npy",
            "sofa_file_name_format": "P{}_FreeFieldComp_48kHz.npy",
            "sh_file_name_format": "P{}_FreeFieldComp_48kHz.npy",
            "hrtf_index": freq_index
        }
    }
    print("main func in dataset.py")
    # 测试方法 使用示例: 确定文件读取路径即可
    dataset_names = ["chedar"]
    datasets = []
    print(len(datasets))
    # 为每个数据集名称创建 CustomDataset 实例
    for name in dataset_names:
        config = dataset_config[name]
        dataset = ShDataset(**config)
        datasets.append(dataset)

    # 将所有数据集组合起来
    combined_dataset = ConcatDataset(datasets)

    # 数据集划分 指定 generator以获得可复现的结果
    total_size = len(combined_dataset)
    train_size = int(total_size * 0.8)  # 80%的数据用于训练
    test_size = total_size - train_size  # 剩余20%的数据用于测试
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

    # 创建训练集和测试集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=2)

    # 创建数据加载器
    # data_loader = DataLoader(combined_dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    # 测试集评估函数

    # 使用组合后的数据集
    print("init success")
    epoch=0
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0  # 初始化训练损失
        test_loss = 'N/A'  # 初始化测试损失为'N/A'，表示未进行测试
        # 初始化计数器：保存最优训练和测试结果

        # 初始化结果保存目录
        output_dir = f"{OUTPUT_DIR_BASE}/index{freq_index}"
        log_file = f"{output_dir}/training_log.txt"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        epoch_start_time = time.time()  # 开始计时

        for batch_idx, (voxel, hrtf, sh) in enumerate(train_loader):
            # 数据加载到设备上
            voxel = voxel.unsqueeze(1).float().to(device)
            # print(hrtf.shape)
            hrtf = hrtf[:, :, 1,:].squeeze().float().to(device)
            hrtf = hrtf[:, :, freq_index].squeeze().float().to(device)

            optimizer.zero_grad()
            # print("model input shape")
            # print(np.array(voxel.cpu()).shape)
            # 30,1,32,32,32
            outputs = model(voxel)
            loss = criterion(outputs, hrtf)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * voxel.size(0)  # 累加批次损失，考虑批次大小

        train_loss /= len(train_loader.dataset)  # 计算平均训练损失
        scheduler.step()

        # 保存模型和日志
        if epoch % 5 == 0:
            # 注意：这里直接使用os.path.join来构建路径，以确保路径在不同操作系统上的兼容性
            model_save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_model_path = os.path.join(output_dir, "best_train_model.pth")
                torch.save(model.state_dict(), best_train_model_path)
                print(f"Best training model saved to {best_train_model_path}")

            if epoch % log_interval == 0:
                test_loss = evaluate_test_set(model,freq_index , test_loader, device)
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_model_path = os.path.join(output_dir, "best_test_model.pth")
                    torch.save(model.state_dict(), best_test_model_path)
                    print(f"Best testing model saved to {best_test_model_path}")

            # 记录日志
            with open(log_file, "a") as file:
                if test_loss != 'N/A':
                    file.write(
                        f"Epoch: {freq_index}_{epoch}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n")
                else:
                    file.write(f"Epoch: {freq_index}_{epoch}, Training Loss: {train_loss:.4f}\n")
        # 打印训练和测试损失
        print(
            f"Epoch: {epoch}, Training Loss: {float(train_loss):.4f}, Test Loss: {test_loss if test_loss != 'N/A' else 'N/A'}")
        epoch_time = time.time() - epoch_start_time  # Epoch总耗时
        print(f"Epoch {epoch}: Total epoch time: {epoch_time:.4f} seconds.")

def test_block_output_shape(input_shape):
    block = Block(1, 32)  # Assuming input_shape[1] represents the number of input channels
    input_tensor = torch.randn(input_shape[0], input_shape[1], input_shape[2],  input_shape[3])  # Generating random input tensor
    input_tensor = np.array([input_tensor])
    input_tensor = torch.tensor(input_tensor)
    print(input_tensor.shape)
    output_tensor = block(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)

def test_CNN_output_shape(input_shape):
    CNN_Model = CNNModel(360)  # Assuming input_shape[1] represents the number of input channels
    input_tensor = torch.randn(1, input_shape[0], input_shape[1], input_shape[2],  input_shape[3])  # Generating random input tensor
    print(input_tensor.shape)
    output_tensor = CNN_Model(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)
    
def run_train_model_on_gpu(range_start, range_end, gpu_id):
    for i in range(range_start, range_end):
        train_model(i, gpu_id)

from multiprocessing import Process
import multiprocessing as mp

if __name__ == '__main__':
    # input_shape = (1, 32, 32, 32)  # 1通道，32x32x32的体积
    # a = torch.randn(1, 32, 16, 16, 16)
    # # 将第二个维度扩展为大小为64
    # b = a.repeat(1, 64 // 32, 1, 1, 1)
    # print("原始张量形状:", a.shape)
    # print("扩展后的张量形状:", b.shape)
    # a = torch.randn(1, 2,2 )
    # # 将第二个维度扩展为大小为64
    # b = a.repeat_interleave(2, dim=1)
    # print("原始张量形状:", a)
    # print("扩展后的张量形状:", b)

    # test_block_output_shape(input_shape)
    # test_CNN_output_shape(input_shape)
    mp.set_start_method('spawn')
    processes = []
    # 为每块GPU创建一个进程

    gpu_ranges = [
    (0, 2, 0),
    (2, 4, 1),
    (4, 6, 2),
   (6, 8, 3),
   (8, 10, 0),
   (10, 12, 1),
   (12, 14, 2),
   (14, 16, 3),
   (16, 18, 0),
   (18, 20, 1),
   (20, 22, 2),
   (22, 24, 3),
   (24, 26, 0),
   (26, 28, 1),
   (28, 30, 2),
   (30, 32, 3)
   ]

    # run_train_model_on_gpu(0, 8, 0)
    for range_start, range_end, gpu_id in gpu_ranges:
        p = Process(target=run_train_model_on_gpu, args=(range_start, range_end, gpu_id))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()