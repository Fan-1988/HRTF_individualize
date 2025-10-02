import logging
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
from unet_model import UNet_h,UNet_c
from dataset import CIPICDataset,HUTUBSDataset
from sklearn.model_selection import KFold
import torch
import os
from datetime import datetime

# 初始化日志记录器
def setup_logger(log_prefix, log_dir="logs"):
    # 获取当前时间作为日志文件名的一部分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")

    # 设置日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return log_filename

def test(model, dataloader, criterion, device):
    """
    测试函数，用于在验证集上评估模型性能
    参数：
        model: 神经网络模型
        dataloader: 数据加载器 (DataLoader)
        criterion: 损失函数
        device: 运行设备 ("cpu" 或 "cuda")
    """
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0  # 用于累积测试集上的损失

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(dataloader):
            # 将数据加载到设备
            hrtf_data = hrtf_data.to(device)
            anthropometric = anthropometric.to(device)
            kemar_data = kemar_data.to(device)

            # 前向传播
            output = model(kemar_data, anthropometric)

            # 计算损失
            loss = criterion(output, hrtf_data)
            test_loss += loss.item()

    # 计算平均测试损失
    avg_test_loss = test_loss / len(dataloader)
    return avg_test_loss

def train_and_test(model, dataset, criterion, optimizer_class, device, num_epochs, test_split=0.2, lr=0.0002):
    """
    单次训练和测试函数，将数据集划分为训练集和测试集，每个 epoch 后进行测试。
    参数：
        model: 待训练的神经网络模型
        dataset: 数据集
        criterion: 损失函数
        optimizer_class: 优化器的类（如 torch.optim.Adam）
        device: 运行设备 ("cpu" 或 "cuda")
        num_epochs: 训练的轮数
        test_split: 测试集所占比例
        lr: 学习率
    """
    logging.info("Starting train and test process...")

    # 将数据集按比例划分为训练集和测试集
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # 初始化模型和优化器
    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-4)

    # 训练和测试过程
    for epoch in range(num_epochs):
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}]")

        # 训练阶段
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(train_loader):
            # 将数据加载到设备
            hrtf_data = hrtf_data.to(device)
            anthropometric = anthropometric.to(device)
            kemar_data = kemar_data.to(device)

            # 前向传播
            output = model(kemar_data, anthropometric)
            loss = criterion(output, hrtf_data)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # 检查梯度（确保梯度不为 None）
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         logging.info(f"Layer {name}: gradient mean {param.grad.mean():.4e}")

        running_loss += loss.item()

        # 计算并打印当前 epoch 的平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Train Loss: {avg_train_loss:.4f}")

        # 测试阶段
        model.eval()  # 设置模型为评估模式
        test_loss = 0.0
        with torch.no_grad():  # 关闭梯度计算
            for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(test_loader):
                # 将数据加载到设备
                hrtf_data = hrtf_data.to(device)
                anthropometric = anthropometric.to(device)
                kemar_data = kemar_data.to(device)

                # 前向传播并计算损失
                output = model(kemar_data, anthropometric)
                loss = criterion(output, hrtf_data)
                test_loss += loss.item()

        # 计算并打印平均测试损失
        avg_test_loss = test_loss / len(test_loader)
        logging.info(f"Test Loss: {avg_test_loss:.4f}")

    logging.info("Training and Testing Finished.")

def train_with_k_fold(model, dataset, criterion, optimizer_class, device, num_epochs, k_folds=5):
    """
    使用 K 折交叉验证训练模型
    参数：
        model: 待训练的神经网络模型
        dataset: 数据集
        criterion: 损失函数
        optimizer_class: 优化器的类（如 torch.optim.Adam）
        device: 运行设备 ("cpu" 或 "cuda")
        num_epochs: 训练的轮数
        k_folds: K 折交叉验证的折数
    """
    model.to(device)  # 将模型转移到指定设备
    kfold = KFold(n_splits=k_folds, shuffle=True)  # 初始化 K 折

    # 记录每折的验证损失
    fold_results = []
    logging.info("K-Fold: ")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        logging.info(f'Fold {fold + 1}/{k_folds}')

        # 每折重新初始化模型和优化器
        model_copy = model.to(device)  # 复制模型到设备
        optimizer = optimizer_class(model_copy.parameters(), lr=0.0005, weight_decay=1e-4)

        # 创建每折的训练和验证数据加载器
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)
        train_loader = DataLoader(train_subsampler, batch_size=5, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=5, shuffle=False)

        # 进行 num_epochs 训练
        for epoch in range(num_epochs):
            model_copy.train()  # 设置模型为训练模式
            running_loss = 0.0

            for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(train_loader):
                # 将数据加载到设备
                hrtf_data = hrtf_data.to(device)
                anthropometric = anthropometric.to(device)
                kemar_data = kemar_data.to(device)

                print(
                    f"[DEBUG] kemar: {kemar_data.device}, anthropometric: {anthropometric.device}, hrtf: {hrtf_data.device}, model: {next(model_copy.parameters()).device}")

                # 前向传播
                output = model_copy(kemar_data, anthropometric)
                loss = criterion(output, hrtf_data)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 计算并打印当前 epoch 的平均训练损失
            avg_train_loss = running_loss / len(train_loader)
            logging.info(f"Fold [{fold + 1}/{k_folds}], Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # 每个 epoch 结束后在验证集上评估模型
            model_copy.eval()  # 设置模型为评估模式
            val_loss = 0.0
            with torch.no_grad():  # 关闭梯度计算
                for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(val_loader):
                    # 将数据加载到设备
                    hrtf_data = hrtf_data.to(device)
                    anthropometric = anthropometric.to(device)
                    kemar_data = kemar_data.to(device)

                    # 前向传播并计算损失
                    output = model_copy(kemar_data, anthropometric)
                    loss = criterion(output, hrtf_data)
                    val_loss += loss.item()

            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Fold [{fold + 1}], Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # 保存每折的验证损失结果
        fold_results.append(avg_val_loss)

    # 计算并打印所有折的平均验证损失
    avg_fold_loss = sum(fold_results) / k_folds
    logging.info(f"\nK-Fold Cross Validation Results: {k_folds} folds")
    logging.info(f"Average Validation Loss: {avg_fold_loss:.4f}")

    logging.info("K-Fold Training and Validation Finished.")

def main_h():
    #需要修改：log文件名；训练函数调用；保存模型文件名

    log_filename = setup_logger("mag_hutubs_training_log_5fold_new",log_dir="/data/zym/hrtf_prediction/2023_caai/log")
    print(f"Logging to file: {log_filename}")

    # 指定使用的 GPU ID，例如 "0" 或 "1"（根据你的 GPU 配置选择）
    gpu_id = 0  # 你可以修改这里的值，指定要使用的显卡 ID

    # 设置环境变量 CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 使用指定的显卡进行计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # 加载数据集
    # dataset = CIPICDataset(hrtf_dir=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/",
    #                        anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv",
    #                        kemar_path=r"/data/zym/dataset/HUTUBS/HRIRs/pp01_HRIRs_measured.sofa")
    dataset = HUTUBSDataset(hrtf_dir=r"/data/zym/dataset/HUTUBS/HRIRs/",
                            anthro_file_path=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv",
                            kemar_path="/data/zym/dataset/HUTUBS/HRIRs/pp01_HRIRs_measured.sofa")

    # 定义模型、损失函数和优化器
    model = UNet_h().to(device)
    criterion = nn.MSELoss()
    optimizer_class = torch.optim.Adam

    # 训练和测试
    # train_and_test(model=model,
    #                dataset=dataset,
    #                criterion=criterion,
    #                optimizer_class=optimizer_class,
    #                device=device,
    #                num_epochs=10,
    #                test_split=0.2,
    #                lr = 0.0002)

    # # K 折交叉验证训练
    train_with_k_fold(model=model,
                      dataset=dataset,
                      criterion=criterion,
                      optimizer_class=optimizer_class,
                      device=device,
                      num_epochs=10,
                      k_folds=5)

    torch.save(model.state_dict(), '/data/zym/hrtf_prediction/2023_caai/mag_model/5fold_hutubs_hrtf_model_new.pth')

def main_c():
    # 需要修改：log文件名；训练函数调用；保存模型文件名

    log_filename = setup_logger("mag_cipic_training_log_5fold_new",log_dir="/data/zym/hrtf_prediction/2023_caai/log")
    print(f"Logging to file: {log_filename}")

    # 指定使用的 GPU ID，例如 "0" 或 "1"（根据你的 GPU 配置选择）
    gpu_id = 3  # 你可以修改这里的值，指定要使用的显卡 ID

    # 设置环境变量 CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 使用指定的显卡进行计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    # dataset = CIPICDataset(hrtf_dir=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/",
    #                        anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv",
    #                        kemar_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/2d_cut_hrtf_021.mat")

    dataset = CIPICDataset(hrtf_dir=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf",
                           anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv",
                           kemar_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/2d_cut_hrtf_021.mat")

    # 定义模型、损失函数和优化器
    # model = normalized_UNet().to(device)
    model = UNet_c().to(device)
    criterion = nn.MSELoss()
    optimizer_class = torch.optim.Adam

    # 训练和测试
    # train_and_test(model=model,
    #                dataset=dataset,
    #                criterion=criterion,
    #                optimizer_class=optimizer_class,
    #                device=device,
    #                num_epochs=20,
    #                test_split=0.2,
    #                lr=0.0002)

    # # K 折交叉验证训练
    train_with_k_fold(model=model,
                      dataset=dataset,
                      criterion=criterion,
                      optimizer_class=optimizer_class,
                      device=device,
                      num_epochs=30,
                      k_folds=5)
    torch.save(model.state_dict(), '/data/zym/hrtf_prediction/2023_caai/mag_model/5fold_cipic_hrtf_model_new.pth')

def RMSE_LSD():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = UNet_h()  # 初始化模型结构
    model.load_state_dict(torch.load('/data/zym/hrtf_prediction/2023_caai/mag_model/5fold_hrtf_model.pth'))
    model.to(device)
    model.eval()  # 设置为评估模式

    # 加载数据集
    dataset = CIPICDataset(
        hrtf_dir=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/",
        anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv",
        kemar_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/2d_cut_hrtf_021.mat"
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 表示一次处理一个样本

    # 初始化用于计算平均值的总和
    total_rmse_per_direction = []
    total_lsd_per_direction = []
    num_samples = 0  # 样本总数

    # 遍历数据集，计算每个样本的 RMSE 和 LSD 并累加
    for hrtf_data, anthropometric, kemar_data in data_loader:
        # 将数据加载到设备
        hrtf_data = hrtf_data.to(device)  # 真实 HRTF 数据
        anthropometric = anthropometric.to(device)  # 人体测量数据
        kemar_data = kemar_data.to(device)  # KEMAR 数据

        # 进行预测
        with torch.no_grad():
            predicted_hrtf = model(kemar_data, anthropometric)

        # 使用 KEMAR 数据代替预测值
        # predicted_hrtf = kemar_data.squeeze()  # KEMAR 作为“generic”方法的预测值
        hrtf_data = hrtf_data.squeeze()  # 真实 HRTF

        # Step 1: 计算 RMSE(m) 并累加
        squared_error = (predicted_hrtf - hrtf_data) ** 2
        mean_squared_error_per_freq = torch.mean(squared_error, dim=1)  # 先对频率维度求均值
        rmse_m = 20 * torch.log10(torch.sqrt(mean_squared_error_per_freq + 1e-8))  # 对每个方向取平方根再取 log10
        total_rmse_per_direction.append(rmse_m)  # 存储每个方向的 RMSE(m)

        # Step 2: 计算 LSD(m) 并累加
        ratio = (torch.abs(predicted_hrtf) + 1e-8) / (torch.abs(hrtf_data) + 1e-8)
        log_ratio = 20 * torch.log10(ratio)
        lsd_m = torch.sqrt(torch.mean(log_ratio ** 2, dim=1))  # 对频率维度求平方均值再取平方根
        total_lsd_per_direction.append(lsd_m)  # 存储每个方向的 LSD(m)

        num_samples += 1  # 增加样本计数

    # 将每个方向的 RMSE(m) 和 LSD(m) 堆叠为张量并对方向维度取平均
    average_rmse = torch.stack(total_rmse_per_direction, dim=1).mean(dim=1).mean().item()
    average_lsd = torch.stack(total_lsd_per_direction, dim=1).mean(dim=1).mean().item()

    print(f"Average RMSE (RMSE_ave) over all samples and directions: {average_rmse}")
    print(f"Average LSD (LSD_ave) over all samples and directions: {average_lsd}")


if __name__ == '__main__':
    main_c()