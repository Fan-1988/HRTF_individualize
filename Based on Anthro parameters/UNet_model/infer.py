import matplotlib.pyplot as plt
import os
import torch
import logging
from torch.utils.data import DataLoader
from unet_model import UNet_h, UNet_c
from dataset import CIPICDataset,HUTUBSDataset

def infer_one(model, dataset, device, model_path, batch_size=5):
    """
    推理函数，加载训练好的模型并进行推理，输出 HRTF 结果并绘制图像
    参数：
        model: 神经网络模型
        dataset: 数据集
        device: 运行设备 ("cpu" 或 "cuda")
        model_path: 训练好的模型文件路径
        batch_size: 每个批次的大小
    """
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置模型为评估模式

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(dataloader):
            # 将数据加载到设备
            hrtf_data = hrtf_data.to(device)
            anthropometric = anthropometric.to(device)
            kemar_data = kemar_data.to(device)

            # 前向传播
            output = model(kemar_data, anthropometric)

            # 转移到 CPU 并转换为 numpy 数组
            hrtf_data = hrtf_data.cpu().numpy()  # 维度 (batch_size, 129, 1250, 1)
            output = output.cpu().numpy()  # 维度 (batch_size, 129, 1250, 1)

            # 选择第一个样本进行绘制
            hrtf_sample = hrtf_data[0].squeeze()  # 形状变为 (129, 1250)
            output_sample = output[0].squeeze()  # 形状变为 (129, 1250)

            # 绘制 HRTF 图像
            plt.figure(figsize=(12, 6))

            # 原始 HRTF 数据
            plt.subplot(1, 2, 1)
            plt.imshow(hrtf_sample, aspect='auto', cmap='jet', origin='lower')
            plt.title('Original HRTF')
            plt.xlabel('Time / Angle')
            plt.ylabel('Frequency')
            plt.colorbar()

            # 模型预测的 HRTF 数据
            plt.subplot(1, 2, 2)
            plt.imshow(output_sample, aspect='auto', cmap='jet', origin='lower')
            plt.title('Predicted HRTF')
            plt.xlabel('Time / Angle')
            plt.ylabel('Frequency')
            plt.colorbar()

            plt.tight_layout()
            plt.show()

            # 这里只做一次推理，若需要处理更多数据可以继续迭代
            break

def infer(model, dataset, device, model_path, batch_size=5):
    """
    推理函数，加载训练好的模型并进行推理，输出 HRTF 结果并计算平均 RMSE 和 LSD。
    """
    import numpy as np

    # 加载模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_rmse = 0
    total_lsd = 0
    count = 0
    first_batch_done = False

    with torch.no_grad():
        for batch_idx, (hrtf_data, anthropometric, kemar_data) in enumerate(dataloader):
            hrtf_data = hrtf_data.to(device)
            anthropometric = anthropometric.to(device)
            kemar_data = kemar_data.to(device)

            output = model(kemar_data, anthropometric)

            hrtf_data = hrtf_data.cpu().numpy()     # shape: (B, 1, 440, 129)
            output = output.cpu().numpy()           # same shape

            batch_size = output.shape[0]
            for i in range(batch_size):
                pred = output[i].squeeze()          # (440, 129)
                target = hrtf_data[i].squeeze()     # (440, 129)

                # RMSE
                rmse = np.sqrt(np.mean((pred - target) ** 2))

                # LSD
                eps = 1e-8
                log_pred = np.log10(np.abs(pred) + eps)
                log_target = np.log10(np.abs(target) + eps)
                lsd = np.sqrt(np.mean((log_pred - log_target) ** 2))

                total_rmse += rmse
                total_lsd += lsd
                count += 1

                # 只画第一个样本
                if not first_batch_done and i == 0:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 6))

                    plt.subplot(1, 2, 1)
                    plt.imshow(target, aspect='auto', cmap='jet', origin='lower')
                    plt.title('Original HRTF')
                    plt.xlabel('Time / Angle')
                    plt.ylabel('Frequency')
                    plt.colorbar()

                    plt.subplot(1, 2, 2)
                    plt.imshow(pred, aspect='auto', cmap='jet', origin='lower')
                    plt.title('Predicted HRTF')
                    plt.xlabel('Time / Angle')
                    plt.ylabel('Frequency')
                    plt.colorbar()

                    plt.tight_layout()
                    plt.show()
                    first_batch_done = True

    avg_rmse = total_rmse / count
    avg_lsd = total_lsd / count

    print(f"\n✅ 所有样本推理完成，共计 {count} 个样本")
    print(f"✅ 平均 RMSE: {avg_rmse:.6f}")
    print(f"✅ 平均 LSD : {avg_lsd:.6f}")


def setup_logger(log_name, log_dir):
    """
    设置日志记录器
    """
    log_filename = f"{log_dir}/{log_name}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return log_filename

def main():
    # 设置日志
    log_filename = setup_logger("mag_cipic_infer_log", log_dir="/data/zym/hrtf_prediction/2023_caai/log")
    print(f"Logging to file: {log_filename}")

    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    dataset = CIPICDataset(hrtf_dir=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/",
                           anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv",
                           kemar_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/2d_cut_hrtf_021.mat")
    # 加载数据集
    # dataset = HUTUBSDataset(
    #     hrtf_dir=r"/data/zym/dataset/HUTUBS/HRIRs/",
    #     anthro_file_path=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv",
    #     kemar_path="/data/zym/dataset/HUTUBS/HRIRs/pp01_HRIRs_measured.sofa"
    # )

    # 定义模型
    model = UNet_c().to(device)

    # 加载训练好的模型权重
    model_path = '/data/zym/hrtf_prediction/2023_caai/mag_model/5fold_cipic_hrtf_model_new.pth'
    print(f"Loading model from {model_path}")

    # 调用推理函数
    infer(model=model, dataset=dataset, device=device, model_path=model_path)

def check():
    # 加载模型
    model = UNet_h()
    model_path = '/data/zym/hrtf_prediction/2023_caai/mag_model/5fold_hrtf_model.pth'
    model.load_state_dict(torch.load(model_path))
    # 查看模型的结构
    print(model)

if __name__ == "__main__":
    # check()
    main()
