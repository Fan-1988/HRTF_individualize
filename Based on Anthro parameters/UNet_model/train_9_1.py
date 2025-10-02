import os
import torch
import random
import logging
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch import nn

from dataset import CIPICDataset
from unet_model import UNet_c,MLP
from datetime import datetime



def setup_logger(log_prefix, log_dir):
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

def train_and_test(model, dataset, criterion, optimizer_class, device,
                   num_epochs, test_split=0.1, lr=0.0002, seed=42,
                   test_index_save_path="test_subjects.txt"):
    logging.info("Starting train and test process...")

    # 提取所有 subject_id 并记录原始行号
    subject_ids = [sample['subject_id'] for sample in dataset]  # 假设你的 Dataset 中每个样本含有 'subject_id'
    indices = list(range(len(subject_ids)))

    # 设置随机种子，打乱索引
    random.seed(seed)
    random.shuffle(indices)

    test_size = int(len(indices) * test_split)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]

    # 保存测试集 subject_id
    with open(test_index_save_path, 'w') as f:
        for idx in test_indices:
            f.write(f"{subject_ids[idx]}\n")
    logging.info(f"Saved test subject_id to: {test_index_save_path}")

    # 创建训练和测试集
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}]")

        # ------- Training -------
        model.train()
        running_loss = 0.0
        for sample in train_loader:
            hrtf_data = sample['hrtf'].to(device)
            anthropometric = sample['anthro'].to(device)
            kemar_data = sample['kemar'].to(device)

            output = model(kemar_data, anthropometric)
            loss = criterion(output, hrtf_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Train Loss: {avg_train_loss:.4f}")

        # ------- Testing -------
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sample in test_loader:
                hrtf_data = sample['hrtf'].to(device)
                anthropometric = sample['anthro'].to(device)
                kemar_data = sample['kemar'].to(device)

                output = model(kemar_data, anthropometric)
                loss = criterion(output, hrtf_data)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        logging.info(f"Test Loss: {avg_test_loss:.4f}")

    logging.info("Training and Testing Finished.")


def main_c():
    log_filename = setup_logger("cipic_9_1_MLP", log_dir="/data1/fsj/mag_model/log")
    print(f"Logging to file: {log_filename}")

    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CIPICDataset(
        hrtf_dir=r"/data1/fsj/mag_model/rearrange_2d_hrtf",
        anthro_file_path=r"/data1/fsj/mag_model/13anthro.csv",
        kemar_path=r"/data1/fsj/mag_model/rearrange_2d_hrtf/2d_cut_hrtf_021.mat"
    )

    # model = UNet_c().to(device)
    model = MLP().to(device)

    criterion = nn.MSELoss()
    optimizer_class = torch.optim.Adam

    train_and_test(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer_class=optimizer_class,
        device=device,
        num_epochs=30,
        test_split=0.1,
        lr=0.0002,
        test_index_save_path="/data1/fsj/mag_model/cipic_9_1_test_MLP.txt"
    )

    torch.save(model.state_dict(), '/data1/fsj/mag_model/cipic_9_1_best_model_MLP.pth')
    print("✅ 模型已保存。")

def main_c_prl():
    log_filename = setup_logger("cipic_9_1_MLP", log_dir="/data1/fsj/mag_model/log")
    print(f"Logging to file: {log_filename}")

    gpu_ids = [0, 1, 2, 3]  # 选择多个 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CIPICDataset(
        hrtf_dir=r"/data1/fsj/mag_model/rearrange_2d_hrtf",
        anthro_file_path=r"/data1/fsj/mag_model/13anthro.csv",
        kemar_path=r"/data1/fsj/mag_model/rearrange_2d_hrtf/2d_cut_hrtf_021.mat"
    )

    model = MLP()  # 使用 MLP 模型
    model = nn.DataParallel(model, device_ids=gpu_ids).to(device)  # 使用 DataParallel 将模型分配到多个 GPU

    criterion = nn.MSELoss()
    optimizer_class = torch.optim.Adam

    train_and_test(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer_class=optimizer_class,
        device=device,
        num_epochs=30,
        test_split=0.1,
        lr=0.0002,
        test_index_save_path="/data1/fsj/mag_model/cipic_9_1_test_MLP.txt"
    )

    torch.save(model.state_dict(), '/data1/fsj/mag_model/cipic_9_1_best_model_MLP.pth')
    print("✅ 模型已保存。")


if __name__ == '__main__':
    main_c_prl()
