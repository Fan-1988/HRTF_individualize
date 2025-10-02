import torch
import torch.optim as optim
import torch.nn as nn
import os
import logging
from torch.utils.data import DataLoader


from LossCos import CosineSimilarityLoss
from dataset import PretrainDatasetChedar
from pretrain_model import PretrainModel
from torch.utils.data import Dataset, DataLoader, random_split


# 设置日志记录
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 添加处理器到 logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def select_device(device_id=None):
    if torch.cuda.is_available():
        if device_id is not None and device_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(device)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def train_pretrain_model():
    torch.cuda.empty_cache()
    log_dir ='/data1/fsj/myHRTF/PretrainLog'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pretrain_log.txt')
    logger = setup_logger(log_file)  # 设置日志记录

    device = select_device(device_id=1)
    # device = torch.device('cpu')
    # print('Using CPU')
    learning_rate = 0.001
    num_epochs = 15
    batch_size = 4


    # 创建数据集
    dataset = PretrainDatasetChedar('/data1/fsj/myHRTF/3Dmeshvoxel',
                                    '/data1/fsj/myHRTF/measures.csv')
    print('dataset chedar has been created')

    # 划分训练集和验证集
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f'train size: {train_size}  val size:{val_size}')

    model = PretrainModel().to(device)
    # criterion = nn.MSELoss().to(device)
    criterion = CosineSimilarityLoss().to(device)  # 使用余弦相似度损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            mesh, body_params = data
            mesh, body_params = mesh.to(device), body_params.to(device)
            optimizer.zero_grad()
            features_params = model(mesh)
            loss = criterion(features_params, body_params)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'data num in trainloader: {i}')

        avg_train_loss = running_loss / len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mesh, body_params in val_loader:
                mesh, body_params = mesh.to(device), body_params.to(device)
                features_params = model(mesh)
                loss = criterion(features_params, body_params)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f'Epoch [{epoch + 1}/{num_epochs}] finished, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print('this epoch finished')

    print('Pretraining Finished')
    save_dir = '/data1/fsj/myHRTF/PretrainModels'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.feature_extraction.state_dict(), os.path.join(save_dir, 'pretrained_model_chedar.pth'))
    print('Model saved to', os.path.join(save_dir, 'pretrained_model_chedar.pth'))


if __name__ == "__main__":
    train_pretrain_model()
