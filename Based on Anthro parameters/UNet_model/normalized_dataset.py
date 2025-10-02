import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

class CIPICNormDataset(Dataset):
    """
    CIPIC 正则化处理后结果 共37个
    """
    def __init__(self, hrtf_dir, anthro_file_path):
        self.hrtf = np.load(hrtf_dir)
        self.csv_data = pd.read_csv(anthro_file_path)

    def __len__(self):
        return self.hrtf.shape[0]

    def __getitem__(self, idx):
        subject_id = idx//2

        if idx % 2 == 0:  # 偶数索引 -> 左耳
            hrtf_data = self.hrtf[idx]
            anthro_params = self.csv_data.loc[subject_id, ['x1', 'x3', 'x12', 'l_d1', 'l_d3', 'l_d4', 'l_d5', 'l_d6']].values
            ear = 0
            # kemar_data = self.hrtf[8]
        else:  # 奇数索引 -> 右耳
            hrtf_data = self.hrtf[idx]
            anthro_params = self.csv_data.loc[subject_id, ['x1', 'x3', 'x12', 'r_d1', 'r_d3', 'r_d4', 'r_d5', 'r_d6']].values
            ear = 1
            # kemar_data = self.hrtf[9]

        # 将 numpy 数组转换为 Tensor，并增加一个维度
        hrtf_data = torch.tensor(hrtf_data, dtype=torch.float32).unsqueeze(0)
        # kemar_data = torch.tensor(kemar_data, dtype=torch.float32).unsqueeze(0)
        anthro_params = torch.tensor(anthro_params, dtype=torch.float32)

        return hrtf_data, anthro_params, ear

def test_CIPICNormDataset():
    # 创建数据集，随机选择左右耳数据
    dataset = CIPICNormDataset(hrtf_dir=r"/data/zym/dataset/normalized/cipic_normalized_hrtf.npy",
                           anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv")

    # 使用 DataLoader 测试
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 打印一些批量数据
    for batch in dataloader:
        print("Batch:", batch)

class HUTUBSNormDataset(Dataset):
    """
    HUTUBS 正则化处理后结果 共93个
    """
    def __init__(self, hrtf_dir, anthro_file_path):
        self.hrtf = np.load(hrtf_dir)
        self.csv_data = pd.read_csv(anthro_file_path)

    def __len__(self):
        return self.hrtf.shape[0]

    def __getitem__(self, idx):
        subject_id = idx//2

        if idx % 2 == 0:  # 偶数索引 -> 左耳
            hrtf_data = self.hrtf[idx]
            anthro_params = self.csv_data.loc[subject_id, ['x1', 'x3', 'x12', 'L_d1', 'L_d3', 'L_d4', 'L_d5', 'L_d6']].values
            ear = 0
            # kemar_data = self.hrtf[8]
        else:  # 奇数索引 -> 右耳
            hrtf_data = self.hrtf[idx]
            anthro_params = self.csv_data.loc[subject_id, ['x1', 'x3', 'x12', 'R_d1', 'R_d3', 'R_d4', 'R_d5', 'R_d6']].values
            ear = 1
            # kemar_data = self.hrtf[9]

        # 将 numpy 数组转换为 Tensor，并增加一个维度
        hrtf_data = torch.tensor(hrtf_data, dtype=torch.float32).unsqueeze(0)
        # kemar_data = torch.tensor(kemar_data, dtype=torch.float32).unsqueeze(0)
        anthro_params = torch.tensor(anthro_params, dtype=torch.float32)

        return hrtf_data, anthro_params, ear

def test_HUTUBSNormDataset():
    # 创建数据集，随机选择左右耳数据
    dataset = HUTUBSNormDataset(hrtf_dir=r"/data/zym/dataset/normalized/hutubs_normalized_hrtf.npy",
                           anthro_file_path=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv")

    # 使用 DataLoader 测试
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 打印一些批量数据
    for batch in dataloader:
        print("Batch:", batch)

# def create_combined_dataset(hutubs_path=r"/data/zym/dataset/normalized/hutubs_normalized_hrtf.npy",
#                             hutubs_anthro=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv",
#                             cipic_path=r"/data/zym/dataset/normalized/cipic_normalized_hrtf.npy",
#                             cipic_anthro=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv"):
#     """
#        创建合并后的数据集
#        """
#     hutubs_dataset = HUTUBSNormDataset(hrtf_dir=hutubs_path, anthro_file_path=hutubs_anthro)
#     cipic_dataset = CIPICNormDataset(hrtf_dir=cipic_path, anthro_file_path=cipic_anthro)
#     combined_dataset = ConcatDataset([hutubs_dataset, cipic_dataset])
#     return combined_dataset

class CombinedDataset(Dataset):
    def __init__(self, datasets):
        """
        初始化 CombinedDataset
        :param datasets: 一个包含多个数据集的列表
        :param kemar_data: 全局共享的 kemar_data（numpy 或 tensor）
        """
        self.concat_dataset = ConcatDataset(datasets)
        self.cipic_hrtf = np.load(r"/data/zym/dataset/normalized/cipic_full_normalized_hrtf.npy")

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        # 获取合并数据集中的样本
        hrtf_data, anthro_params, ear = self.concat_dataset[idx]

        if ear == 0:
            kemar_data = self.cipic_hrtf[8]
        elif ear == 1:
            kemar_data = self.cipic_hrtf[9]
        else:
            print("Invalid ear value")

        kemar_data = torch.tensor(kemar_data, dtype=torch.float32).unsqueeze(0)
        # 统一返回相同的 kemar_data
        return hrtf_data, anthro_params, kemar_data

def test_CombinedDataset(hutubs_path=r"/data/zym/dataset/normalized/hutubs_normalized_hrtf.npy",
                            hutubs_anthro=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv",
                             cipic_path=r"/data/zym/dataset/normalized/cipic_normalized_hrtf.npy",
                           cipic_anthro=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv"):
    hutubs_dataset = HUTUBSNormDataset(hrtf_dir=hutubs_path, anthro_file_path=hutubs_anthro)
    cipic_dataset = CIPICNormDataset(hrtf_dir=cipic_path, anthro_file_path=cipic_anthro)

    combined_dataset = CombinedDataset(datasets=[hutubs_dataset, cipic_dataset])

    # 创建 DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)

    # 测试 DataLoader
    for batch_idx, (hrtf_data, anthro_params, kemar_data) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print("HRTF Data Shape:", hrtf_data.shape)
        print("Anthro Params Shape:", anthro_params.shape)
        print("KEMAR Data Shape:", kemar_data.shape)

        # 打印一个批次的内容
        if batch_idx == 0:
            print("Sample HRTF Data:", hrtf_data[0])
            print("Sample Anthro Params:", anthro_params[0])
            print("Sample KEMAR Data:", kemar_data[0])

def create_combined_dataset(hutubs_path=r"/data/zym/dataset/normalized/hutubs_normalized_hrtf.npy",
                            hutubs_anthro=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv",
                             cipic_path=r"/data/zym/dataset/normalized/cipic_normalized_hrtf.npy",
                           cipic_anthro=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv"):
    hutubs_dataset = HUTUBSNormDataset(hrtf_dir=hutubs_path, anthro_file_path=hutubs_anthro)
    cipic_dataset = CIPICNormDataset(hrtf_dir=cipic_path, anthro_file_path=cipic_anthro)

    combined_dataset = CombinedDataset(datasets=[hutubs_dataset, cipic_dataset])

    return combined_dataset

if __name__ == "__main__":
    test_HUTUBSNormDataset()