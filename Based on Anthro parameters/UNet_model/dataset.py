import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.io import loadmat
import numpy as np
from pysofaconventions import *


class CIPICDataset(Dataset):
    def __init__(self, hrtf_dir, anthro_file_path, kemar_path):
        self.hrtf_dir = hrtf_dir
        self.anthro_file = pd.read_csv(anthro_file_path)
        self.valid_indices = self.anthro_file["subject_id"].to_list()
        self.kemar_path = kemar_path

    def __len__(self):
        return len(self.valid_indices) * 2

    def __getitem__(self, idx):
        # 确定 subject_id 和 ear（左右耳）
        subject_idx = idx // 2  # 每个 subject_id 对应两个 idx 值
        ear = 'left' if idx % 2 == 0 else 'right'  # 偶数为左耳，奇数为右耳

        subject_id = self.valid_indices[subject_idx]
        hrtf_name = f"2d_cut_hrtf_{subject_id:03d}.mat"
        hrtf_path = os.path.join(self.hrtf_dir, hrtf_name)

        # 检查文件是否存在
        if not os.path.exists(hrtf_path):
            print(f"File not found: {hrtf_path}")
            return None

        # 尝试加载 .mat 文件
        try:
            hrtf_file = loadmat(hrtf_path)
        except Exception as e:
            print(f"Error loading .mat file for subject {subject_id}: {e}")
            return None

        # 通用HRTF加载
        kemar = loadmat(self.kemar_path)

        # 根据 ear 确定列名和 HRTF 数据
        if ear == 'left':
            columns = ['x1', 'x3', 'x12', 'l_d1', 'l_d3', 'l_d4', 'l_d5', 'l_d6']
            hrtf_data = hrtf_file.get('hrtf_l_2d')
            kemar_data = kemar.get('hrtf_l_2d')
        else:
            columns = ['x1', 'x3', 'x12', 'r_d1', 'r_d3', 'r_d4', 'r_d5', 'r_d6']
            hrtf_data = hrtf_file.get('hrtf_r_2d')
            kemar_data = kemar.get('hrtf_r_2d')

        # 检查 HRTF 数据是否成功加载
        if hrtf_data is None:
            print(f"HRTF data for subject {subject_id} in {ear} ear is None.")
            return None

        # 从 CSV 数据中选取特定列的数据
        anthro_params = self.anthro_file.loc[self.anthro_file['subject_id'] == subject_id, columns].values.flatten()
        if anthro_params.size == 0:
            print(f"Anthropometric parameters for subject {subject_id} are empty.")
            return None

        # 将 numpy 数组转换为 Tensor，并增加一个维度
        hrtf_data = torch.tensor(hrtf_data, dtype=torch.float32).unsqueeze(0)
        kemar_data = torch.tensor(kemar_data, dtype=torch.float32).unsqueeze(0)
        anthro_params = torch.tensor(anthro_params, dtype=torch.float32)

        # return hrtf_data, anthro_params, kemar_data

        return {
            'hrtf': hrtf_data,
            'anthro': anthro_params,
            'kemar': kemar_data,
            'subject_id': subject_id
        }

def test_CIPICDataset():
    # 创建数据集，随机选择左右耳数据
    dataset = CIPICDataset(hrtf_dir=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf",
                           anthro_file_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/13anthro.csv",
                           kemar_path=r"/data/zym/dataset/CIPIC/rearrange_2d_hrtf/hrtf/2d_cut_hrtf_021.mat")

    # 使用 DataLoader 测试
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 打印一些批量数据
    for batch in dataloader:
        print("Batch:", batch)

class HUTUBSDataset(Dataset):
    def __init__(self, hrtf_dir, anthro_file_path,kemar_path):
        """

        :param hrtf_dir:
        :param anthro_file_path:
        :param kemar_path: 考虑到采样点位不同，这里使用的不是kemar而是hutubs里第一个人的数据
        """
        self.hrtf_dir = hrtf_dir
        self.anthro_file = pd.read_csv(anthro_file_path)
        self.valid_indices = self.anthro_file["SubjectID"].to_list()
        self.kemar_path = "/data/zym/dataset/HUTUBS/HRIRs/pp01_HRIRs_measured.sofa"

    def __len__(self):
        return len(self.valid_indices) * 2

    def __getitem__(self, idx):
        # 确定 subject_id 和 ear（左右耳）
        subject_idx = idx // 2  # 每个 subject_id 对应两个 idx 值
        ear = 'left' if idx % 2 == 0 else 'right'  # 偶数为左耳，奇数为右耳

        subject_id = self.valid_indices[subject_idx]
        hrtf_name = f"pp{subject_id:02d}_HRIRs_measured.sofa"
        hrtf_path = os.path.join(self.hrtf_dir, hrtf_name)

        # 检查文件是否存在
        if not os.path.exists(hrtf_path):
            print(f"File not found: {hrtf_path}")
            return None

        # 尝试加载 .mat 文件
        try:
            hrtf = SOFAFile(hrtf_path, 'r').getDataIR()
        except Exception as e:
            print(f"Error loading .mat file for subject {subject_id}: {e}")
            return None

        # 通用HRTF加载
        kemar = SOFAFile(self.kemar_path, 'r').getDataIR()

        # 根据 ear 确定列名和 HRTF 数据
        if ear == 'left':
            columns = ['x1', 'x3', 'x12', 'L_d1', 'L_d3', 'L_d4', 'L_d5', 'L_d6']
            hrtf_data = np.fft.fft(hrtf[:, 0, :], n=256)[:,:129]
            kemar_data = np.fft.fft(kemar[:, 0, :], n=256)[:,:129]
        else:
            columns = ['x1', 'x3', 'x12', 'R_d1', 'R_d3', 'R_d4', 'R_d5', 'R_d6']
            hrtf_data = np.fft.fft(hrtf[:, 1, :], n=256)[:,:129]
            kemar_data = np.fft.fft(kemar[:, 1, :], n=256)[:, :129]

        # 检查 HRTF 数据是否成功加载
        if hrtf_data is None:
            print(f"HRTF data for subject {subject_id} in {ear} ear is None.")
            return None

        # 从 CSV 数据中选取特定列的数据
        anthro_params = self.anthro_file.loc[self.anthro_file['SubjectID'] == subject_id, columns].values.flatten()
        if anthro_params.size == 0:
            print(f"Anthropometric parameters for subject {subject_id} are empty.")
            return None

        # 将 numpy 数组转换为 Tensor，并增加一个维度
        hrtf_data = torch.tensor(hrtf_data, dtype=torch.float32).unsqueeze(0)
        kemar_data = torch.tensor(kemar_data, dtype=torch.float32).unsqueeze(0)
        anthro_params = torch.tensor(anthro_params, dtype=torch.float32)

        return hrtf_data, anthro_params, kemar_data

def test_HUTUBSDataset():
    # 创建数据集，随机选择左右耳数据
    dataset = HUTUBSDataset(hrtf_dir=r"/data/zym/dataset/HUTUBS/HRIRs",
                           anthro_file_path=r"/data/zym/dataset/HUTUBS/AntrhopometricMeasures/AntrhopometricMeasures_prediction.csv",
                           kemar_path=r"/data/zym/dataset/HUTUBS/HRIRs/pp01_HRIRs_measured.sofa")

    # 使用 DataLoader 测试
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 打印一些批量数据
    for batch in dataloader:
        hrtf_data, anthro_params, kemar_data = batch
        print(hrtf_data.shape, anthro_params.shape, kemar_data.shape)

if __name__ == "__main__":
    hrtf = SOFAFile("/data/zym/dataset/HUTUBS/HRIRs/pp12_HRIRs_measured.sofa", 'r').getDataIR()
    print(hrtf.shape)
