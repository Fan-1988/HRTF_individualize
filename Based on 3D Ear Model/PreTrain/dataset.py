import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np


def generate_one_hot(frequency_index, num_classes=40):
    one_hot = np.zeros(num_classes)
    one_hot[frequency_index] = 1
    return one_hot


class PretrainDatasetHutubs(Dataset):
    def __init__(self, mesh_dir, csv_file):
        """
        Args:
            mesh_dir (string): Directory with all the 3D mesh files.
            csv_file (string): Path to the csv file with annotations.
        """
        self.mesh_dir = mesh_dir
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mesh_id = self.annotations.iloc[idx, 0]
        mesh_path = os.path.join(self.mesh_dir, f'pp{mesh_id}.npy')
        mesh = np.load(mesh_path).astype(np.float32)
        mesh = np.expand_dims(mesh, axis=0)  # Add channel dimension

        body_params = self.annotations.iloc[idx, 1:].values.astype(np.float32)

        return torch.tensor(mesh), torch.tensor(body_params)


class PretrainDatasetChedar(Dataset):
    def __init__(self, mesh_dir, csv_file):
        """
        Args:
            mesh_dir (string): Directory with all the 3D mesh files.
            csv_file (string): Path to the csv file with annotations.
        """
        self.mesh_dir = mesh_dir
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mesh_id = self.annotations.iloc[idx, 0]
        # Ensure mesh_id is a 4-digit number with leading zeros
        mesh_id = int(round(mesh_id))
        mesh_id_str = str(mesh_id).zfill(4)
        mesh_path = os.path.join(self.mesh_dir, f'chedar_{mesh_id_str}.npy')
        mesh = np.load(mesh_path).astype(np.float32)
        mesh = np.expand_dims(mesh, axis=0)  # Add channel dimension

        # body_params = self.annotations.iloc[idx, 1:].values.astype(np.float32)
        # 只获取从第1列到第10列的参数
        body_params = self.annotations.iloc[idx, 1:11].values.astype(np.float32)

        return torch.tensor(mesh), torch.tensor(body_params)


def test_pretrain_dataset():
    # # pretrain_dataset hutubs 测试
    # dataset = PretrainDatasetHutubs('/data/zym/dataset/HUTUBS/3Dmesh/',
    #                           '/data/zym/dataset/HUTUBS/Antrhopometric measures/pretrain.csv')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    #
    # for i, data in enumerate(dataloader):
    #     meshes, body_params = data
    #     print(f"Mesh shape: {meshes.shape}, Body params shape: {body_params.shape}")
    #     if i == 1:
    #         break

    # pretrain dataset chedar测试
    # mesh_dir = '/data1/fsj/myHRTF/3Dmeshvoxel'
    # csv_file = '/data1/fsj/myHRTF/measures.csv'

    mesh_dir = r'E:\QQfiles\HRTF_Datasets\CHEDAR\3Dmeshvoxel'
    csv_file = r'E:\QQfiles\HRTF_Datasets\CHEDAR\ten_measures.csv'

    # 创建数据集ccd
    pretrain_dataset = PretrainDatasetChedar(mesh_dir, csv_file)

    # 测试数据加载器
    batch_size = 4
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

    # 测试加载一个batch的数据
    for i, (mesh, body_params) in enumerate(pretrain_loader):
        if mesh is None or body_params is None:
            continue
        print(f"Batch {i + 1}")
        print(f"Mesh shape: {mesh.shape}")  # Expected: [batch_size, 1, 32, 32, 32]
        print(f"Body Params shape: {body_params.shape}")  # Expected: [batch_size, 8] (or the number of body parameters)
        break  # 只测试一个batch的数据加载

    # 打印数据集长度
    print(f"Dataset Length: {len(pretrain_dataset)}")


class ChedarDataset(Dataset):
    def __init__(self, mesh_dir, csv_file, hrtf_dir, hrtf_indices, split_index=1253):
        """
        Args:
            mesh_dir (string): Directory with all the 3D mesh files.
            csv_file (string): Path to the csv file with annotations.
            hrtf_dir (string): Directory with all the HRTF files.
            hrtf_indices (list): List of HRTF indices for each data sample.
            split_index (int): Index to split left and right ear (default: 1253 for CHEDAR).
        """
        self.mesh_dir = mesh_dir
        self.annotations = pd.read_csv(csv_file)
        self.hrtf_dir = hrtf_dir
        self.hrtf_indices = hrtf_indices
        self.split_index = split_index

    def __len__(self):
        return len(self.annotations) * 2  # Each subject has both left and right ear data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_idx = idx % len(self.annotations)
        is_right_ear = idx >= len(self.annotations)

        mesh_id = self.annotations.iloc[base_idx, 0]
        mesh_id_str = str(mesh_id).zfill(4)
        mesh_path = os.path.join(self.mesh_dir, f'chedar_{mesh_id_str}.npy')
        mesh = np.load(mesh_path).astype(np.float32)
        mesh = np.expand_dims(mesh, axis=0)  # Add channel dimension for compatibility with Conv layers

        torso_data = self.annotations.iloc[base_idx, 1:9].values.astype(np.float32)

        # Determine if the index corresponds to a left or right ear based on mesh_id
        ear = [0, 1] if mesh_id <= self.split_index else [1, 0]

        # Get the frequency index from the input list
        frequency_index = self.hrtf_indices[base_idx]

        # Construct the path to the specific HRTF file for this subject
        hrtf_file_name = f'chedar_{mesh_id_str}_UV1m.npy'
        hrtf_path = os.path.join(self.hrtf_dir, hrtf_file_name)
        hrtf_data = np.load(hrtf_path)  # Assuming HRTF data shape is (40, 2, 16)
        hrtf = hrtf_data[frequency_index, 1, :] if is_right_ear else hrtf_data[frequency_index, 0, :]
        hrtf = hrtf.astype(np.float32)

        return torch.tensor(mesh), torch.tensor(torso_data), torch.tensor(frequency_index), torch.tensor(
            ear), torch.tensor(hrtf)


def test_chedar_dataset():
    mesh_dir = '/data/zym/dataset/CHEDAR/3Dmesh/'
    csv_file = '/data/zym/dataset/CHEDAR/AntrhopometricMeasures/AntrhopometricMeasures.csv'
    hrtf_dir = '/data/zym/dataset/CHEDAR/HRTF_SH/'

    # 读取CSV文件以获取其长度
    annotations = pd.read_csv(csv_file)
    hrtf_indices = [np.random.randint(0, 40) for _ in range(len(annotations) * 2)]

    # 创建数据集
    chedar_dataset = ChedarDataset(mesh_dir, csv_file, hrtf_dir, hrtf_indices)

    # 测试数据加载器
    batch_size = 2
    chedar_loader = DataLoader(chedar_dataset, batch_size=batch_size, shuffle=True)

    # 测试加载一个batch的数据
    for i, (mesh, torso_data, frequency_data, ear_data, hrtf) in enumerate(chedar_loader):
        print(f"Batch {i + 1}")
        print(f"Mesh shape: {mesh.shape}")  # Expected: [batch_size, 1, 32, 32, 32]
        print(f"Torso Data shape: {torso_data.shape}")  # Expected: [batch_size, 8]
        print(f"Frequency Index shape: {frequency_data.shape}")  # Expected: [batch_size]
        print(f"Ear Data shape: {ear_data.shape}")  # Expected: [batch_size, 2]
        print(f"HRTF shape: {hrtf.shape}")  # Expected: [batch_size, 16]
        break  # 只测试一个batch的数据加载

    # 打印数据集长度
    print(f"Dataset Length: {len(chedar_dataset)}")


class HutubsDataset(Dataset):
    def __init__(self, mesh_dir, csv_file, hrtf_dir, hrtf_indices, valid_indices, split_index=100):
        """
        Args:
            mesh_dir (string): Directory with all the 3D mesh files.
            csv_file (string): Path to the csv file with annotations.
            hrtf_dir (string): Directory with all the HRTF files.
            hrtf_indices (list): List of HRTF indices for each data sample.
            valid_indices (list): List of valid indices.
            split_index (int): Index to split left and right ear (default: 100 for HUTUBS).
        """
        self.mesh_dir = mesh_dir
        self.annotations = pd.read_csv(csv_file)
        self.hrtf_dir = hrtf_dir
        self.hrtf_indices = hrtf_indices
        self.valid_indices = valid_indices
        self.split_index = split_index

    def __len__(self):
        return len(self.valid_indices) * 2 * 40  # Each subject has both left and right ear data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Determine the frequency band and subject index
        frequency_band = idx % 40
        subject_idx = idx // 40
        base_idx = self.valid_indices[subject_idx // 2]
        is_right_ear = subject_idx % 2 == 1

        # Determine the mesh_id for the right ear based on the left ear index
        mesh_id = base_idx if not is_right_ear else base_idx + self.split_index

        try:
            mesh_path = os.path.join(self.mesh_dir, f'pp{mesh_id}.npy')
            print(f"Loading mesh from {mesh_path}")  # Debugging output
            mesh = np.load(mesh_path).astype(np.float32)
            mesh = np.expand_dims(mesh, axis=0)  # Add channel dimension for compatibility with Conv layers

            torso_data = self.annotations[self.annotations['SubjectID'] == mesh_id].iloc[:, 1:9].values.astype(
                np.float32).squeeze()
            ear = [1, 0] if is_right_ear else [0, 1]

            frequency_data = generate_one_hot(frequency_band, 40)

            hrtf_file_name = f'pp{mesh_id}_HRIRs_measured.npy'
            hrtf_path = os.path.join(self.hrtf_dir, hrtf_file_name)
            print(f"Loading HRTF from {hrtf_path}")  # Debugging output
            hrtf_data = np.load(hrtf_path)  # Assuming HRTF data shape is (40, 2, 16)

            # Extract the corresponding HRTF data based on ear and frequency index
            hrtf = hrtf_data[frequency_band, 1, :] if is_right_ear else hrtf_data[frequency_band, 0, :]
            hrtf = hrtf.astype(np.float32)

            return (torch.tensor(mesh), torch.tensor(torso_data), torch.tensor(frequency_data),
                    torch.tensor(ear), torch.tensor(hrtf))
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None


def test_hutubs_dataset():
    # 使用示例
    mesh_dir = '/data/zym/dataset/HUTUBS/3Dmesh/'
    csv_file = "/data/zym/dataset/HUTUBS/Antrhopometric measures/AntrhopometricMeasures.csv"
    hrtf_dir = '/data/zym/dataset/HUTUBS/HRTF_SH/'
    valid_indices = [
        1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 16, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 40, 41, 44, 45, 46, 47, 48, 49,
        55, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 80, 81, 82, 88, 89, 90, 91, 95, 96,
        101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 116, 119, 120, 121, 122, 123, 129, 130, 131, 132, 133,
        140,
        141, 144, 145, 146, 147, 148, 149, 155, 157, 158, 159, 160, 161, 162, 163, 166, 167, 168, 169, 170, 171, 172,
        173,
        176, 177, 178, 180, 181, 182, 188, 189, 190, 191, 195, 196
    ]

    # 读取CSV文件以获取其长度
    annotations = pd.read_csv(csv_file)
    hrtf_indices = [np.random.randint(0, 40) for _ in range(len(valid_indices) * 2)]

    # 创建数据集
    hutubs_dataset = HutubsDataset(mesh_dir, csv_file, hrtf_dir, hrtf_indices, valid_indices)

    # 测试数据加载器
    batch_size = 100
    hutubs_loader = DataLoader(hutubs_dataset, batch_size=batch_size, shuffle=True)

    # 测试加载一个batch的数据
    for i, (mesh, torso_data, frequency_data, ear_data, hrtf) in enumerate(hutubs_loader):
        print(f"Batch {i + 1}")
        print(f"Mesh shape: {mesh.shape}")  # Expected: [batch_size, 1, 32, 32, 32]
        print(f"Torso Data shape: {torso_data.shape}")  # Expected: [batch_size, 8]
        print(f"Frequency Index shape: {frequency_data.shape}")  # Expected: [batch_size]
        print(f"Ear Data shape: {ear_data.shape}")  # Expected: [batch_size, 2]
        print(f"HRTF shape: {hrtf.shape}")  # Expected: [batch_size, 16]
        break  # 只测试一个batch的数据加载

    # 打印数据集长度
    print(f"Dataset Length: {len(hutubs_dataset)}")


if __name__ == '__main__':
    test_pretrain_dataset()
