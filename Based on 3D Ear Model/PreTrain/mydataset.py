import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

def generate_one_hot(frequency_index, num_classes=40):
    one_hot = np.zeros(num_classes,dtype=np.float32)
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
        mesh_id = int(round(mesh_id))
        # Ensure mesh_id is a 4-digit number with leading zeros
        mesh_id_str = str(mesh_id).zfill(4)
        mesh_path = os.path.join(self.mesh_dir, f'chedar_{mesh_id_str}.npy')


        mesh = np.load(mesh_path).astype(np.float32)
        mesh = np.expand_dims(mesh, axis=0)  # Add channel dimension

        body_params = self.annotations.iloc[idx, :10].values.astype(np.float32)

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
    # csv_file = '/data1/fsj/myHRTF/ten_measures.csv'

    mesh_dir = r'E:\QQfiles\HRTF_Datasets\CHEDAR\3Dmeshvoxel'
    csv_file = r'E:\QQfiles\HRTF_Datasets\CHEDAR\eight_measures.csv'

    # 创建数据集
    pretrain_dataset = PretrainDatasetChedar(mesh_dir, csv_file)

    # 测试数据加载器
    batch_size = 16
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

        torso_data = self.annotations.iloc[base_idx, :10].values.astype(np.float32)

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

        return torch.tensor(mesh), torch.tensor(torso_data), torch.tensor(frequency_index), torch.tensor(ear), torch.tensor(hrtf)

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
    def __init__(self, mesh_dir, csv_file, hrtf_dir, valid_indices, split_index=100):
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
        self.valid_indices = valid_indices
        self.split_index = split_index

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 如果 idx 是张量，将其转换为列表或整数
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 如果 idx 是列表，取出单个元素进行处理
        if isinstance(idx, list):
            idx = idx[0]

        # 现在 idx 是一个整数，进行后续处理
        # Determine the subject index and whether it is the right ear
        subject_idx = idx // 2
        is_right_ear = idx % 2 == 1
        base_idx = self.valid_indices[subject_idx]

        # Adjust for right ear
        mesh_id = base_idx if not is_right_ear else base_idx - self.split_index if base_idx > self.split_index else base_idx

        try:
            mesh_path = os.path.join(self.mesh_dir, f'pp{mesh_id}.npy')
            mesh = np.load(mesh_path).astype(np.float32)
            mesh = np.expand_dims(mesh, axis=0)  # Add channel dimension for compatibility with Conv layers

            torso_data = self.annotations[self.annotations['SubjectID'] == mesh_id].iloc[:, 1:14].values.astype(
                np.float32).squeeze()
            ear = np.array([1, 0] if is_right_ear else [0, 1], dtype=np.float32)  # Ensure ear data is float32

            hrtf_file_name = f'pp{mesh_id}_HRIRs_measured.npy'
            hrtf_path = os.path.join(self.hrtf_dir, hrtf_file_name)
            hrtf_data = np.load(hrtf_path)  # Assuming HRTF data shape is (40, 2, 16)

            # Select the right or left ear data
            hrtf = hrtf_data[:, 1, :] if is_right_ear else hrtf_data[:, 0, :]
            hrtf = hrtf.astype(np.float32)

            return torch.tensor(mesh), torch.tensor(torso_data), torch.tensor(ear), torch.tensor(hrtf)
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None


if __name__ == '__main__':
    test_pretrain_dataset()
