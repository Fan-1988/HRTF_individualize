import torch
from torch.utils.data import Dataset,DataLoader
import glob
import re
import os
import trimesh
from torch.utils.data.dataset import ConcatDataset
import numpy as np
from read_sofa import read_sofa_from_npy, read_sofa
import open3d as o3d
# from DataProcess.CHEDAR.mesh2voxel import process_ply_to_npy_self
from mesh2voxel import process_ply_to_npy_self
import math


def apply_shift(npy_data, shift):
    # Initialize a shifted_data array filled with zeros
    shifted_data = np.zeros_like(npy_data)

    # Calculate the slicing indices for the original and shifted data
    source_slices = [slice(max(0, -shift[i]), min(npy_data.shape[i], npy_data.shape[i]-shift[i])) for i in range(3)]
    dest_slices = [slice(max(0, shift[i]), min(npy_data.shape[i], npy_data.shape[i]+shift[i])) for i in range(3)]

    # Using slices to copy data
    shifted_data[tuple(dest_slices)] = npy_data[tuple(source_slices)]

    return shifted_data

def convert_voxel_grid_to_features_coords(voxel_grid):
    voxels = voxel_grid
    coords_list = []
    features_list = []

    for voxel in voxels:
        # Extract voxel center coordinates
        x, y, z = voxel.grid_index
        coords_list.append([x, y, z])

        # Use 1 as a placeholder feature
        features_list.append(1)

    # Convert to Tensors
    coords = torch.tensor(coords_list, dtype=torch.float32).transpose(0, 1)
    features = torch.tensor(features_list, dtype=torch.float32).unsqueeze(1)
    return features, coords


def extract_hrtf_at_fixed_frequency(hrtf_data, hrtf_index):
    # hrtf_data: HRTF数据，假设其维度为 [方位数量, 频率数量]
    # hrtf_index: 目标频率对应的索引
    # 返回固定频率下所有方位的HRTF值，封装成一个向量
    return hrtf_data[:, hrtf_index]

class CustomDataset(Dataset):
    # 存放文件对 的变量
    def __init__(self, ply_path, sofa_path, ply_pattern, sofa_pattern, ply_file_name_format,sofa_file_name_format, hrtf_index):
        self.file_pairs = []
        self.hrtf_index = hrtf_index
        ply_filepaths = glob.glob(os.path.join(ply_path, ply_pattern))
        # print(ply_filepaths)
        # 使用正则表达式提取索引
        regex = re.compile(ply_file_name_format.format(r"(\d+)"))
        file_idxs = []
        for path in ply_filepaths:
            match = regex.search(os.path.basename(path))
            if match:
                file_idxs.append(match.group(1))
            else:
                print(f"No match found for file: {path}")  # 若没有找到匹配项，打印出该文件路径
        # print(file_idxs)
        # 根据 idx 构造文件对
        for idx in file_idxs:
            ply_file_name = ply_file_name_format.format(idx)
            sofa_file_name = sofa_file_name_format.format(idx)
            ply_filepath = os.path.join(ply_path, ply_file_name)
            sofa_filepath = os.path.join(sofa_path, sofa_file_name)
            # print("ply")
            # print(ply_filepath)
            # print("sofa")
            # print(sofa_filepath)
            if os.path.exists(ply_filepath) and os.path.exists(sofa_filepath):
                # print(ply_filepath)
                self.file_pairs.append((ply_filepath, sofa_filepath))
        print(len(self.file_pairs))
        # 加载.ply文件
        # self.input_data = self.load_ply_data(ply_filepath)
        # 加载.sofa文件
        # self.output_data = self.load_sofa_data(sofa_filepath)

    def load_ply_data(self, filepath):
        # # 使用plyfile读取.ply文件
        # ply_data = PlyData.read(filepath)
        # # 提取所需的数据 - 这里需要根据你的数据进行适当的调整
        # # 示例: 提取顶点数据
        # vertices = torch.tensor([list(vertex) for vertex in ply_data['vertex'].data])
        # print("v shape",vertices.shape)
        # face_indices = [list(face['vertex_indices']) for face in ply_data['face'].data]
        # faces = torch.tensor(face_indices, dtype=torch.long)
        # print("face shape", faces.shape)
        #
        # return vertices, faces
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()
        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,0.002)
        return convert_voxel_grid_to_features_coords(voxel_grid.get_voxels())

    def load_voxel_fromply(self, filepath):
        return process_ply_to_npy_self(filepath)

    def load_obj_data(self, filepath):
        mesh = trimesh.load(filepath)
        # 创建图结构的数据
        x = torch.tensor(mesh.vertices, dtype=torch.float)  # 顶点作为节点特征
        edge_index = torch.tensor(mesh.edges.T, dtype=torch.long)  # 边索引
        return x,edge_index

    def load_npy_data(self, filepath):
        # Load data from .npy file
        npy_data = np.load(filepath)
        # Convert NumPy array to PyTorch tensor
        tensor_data = torch.tensor(npy_data, dtype=torch.float)
        return tensor_data

    def load_npy_data_flex(self, filepath):
        # Load data from .npy file
        npy_data = np.load(filepath)

        # Generate random shift values
        shift = np.random.randint(-2, 3, size=3)  # Generates values in [-2, 2]
        # print("Random shift:", shift)

        # Apply the shift using slicing
        shifted_data = apply_shift(npy_data, shift)

        # Convert shifted NumPy array to PyTorch tensor
        tensor_data = torch.tensor(shifted_data, dtype=torch.float)

        # print("Input shape:", npy_data.shape)
        return tensor_data

    def load_sofa_data(self, filepath):
        # 加载.sofa文件
        loc, hrtf = read_sofa_from_npy(filepath)
        #print(hrtf.shape)
        # (locs, 2, 480)
        if self.hrtf_index is not None:
            # 提取固定频率下的所有位置的HRTF值，适应左右耳
            hrtf = hrtf[:, :, self.hrtf_index]  # 此处修改为正确的维度索引
        # print(self.hrtf_index)
        return torch.tensor(loc), torch.tensor(hrtf)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        obj_path, sofa_path = self.file_pairs[idx]
        # 点、 边、 loc、 hrtf
        return self.load_npy_data(obj_path),  self.load_sofa_data(sofa_path)[0], self.load_sofa_data(sofa_path)[1]
        # ply_path, sofa_path = self.file_pairs[idx]
        # return self.load_ply_data(ply_path), self.load_sofa_data(sofa_path)[0], self.load_sofa_data(sofa_path)[1]
        
class ShDataset(Dataset):
    # 存放文件对 的变量
    def __init__(self, ply_path, sofa_path, sh_path, ply_pattern, sofa_pattern, sh_pattern, ply_file_name_format, sofa_file_name_format,  sh_file_name_format, hrtf_index
    
    ):
        self.file_pairs = []
        self.hrtf_index = hrtf_index
        ply_filepaths = glob.glob(os.path.join(ply_path, ply_pattern))
        # print(ply_filepaths)
        # 使用正则表达式提取索引
        regex = re.compile(ply_file_name_format.format(r"(\d+)"))
        file_idxs = []
        for path in ply_filepaths:
            match = regex.search(os.path.basename(path))
            if match:
                file_idxs.append(match.group(1))
            else:
                print(f"No match found for file: {path}")  # 若没有找到匹配项，打印出该文件路径
        # print(file_idxs)
        # 根据 idx 构造文件对
        for idx in file_idxs:
            ply_file_name = ply_file_name_format.format(idx)
            sofa_file_name = sofa_file_name_format.format(idx)
            sh_file_name = sh_file_name_format.format(idx)
            ply_filepath = os.path.join(ply_path, ply_file_name)
            sofa_filepath = os.path.join(sofa_path, sofa_file_name)
            sh_filepath = os.path.join(sh_path,sh_file_name)
            # print("ply")
            # print(ply_filepath)
            # print("sofa")
            # print(sofa_filepath)
            if os.path.exists(ply_filepath) and os.path.exists(sofa_filepath) and os.path.exists(sh_filepath):
                # print(ply_filepath)
                self.file_pairs.append((ply_filepath, sofa_filepath, sh_filepath))
        print(len(self.file_pairs))
        # 加载.ply文件
        # self.input_data = self.load_ply_data(ply_filepath)
        # 加载.sofa文件
        # self.output_data = self.load_sofa_data(sofa_filepath)

class newShDataset(Dataset):
    def __init__(self, ply_path, sofa_path, sh_path, ply_pattern, sofa_pattern, sh_pattern,
                 ply_file_name_format, sofa_file_name_format, sh_file_name_format, hrtf_index):
        self.file_pairs = []
        self.hrtf_index = hrtf_index
        self.ply_file_names = []  # 新增：记录文件名列表
        ply_filepaths = glob.glob(os.path.join(ply_path, ply_pattern))

        regex = re.compile(ply_file_name_format.format(r"(\d+)"))
        file_idxs = []
        for path in ply_filepaths:
            match = regex.search(os.path.basename(path))
            if match:
                file_idxs.append(match.group(1))
            else:
                print(f"No match found for file: {path}")

        for idx in file_idxs:
            ply_file_name = ply_file_name_format.format(idx)
            sofa_file_name = sofa_file_name_format.format(idx)
            sh_file_name = sh_file_name_format.format(idx)
            ply_filepath = os.path.join(ply_path, ply_file_name)
            sofa_filepath = os.path.join(sofa_path, sofa_file_name)
            sh_filepath = os.path.join(sh_path, sh_file_name)

            if os.path.exists(ply_filepath) and os.path.exists(sofa_filepath) and os.path.exists(sh_filepath):
                self.file_pairs.append((ply_filepath, sofa_filepath, sh_filepath))
                self.ply_file_names.append(ply_file_name)  # 新增：记录对应的 ply 文件名
        print(f"Loaded {len(self.file_pairs)} samples.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        obj_path, sofa_path, sh_path = self.file_pairs[idx]
        voxel = self.load_npy_data(obj_path)
        hrtf = self.load_npy_sofa_data(sofa_path)
        sh = self.load_npy_sh_data(sh_path)
        filename = self.ply_file_names[idx]  # 获取文件名
        return voxel, hrtf, sh, filename  # 返回加上文件名！

    def load_ply_data(self, filepath):
        # # 使用plyfile读取.ply文件
        # ply_data = PlyData.read(filepath)
        # # 提取所需的数据 - 这里需要根据你的数据进行适当的调整
        # # 示例: 提取顶点数据
        # vertices = torch.tensor([list(vertex) for vertex in ply_data['vertex'].data])
        # print("v shape",vertices.shape)
        # face_indices = [list(face['vertex_indices']) for face in ply_data['face'].data]
        # faces = torch.tensor(face_indices, dtype=torch.long)
        # print("face shape", faces.shape)
        #
        # return vertices, faces
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()
        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,0.002)
        return convert_voxel_grid_to_features_coords(voxel_grid.get_voxels())

    def load_voxel_fromply(self, filepath):
        return process_ply_to_npy_self(filepath)

    def load_obj_data(self, filepath):
        mesh = trimesh.load(filepath)
        # 创建图结构的数据
        x = torch.tensor(mesh.vertices, dtype=torch.float)  # 顶点作为节点特征
        edge_index = torch.tensor(mesh.edges.T, dtype=torch.long)  # 边索引
        return x,edge_index
    
    # 屏蔽虚部
    def load_npy_sofa_data(self, filepath):
        # Load data from .npy file
        data_loaded = np.load(filepath, allow_pickle=True)
        # Convert NumPy array to PyTorch tensor
        # tensor_data = torch.tensor(npy_data.real, dtype=torch.float)
        return data_loaded

    def load_npy_data(self, filepath):
        data = np.load(filepath)
        return data

    def load_npy_sh_data(self, filepath):
        data = np.load(filepath)
        tensor_data = torch.tensor(data.real, dtype=torch.float)
        return tensor_data


    def load_npy_data_flex(self, filepath):
        # Load data from .npy file
        npy_data = np.load(filepath)

        # Generate random shift values
        shift = np.random.randint(-2, 3, size=3)  # Generates values in [-2, 2]
        # print("Random shift:", shift)

        # Apply the shift using slicing
        shifted_data = apply_shift(npy_data, shift)

        # Convert shifted NumPy array to PyTorch tensor
        tensor_data = torch.tensor(shifted_data, dtype=torch.float)

        # print("Input shape:", npy_data.shape)
        return tensor_data

    def load_sofa_data(self, filepath):
        # 加载.sofa文件
        loc, hrtf = read_sofa_from_npy(filepath)
        #print(hrtf.shape)
        # (locs, 2, 480)
        if self.hrtf_index is not None:
            # 提取固定频率下的所有位置的HRTF值，适应左右耳
            hrtf = hrtf[:, :, self.hrtf_index]  # 此处修改为正确的维度索引
        # print(self.hrtf_index)
        return torch.tensor(loc), torch.tensor(hrtf)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        obj_path, sofa_path, sh_path = self.file_pairs[idx]
        # 点、 边、 loc、 hrtf
        return self.load_npy_data(obj_path),  self.load_npy_sofa_data(sofa_path), self.load_npy_sh_data(sh_path)
        # ply_path, sofa_path = self.file_pairs[idx]
        # return self.load_ply_data(ply_path), self.load_sofa_data(sofa_path)[0], self.load_sofa_data(sofa_path)[1]
        
class UNetDataset(Dataset):
    # 存放文件对 的变量
    def __init__(self, ply_path, sofa_path, ply_pattern, sofa_pattern, ply_file_name_format,sofa_file_name_format, hrtf_index):
        self.file_pairs = []
        self.hrtf_index = hrtf_index
        ply_filepaths = glob.glob(os.path.join(ply_path, ply_pattern))
        # print(ply_filepaths)
        # 使用正则表达式提取索引
        regex = re.compile(ply_file_name_format.format(r"(\d+)"))
        file_idxs = []
        for path in ply_filepaths:
            match = regex.search(os.path.basename(path))
            if match:
                file_idxs.append(match.group(1))
            else:
                print(f"No match found for file: {path}")  # 若没有找到匹配项，打印出该文件路径
        # print(file_idxs)
        # 根据 idx 构造文件对
        for idx in file_idxs:
            ply_file_name = ply_file_name_format.format(idx)
            sofa_file_name = sofa_file_name_format.format(idx)
            ply_filepath = os.path.join(ply_path, ply_file_name)
            sofa_filepath = os.path.join(sofa_path, sofa_file_name)
            # print("ply")
            # print(ply_filepath)
            # print("sofa")
            # print(sofa_filepath)
            if os.path.exists(ply_filepath) and os.path.exists(sofa_filepath):
                # print(ply_filepath)
                self.file_pairs.append((ply_filepath, sofa_filepath))
        print(len(self.file_pairs))
        # 加载.ply文件
        # self.input_data = self.load_ply_data(ply_filepath)
        # 加载.sofa文件
        # self.output_data = self.load_sofa_data(sofa_filepath)

    def load_ply_data(self, filepath):
        # # 使用plyfile读取.ply文件
        # ply_data = PlyData.read(filepath)
        # # 提取所需的数据 - 这里需要根据你的数据进行适当的调整
        # # 示例: 提取顶点数据
        # vertices = torch.tensor([list(vertex) for vertex in ply_data['vertex'].data])
        # print("v shape",vertices.shape)
        # face_indices = [list(face['vertex_indices']) for face in ply_data['face'].data]
        # faces = torch.tensor(face_indices, dtype=torch.long)
        # print("face shape", faces.shape)
        #
        # return vertices, faces
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()
        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,0.002)
        return convert_voxel_grid_to_features_coords(voxel_grid.get_voxels())

    def load_voxel_fromply(self, filepath):
        return process_ply_to_npy_self(filepath)

    def load_obj_data(self, filepath):
        mesh = trimesh.load(filepath)
        # 创建图结构的数据
        x = torch.tensor(mesh.vertices, dtype=torch.float)  # 顶点作为节点特征
        edge_index = torch.tensor(mesh.edges.T, dtype=torch.long)  # 边索引
        return x,edge_index

    def load_npy_data(self, filepath):
        # Load data from .npy file
        npy_data = np.load(filepath,allow_pickle=True)
        # Convert NumPy array to PyTorch tensor
        tensor_data = torch.tensor(npy_data, dtype=torch.float)
        return tensor_data

    def load_sofa_data(self, filepath):
        # 加载.sofa文件
        loc, hrtf = read_sofa(filepath)
        #print(hrtf.shape)
        # (locs, 2, 480)
        if self.hrtf_index is not None:
            # 提取固定频率下的所有位置的HRTF值，适应左右耳
            hrtf = hrtf[:, :, self.hrtf_index]  # 此处修改为正确的维度索引
        # print(self.hrtf_index)
        return torch.tensor(loc), torch.tensor(hrtf)

    def load_sofaMatrix_data(self, filepath):
        hrtf_matrix = np.load(filepath)
        output_matrix = hrtf_matrix[:,self.hrtf_index,:,:,:,:]
        output_matrix = np.squeeze(output_matrix,axis=1)
        return output_matrix

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        obj_path, sofa_path = self.file_pairs[idx]
        # 点、 边、 loc、 hrtf
        return self.load_npy_data(obj_path),  self.load_sofaMatrix_data(sofa_path)
        # ply_path, sofa_path = self.file_pairs[idx]
        # return self.load_ply_data(ply_path), self.load_sofa_data(sofa_path)[0], self.load_sofa_data(sofa_path)[1]



def main():
    freq_index = 0
    dataset_config = {
        "chedar": {
            "ply_path": "/data/fyw/dataset/dataset/chedar/cropped_chader/",
            "sofa_path": "/data/fyw/dataset/dataset/chedar/hrtf_npy/",
            "sh_path": "/data/fyw/dataset/dataset/chedar/hrtf_sh/",
            "ply_pattern": "chedar_*_centered.npy",
            "sofa_pattern": "chedar_*_UV1m.npy",
            "sh_pattern": "chedar_*_UV1m_sh_order_7.npy",
            "ply_file_name_format": "chedar_{}_centered.npy",
            "sofa_file_name_format": "chedar_{}_UV1m.npy",
            "sh_file_name_format": "chedar_{}_UV1m_sh_order_7.npy",
            "hrtf_index" : freq_index
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
            "ply_path": "/data1/fyw/hrtf/dataset/SONICOM/cropped_ear/",
            "sofa_path": "/data1/fyw/hrtf/dataset/SONICOM/hrtf/",
            "ply_pattern": "adjusted_P*_watertight.npy",
            "sofa_pattern": "P*_FreeFieldComp_48kHz.sofa",
            "ply_file_name_format": "adjusted_P{}_watertight.npy",
            "sofa_file_name_format": "P{}_FreeFieldComp_48kHz.sofa",
            "hrtf_index": freq_index
        }
    }
    print("main func in dataset.py")
    # 测试方法 使用示例: 确定文件读取路径即可
    dataset_names = ["chedar"]
    datasets = []

    # 为每个数据集名称创建 CustomDataset 实例
    for name in dataset_names:
        config = dataset_config[name]
        dataset = ShDataset(**config)
        datasets.append(dataset)

    # 将所有数据集组合起来
    combined_dataset = ConcatDataset(datasets)

    # 创建数据加载器
    data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True, num_workers=1)

    # 使用组合后的数据集
    print("init success")

    for batch_idx, (data, hrtf, sh) in enumerate(data_loader):
        # 假设load_npy_data_flex返回了movable_range_positive和movable_range_negative
        print(sh.shape)

if __name__ == '__main__':
    main()


