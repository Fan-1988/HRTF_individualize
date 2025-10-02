import h5py
import numpy as np

def print_sofa_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        print(" SOFA 文件结构：\n")

        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                print(f"量: {name}")
                print(f"维度: {shape}, 类型: {dtype}")
                print("-" * 40)

        f.visititems(print_dataset_info)


def extract_all_hrir_channels(sofa_path, left_save_path, right_save_path):
    with h5py.File(sofa_path, 'r') as f:
        hrir_data = f['Data.IR'][:]

        # 确认维度
        M, R, N = hrir_data.shape
        assert R == 2, "不是双耳数据，通道数不为2"

        # 提取左耳（通道0）和右耳（通道1）
        left_hrir = hrir_data[:, 0, :]  # shape: (M, N)
        right_hrir = hrir_data[:, 1, :]  # shape: (M, N)

        # 保存为 .npy 文件
        np.save(left_save_path, left_hrir)
        np.save(right_save_path, right_hrir)

        print(f"已保存所有左耳 HRIR 到: {left_save_path}")
        print(f"已保存所有右耳 HRIR 到: {right_save_path}")

if __name__ == '__main__':
    sofa_file=r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR_HRTF\chedar_0001_UV1m.sofa"
    print_sofa_structure(sofa_file)

    chedar0_left_hrir_path=r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR_HRTF\chedar0_left_hrir.npy"
    chedar0_right_hrir_path=r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR_HRTF\chedar0_right_hrir.npy"

    extract_all_hrir_channels(
        sofa_path=sofa_file,
        left_save_path=chedar0_left_hrir_path,
        right_save_path=chedar0_right_hrir_path
    )



