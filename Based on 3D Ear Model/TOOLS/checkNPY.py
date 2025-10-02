import os
import numpy as np


def print_npy_info(directory):
    # 遍历指定目录下的文件
    np.set_printoptions(threshold=np.inf, linewidth=500)

    for filename in os.listdir(directory):
        # 只处理.npy文件
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)

            try:
                # 加载npy文件
                data = np.load(file_path)

                # 打印文件名、维度和内容
                print(f"File: {filename}")
                print(f"Shape: {data.shape}")  # 打印维度
                print(f"Contents: \n{data}")  # 打印内容

                print("=" * 50)  # 分隔线

            except Exception as e:
                print(f"Error loading {filename}: {e}")


def single_npy(file_path):
    np.set_printoptions(threshold=np.inf, edgeitems=1000, linewidth=150)
    data = np.load(file_path)
    print("维度（shape）", data.shape)
    # print(data[0, 100:200])
    print("mean", data[0, :])
    print("std", data[1, :])



def find_top_10_min_max(save_path):
    """
    从 (32, 2522) 形状的文件中筛选出最大和最小的 10 个值及其所在的位置
    返回最大和最小的 10 个值及其对应的频点和位置
    """
    # 1. 加载差值矩阵
    difference = np.load(save_path)  # shape: (32, 2522)

    # 2. 扁平化差值矩阵，找到最大和最小的 10 个值的索引
    flat_difference = difference.flatten()

    # 3. 获取最大的 10 个值及其索引
    max_indices = np.argsort(flat_difference)[-10:]  # 最大的 10 个索引
    min_indices = np.argsort(flat_difference)[:10]  # 最小的 10 个索引

    # 4. 将扁平化索引转换回二维索引
    max_positions = [(index // difference.shape[1], index % difference.shape[1]) for index in max_indices]
    min_positions = [(index // difference.shape[1], index % difference.shape[1]) for index in min_indices]

    # 5. 输出结果
    max_values = flat_difference[max_indices]
    min_values = flat_difference[min_indices]

    print("最大值及其位置：")
    for val, pos in zip(max_values, max_positions):
        print(f"值: {val}, 位置: {pos} (频点, 位置)")

    print("\n最小值及其位置：")
    for val, pos in zip(min_values, min_positions):
        print(f"值: {val}, 位置: {pos} (频点, 位置)")

    return max_values, max_positions, min_values, min_positions


if __name__ == "__main__":
    # directory_path =r'E:\grad_project\infer_chedar_0438'  # 替换为你要查看的目录路径
    # print_npy_info(directory_path)

    # chedar0_left_hrir_path=r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR_HRTF\chedar0_left_hrir.npy"
    # single_npy(chedar0_left_hrir_path)

    # output=r"E:\grad_project\SDE_chedar\sde_chedar_1001.npy"
    # single_npy(output)

    # fp = r"E:\grad_project\ground_truth\right_ear_chedar_0438_UV1m.npy"
    # fp=r"E:\grad_project\infer_CHEDAR_3\new_0001_32.npy"
    fp = r"E:\grad_project\lsd_per_freq.npy"
    single_npy(fp)
