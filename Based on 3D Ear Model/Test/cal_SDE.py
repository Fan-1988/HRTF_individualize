import os
import numpy as np

# 设置A、B、C目录路径
# A_dir = '/data1/fsj/combine/CHEDAR/all_test_gtruth_left'  # 真实幅度谱目录
# B_dir = '/data1/fsj/precombine/chedartrain/all_test_infer_32_250'  # 预测幅度谱目录
# C_dir = '/data1/fsj/combine/CHEDAR/sub'  # 结果保存目录
#
# # 遍历A目录和B目录
# for file_A in os.listdir(A_dir):
#     if file_A.endswith('.npy'):
#         # 从文件名中提取序号（如0394）
#         file_id = file_A.split('_')[1].split('.')[0]
#
#         # 对应的B目录中的预测文件名
#         file_B = f'chedar_{file_id}_infer_32.npy'
#
#         # 检查B目录中是否有对应的文件
#         if file_B in os.listdir(B_dir):
#             # 加载真实和预测的幅度谱
#             real_spectrum = np.load(os.path.join(A_dir, file_A))  # 真实幅度谱 (32, 2522)
#             predicted_spectrum = np.load(os.path.join(B_dir, file_B))  # 预测幅度谱 (32, 2522)
#
#             # 计算真实和预测的幅度谱差异
#             difference_spectrum = real_spectrum - predicted_spectrum
#
#             # 保存结果到C目录
#             result_filename = f'sub_{file_id}.npy'
#             np.save(os.path.join(C_dir, result_filename), difference_spectrum)
#             print(f'Saved difference for {file_id} to {os.path.join(C_dir, result_filename)}')

import numpy as np
import glob
import os


def calculate_lsd_statistics(folder_path):
    """
    计算 LSD 误差的统计量（均值、最小值、最大值、标准差）
    参数:
        folder_path (str): 包含误差文件的文件夹路径
    返回:
        stats (numpy.ndarray): 统计量数组，shape 为 (4, 32)
    """
    # 获取所有 .npy 文件路径
    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.npy')))

    # 确保文件数量正确
    assert len(file_paths) == 250, f"应有250个样本，当前读取到 {len(file_paths)} 个"

    # 加载所有误差文件，shape 应为 (32, 2522)
    all_errors = []

    for path in file_paths:
        err = np.load(path)  # shape: (32, 2522)
        assert err.shape == (32, 2522), f"文件 {path} 维度错误: {err.shape}"
        all_errors.append(err)

    # 堆叠成 (250, 32, 2522)
    all_errors = np.stack(all_errors, axis=0)

    # 先取绝对值，确保误差为 LSD
    all_errors = np.abs(all_errors)

    # 计算每个频率点的统计量（对所有人和空间位置）
    mean_per_freq = all_errors.mean(axis=(0, 2))  # shape: (32,)
    std_per_freq = all_errors.std(axis=(0, 2))  # shape: (32,)

    stats = np.stack([mean_per_freq, std_per_freq], axis=0)

    # 保存为 .npy 文件
    output_file = os.path.join(folder_path, 'lsd_per_freq.npy')
    np.save(output_file, stats)

    print(f"✅ 结果已保存为 {output_file}")
    print("数组 shape: (4, 32)")
    print("含义：")
    print("  stats[0, :] = 每个频率点的均值")
    print("  stats[1, :] = 每个频率点的标准差")

    return output_file


if __name__ == '__main__':
    folder_path = '/data1/fsj/combine/CHEDAR/sub'
    calculate_lsd_statistics(folder_path)
