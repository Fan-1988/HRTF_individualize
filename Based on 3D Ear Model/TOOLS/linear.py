import os
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import BarycentricInterpolator


def log2linear(A_dir, B_dir):
    os.makedirs(B_dir, exist_ok=True)

    for i in range(32):
        filename = f'index{i}_hrtf.npy'
        filepath = os.path.join(A_dir, filename)

        # 加载对数幅度谱（单位 dB）
        log_magnitude = np.load(filepath)  # shape: (1, 2522)

        # 还原为线性幅度谱
        linear_magnitude = 10 ** (log_magnitude / 20)

        # 保存到B目录
        save_path = os.path.join(B_dir, filename)
        np.save(save_path, linear_magnitude)

        print(f'{filename} 已处理并保存到 {B_dir}')


# 32频点 线性HRTF幅度谱 插值到256
def interpolate_hrtf_to_256(input_folder, edge_freq_file, output_file, sampling_rate=48000):
    original_freqs = np.array([
        1070.44498013, 1175.44213048, 1288.92422067, 1411.57692843,
        1544.14134183, 1687.41843716, 1842.27391851, 2009.64344848,
        2190.53830168, 2386.05147494, 2597.36429144, 2825.75353842,
        3072.59918177, 3339.39270401, 3627.74611603, 3939.40169722,
        4276.24252252, 4640.30384032, 5033.78536977, 5459.06459183,
        5918.71111444, 6415.50219846, 6952.4395384, 7532.76739915,
        8159.99221833, 8837.90379282, 9570.59817731, 10362.50243335,
        11218.40137842, 12143.46649655, 13143.28718534, 14223.90452797
    ])

    # 映射到 [0, 255] 的频点位置
    freq_index_pos = (original_freqs / (sampling_rate / 2)) * 256  # shape: (32,)
    x_known = freq_index_pos  # 插值的x坐标 (32个频点)
    x_target = np.arange(256)  # 目标频率索引 (0-255)

    # 读取HRTF数据
    hrtf_list = []
    for i in range(32):
        file_path = os.path.join(input_folder, f"index{i}_hrtf.npy")
        data = np.load(file_path).reshape(2522)
        hrtf_list.append(data)

    print('32 index loaded')
    hrtf_array = np.array(hrtf_list).T  # (2522, 32)

    # 读取边界值
    edge_data = np.load(edge_freq_file)  # (2522, 256)

    # 初始化插值结果
    interpolated_hrtf = np.zeros((2522, 256))

    # 先填入边界列
    interpolated_hrtf[:, 0] = edge_data[:, 0]
    interpolated_hrtf[:, 255] = edge_data[:, 255]

    # 中间部分插值 (不外推)
    for i in range(2522):
        y = hrtf_array[i]  # 原始32个频点数据
        f = interp1d(x_known, y, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_hrtf[i, 1:255] = f(x_target[1:255])  # 只插值中间部分

    # 保存
    np.save(output_file, interpolated_hrtf)
    print(f"HRTF插值完成，保存至：{output_file}")


def hrtf_frequency_interpolation(input_folder, output_path, boundary_path):
    freqs = np.array([
        1070.44498013, 1175.44213048, 1288.92422067, 1411.57692843,
        1544.14134183, 1687.41843716, 1842.27391851, 2009.64344848,
        2190.53830168, 2386.05147494, 2597.36429144, 2825.75353842,
        3072.59918177, 3339.39270401, 3627.74611603, 3939.40169722,
        4276.24252252, 4640.30384032, 5033.78536977, 5459.06459183,
        5918.71111444, 6415.50219846, 6952.4395384, 7532.76739915,
        8159.99221833, 8837.90379282, 9570.59817731, 10362.50243335,
        11218.40137842, 12143.46649655, 13143.28718534, 14223.90452797
    ])

    # 2. 计算对应的256点DFT频点位置
    sample_rate = 48000
    nyquist = sample_rate / 2
    dft_bins = (freqs / nyquist) * 256
    dft_bins = np.round(dft_bins).astype(int)  # 四舍五入到最近的整数频点

    # 3. 加载边界频点数据 (从2522,256中提取0和255频点)
    full_spectrum = np.load(boundary_path)  # (2522,256)
    low_bound = full_spectrum[:, 0]  # 0频点 (2522,)
    high_bound = full_spectrum[:, 255]  # 255频点 (2522,)

    # 4. 加载所有32个频点数据并组合
    hrtf_data = []
    for i in range(32):
        file_path = f"{input_folder}/index{i}_hrtf.npy"
        data = np.load(file_path)  # (1,2522)
        hrtf_data.append(data[0])  # 取出(2522,)的数据

    hrtf_data = np.array(hrtf_data)  # (32,2522)

    # 5. 准备插值用的x轴位置 (34个点: 0 + 32 + 255)
    full_bins = np.concatenate([
        [0],  # 0频点
        dft_bins,  # 32个频点
        [255]  # 255频点
    ])

    # 6. 对每个声源位置(共2522个)进行插值处理
    interpolated_results = []
    for pos in range(2522):
        # 获取当前位置的所有频点数据
        known_values = hrtf_data[:, pos]  # (32,)

        # 组合完整数据 (0 + 32 + 255)
        full_values = np.concatenate([
            [low_bound[pos]],  # 0频点
            known_values,  # 32个频点
            [high_bound[pos]]  # 255频点
        ])

        # 创建插值函数 (二次插值)
        # 注意: 即使设置fill_value='extrapolate'，由于我们的x_new在x范围内，不会实际外推
        f = interpolate.interp1d(
            full_bins,
            full_values,
            kind='quadratic',
            fill_value='extrapolate',  # 理论上不会用到
            assume_sorted=True
        )

        # 生成所有256个频点的值
        all_bins = np.arange(256)
        interpolated = f(all_bins)

        # 确保边界值准确使用预存值
        interpolated[0] = low_bound[pos]  # 强制使用预存的0频点
        interpolated[255] = high_bound[pos]  # 强制使用预存的255频点

        interpolated_results.append(interpolated)

    # 7. 转换为(2522,256)并保存
    result = np.array(interpolated_results)
    np.save(output_path, result)

    print(f"插值完成，结果已保存到 {output_path}")


if __name__ == '__main__':
    # A_dir = r'E:\grad_project\infer_right'
    # B_dir = r'E:\grad_project\linear_infer_right'
    # log2linear(A_dir,B_dir)

    # A_dir = r'E:\grad_project\infer_chedar'
    # B_dir = r'E:\grad_project\linear_infer_chedar'
    # log2linear(A_dir,B_dir)

    A_dir = r'E:\grad_project\cp_infer_chedar'
    B_dir = r'E:\grad_project\cp_linear_infer_chedar'
    log2linear(A_dir,B_dir)

    # input_f = r"E:\grad_project\linear_infer"
    # edge_freq_f = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR_HRTF\chedar0_left_mag.npy"
    # output_f = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\Interp\infer_256interp_test_left.npy"
    # # interpolate_hrtf_to_256(input_f, edge_freq_f, output_f)
    # hrtf_frequency_interpolation(input_f, output_f, edge_freq_f)
