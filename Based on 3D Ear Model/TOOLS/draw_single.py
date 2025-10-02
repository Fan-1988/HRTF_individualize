import os
import numpy as np
import matplotlib.pyplot as plt

def plot_log_hrtf_amplitude(directory, space_point_index, save_path):
    #遍历32个频点 绘制幅度谱（模型直接推理的结果 对数谱）
    hrtf_amplitude = []

    for i in range(32):  # index0_hrtf.npy 至 index31_hrtf.npy
        file_name = f"index{i}_hrtf.npy"
        file_path = os.path.join(directory, file_name)

        # 加载对数域下的 HRTF 幅度谱数据
        hrtf_data = np.load(file_path)

        # 直接提取原始值（对数域）
        amplitude_log = hrtf_data[0, space_point_index]
        hrtf_amplitude.append(amplitude_log)

    hrtf_amplitude = np.array(hrtf_amplitude)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(32), hrtf_amplitude, marker='o')
    plt.title(f"HRTF Log-Amplitude Spectrum at Space Point Index {space_point_index}")
    plt.xlabel("Frequency Points (Index)")
    plt.ylabel("Log-Amplitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_linear_hrtf_amplitude(directory, space_point_index, save_path):
    #遍历32个频点 绘制幅度谱（对数谱转为线性）
    hrtf_amplitude = []

    for i in range(32):  # index0_hrtf.npy 至 index31_hrtf.npy
        file_name = f"index{i}_hrtf.npy"
        file_path = os.path.join(directory, file_name)

        hrtf_data = np.load(file_path)

        # 提取对应空间点的线性幅度值（无需 abs）
        amplitude_linear = hrtf_data[0, space_point_index]
        hrtf_amplitude.append(amplitude_linear)

    hrtf_amplitude = np.array(hrtf_amplitude)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(32), hrtf_amplitude, marker='o')
    plt.title(f"HRTF Linear Amplitude Spectrum at Space Point Index {space_point_index}")
    plt.xlabel("Frequency Points (Index)")
    plt.ylabel("Linear Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_interp256(npy_file_path: str, position_idx: int, output_image_path: str):
    #     绘制插值后的256 线性HRTF幅度谱
    data = np.load(npy_file_path)  # 假设数据的维度是 (2522, 256)
    print(f"数据维度: {data.shape}")  #  (2522, 256)

    # 3. 选择某一个位置 idx，获取该位置的 HRTF 幅度谱
    hrtf_at_position = data[position_idx]  # 获取位置 idx 对应的 256 个频点的幅度值

    # 4. 绘制该位置的 HRTF 幅度谱
    plt.figure(figsize=(8, 6))
    plt.plot(hrtf_at_position)  # 绘制幅度谱，直接用 npy 中的值
    plt.title(f"HRTF幅度谱 at position {position_idx}")
    plt.xlabel("频点")
    plt.ylabel("幅度")
    plt.grid(True)

    plt.savefig(output_image_path)

    # plt.show()

    print(f"图像已保存至: {output_image_path}")


if __name__ == '__main__':
    # plot_log_hrtf_amplitude(
    #     directory=r"E:\grad_project\infer",
    #     space_point_index=100,
    #     save_path=r'E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\pic\log_32_loc_100.png'
    # )
    # plot_log_hrtf_amplitude(
    #     directory=r"E:\grad_project\infer",
    #     space_point_index=500,
    #     save_path=r'E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\pic\log_32_loc_500.png'
    # )

    plot_linear_hrtf_amplitude(
        directory=r"E:\grad_project\linear_infer_chedar",
        space_point_index=100,
        save_path=r'E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\cp_infer_CHEDAR\linear_32_loc_100.png'
    )
    plot_linear_hrtf_amplitude(
        directory=r"E:\grad_project\linear_infer_chedar",
        space_point_index=500,
        save_path=r'E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\cp_infer_CHEDAR\linear_32_loc_500.png'
    )

    # npy_file_path = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\Interp\infer_256interp_test_left.npy"  # 替换为你的文件路径
    # position_idx = 500
    # output_image_path = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\Interp\interp_test_left_loc500.png"  # 图像保存路径
    #
    # # 调用函数绘制并保存图像
    # plot_interp256(npy_file_path, position_idx, output_image_path)
