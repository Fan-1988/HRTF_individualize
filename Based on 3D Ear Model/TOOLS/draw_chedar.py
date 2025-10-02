import numpy as np
import matplotlib.pyplot as plt

# fp = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR_HRTF\chedar0_left_mag.npy"
# magnitude = np.load(fp)  # shape = (2522, 256)
#
# # 选择某一个位置，例如 index = 123
# index = 100
# spectrum = magnitude[index]  # shape = (256,)
#
# # 可视化（线性）
# plt.plot(20 * np.log10(np.abs(magnitude[100, :256]) + 1e-12))
# plt.title("Magnitude Spectrum (dB) at Position #100")
# plt.xlabel("Frequency Bin Index")
# plt.ylabel("Magnitude (dB)")
# # plt.plot(spectrum)
# # plt.xlabel('Frequency Bin Index (0~255)')
# # plt.ylabel('Magnitude (linear)')
# plt.title(f'Magnitude Spectrum at Position #{index}')
# plt.grid(True)
# # save_path=r'E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\pic\chedar0_loc_100.png'
# # plt.savefig(save_path, dpi=300)
# plt.show()



# npy_f = r"E:\grad_project\chedar_0001_UV1m.npy"
# data = np.load(npy_f)
#
# # 指定要查看的 '位置' 和 '耳朵' (例如位置为1000，耳朵为0)
# position = 100  # 你想要绘制的具体位置
# ear = 0  # 0表示左耳，1表示右耳
#
# # 提取指定位置和耳朵的数据
# magnitude_spectrum = data[position, ear, :]
#
# # 绘制幅度谱
# plt.plot(magnitude_spectrum)
# plt.title(f'Magnitude Spectrum at Position {position}, Ear {ear}')
# plt.xlabel('Frequency Bin')
# plt.ylabel('Magnitude')
# plt.grid(True)
#
# output_image_path = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR-pic\erb_locs_100.png"
# plt.savefig(output_image_path)



npy_f = r"E:\grad_project\chedar_0001_UV1m.npy"
data = np.load(npy_f)

# 指定要查看的 '位置' 和 '耳朵' (例如位置为100，耳朵为0)
position = 100  # 你想要绘制的具体位置
ear = 0  # 0表示左耳，1表示右耳

# 提取指定位置和耳朵的数据
log_spectrum = data[position, ear, :]

# 将对数谱转换为线性谱
linear_spectrum = 10 ** (log_spectrum / 20)

# 绘制线性谱
plt.plot(linear_spectrum)
plt.title(f'Linear Spectrum at Position {position}, Ear {ear}')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)

# 保存图像
output_image_path = r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\CHEDAR-pic\linear_erb_locs_100_linear.png"
plt.savefig(output_image_path)