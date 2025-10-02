import numpy as np
import matplotlib.pyplot as plt

ph = r"E:\grad_project\lsd_per_freq.npy"
LSD_data = np.load(ph)
LSD_values = LSD_data[0, :]

# 原始频率数据（单位：Hz）
frequencies_hz = np.array([1070, 1175, 1288, 1411,
                           1544, 1687, 1842, 2009,
                           2190, 2386, 2597, 2825,
                           3072, 3339, 3627, 3939,
                           4276, 4640, 5033, 5459,
                           5918, 6415, 6952, 7532,
                           8159, 8837, 9570, 10362,
                           11218, 12143, 13143, 14223])
frequencies_khz = frequencies_hz / 1000

# 使用等间距索引作为横轴坐标
x = np.arange(len(frequencies_khz))

plt.figure(figsize=(12, 6))
plt.plot(x, LSD_values, marker='o', linestyle='-', color='b', label='Mean LSD')

plt.ylim(0, 5)
plt.xticks(x, [f"{freq:.2f}" for freq in frequencies_khz])  # 标签为频率
plt.tick_params(axis='x', rotation=45)  # 保持水平

plt.title('CHEDAR Test Set Individual Average LSD')
plt.xlabel('Frequency (kHz)')
plt.ylabel('LSD Mean')
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\TOOLS\LSD-mean.png")
# plt.show()

