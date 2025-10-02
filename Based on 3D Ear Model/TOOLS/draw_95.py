import numpy as np
import matplotlib.pyplot as plt

# 加载数据
ph = r"E:\grad_project\lsd_per_freq.npy"
data = np.load(ph)
mean = data[0, :]
std = data[1, :]

# 定义频率
frequencies_hz = np.array([
    1070, 1175, 1288, 1411, 1544, 1687, 1842, 2009,
    2190, 2386, 2597, 2825, 3072, 3339, 3627, 3939,
    4276, 4640, 5033, 5459, 5918, 6415, 6952, 7532,
    8159, 8837, 9570, 10362, 11218, 12143, 13143, 14223
])
frequencies_khz = frequencies_hz / 1000
x = np.arange(len(frequencies_khz))

# 样本数量和置信区间计算
n = 250
z = 1.96
margin = z * std / np.sqrt(n)

# 配色
mean_color = '#1f77b4'       # 深蓝
ci_color = '#aec7e8'         # 浅蓝

# 绘图
plt.figure(figsize=(12, 6))

# 添加背景竖线
for xi in x:
    plt.axvline(x=xi, color='lightgray', linestyle='--', linewidth=0.5)

# 误差棒和连接线
plt.errorbar(
    x, mean, yerr=margin, fmt='*', color=mean_color,
    ecolor=ci_color, elinewidth=2, capsize=4, label='Mean LSD ± 95% CI'
)
plt.plot(x, mean, linestyle='-', color=mean_color, linewidth=1)

# 图形设置
plt.ylim(0, 3)
plt.xticks(x, [f"{f:.2f}" for f in frequencies_khz], rotation=45)
plt.xlabel("Frequency (kHz)")
plt.ylabel("LSD Value (dB)")
plt.title("Mean LSD with 95% Confidence Interval")
plt.legend()
plt.tight_layout()
plt.savefig(r"E:\BaiduNetdiskDownload\毕业论文代码归档-符悦微\TOOLS\95_with_vlines.png")
plt.show()
