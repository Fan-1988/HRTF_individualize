import numpy as np
import pysofaconventions as sofa
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import glob
from pysofaconventions import SOFAFile
from scipy.fft import fft


def paintline(line,title="line tile"):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(line, label=title)  # 画平均HRIR，举例
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def read_sofa_hrir(sofa_path):
    file = sofa.SOFAFile(sofa_path, 'r')
    # 读取原始hrtf文件
    if file.isValid():
        target_loc_data = file.getVariableValue('SourcePosition')
        target_loc_data = target_loc_data[:, :2]
        target_loc_data = np.array(target_loc_data)
        hrir_data = file.getDataIR()
        # print(hrtf_data.shape)
        file.close()
        return target_loc_data, hrir_data

    else:
        print("not valid file")
        target_loc_data = file.getVariableValue('ListenerPosition')
        target_loc_data = target_loc_data[:, :2]
        target_loc_data = np.array(target_loc_data)
        hrir_data = file.getDataIR()
        file.close()
        return target_loc_data, hrir_data


def mel_fliter(file_path):
    file = sofa.SOFAFile(file_path, 'r')
    attributes = file.getGlobalAttributesAsDict()
    # print(attributes)
    sample_rate = int(file.getSamplingRate())
    # print(sample_rate)
    loc, hrir = read_sofa_hrir(file_path)

    sr = sample_rate
    n_fft = hrir.shape[2]
    n_mels = 64
    hop_length = n_fft * 2

    hrtf = np.fft.fft(hrir[0,0,:])
    hrtf = 20*np.log10(abs(hrtf))

    # paintline(hrtf)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=sr / 2,
        pad=0,
        n_mels=n_mels,
    )

    # print(hrir.shape)
    hrir = np.array(hrir)
    # 2522*2*480
    hrir1 = hrir[:, 0, :]
    hrir2 = hrir[:, 1, :]

    hrir1 = torch.tensor(hrir1, dtype=torch.float32)
    hrir2 = torch.tensor(hrir2, dtype=torch.float32)
    hrir1 = mel_spectrogram(hrir1)
    hrir2 = mel_spectrogram(hrir2)

    output = np.stack([hrir1, hrir2], axis=1)

    output = np.squeeze(output)
    # print(output.shape)
    print("np min", np.min(output[:,:,20:56]))
    # 替换最小值
    non_zero_elements = output[output > 0]
    min_non_zero = np.min(non_zero_elements)
    output[output == 0] = min_non_zero
    output = 20*np.log10(output)
    # paint_target = output[0,0,:]
    # print("mel result",paint_target[0])
    # paint_target = 20*np.log10(abs(paint_target))
    # paintline(output[0,0])
    return output

def process_sofa_file_in_mel(sofa_file):
    output = mel_fliter(sofa_file)
    base_name = os.path.basename(sofa_file)
    save_name = base_name.replace('.sofa', '.npy')
    save_path = os.path.join(target_folder, save_name)
    np.save(save_path, output)

if __name__ == "__main__":
    # audio=np.random.randn(2,480)
    # print(dogfcc(audio,48000).shape)
    # mel_fliter("D:\hrtf\SONICOM\hrtf\P0021_FreeFieldComp_48kHz.sofa")
    sofa_folder = "/data/fyw/dataset/dataset/chedar/hrtfs/"
    target_folder = "/data/fyw/dataset/dataset/chedar/hrtfs_mel64_log/"
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # 使用glob找到所有满足模式的SOFA文件
    sofa_files = glob.glob(os.path.join(sofa_folder, "chedar_*_UV1m.sofa"))
    for sofa_file in sofa_files:
        process_sofa_file_in_mel(sofa_file)
