import numpy as np
import pysofaconventions as sofa
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import glob
from pysofaconventions import SOFAFile
from scipy.fft import fft
from scipy.special import sph_harm

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
    
def hrtf_to_spherical_harmonics(hrtf_data, directions, max_order):
    """
    Convert HRTF data to spherical harmonics coefficients.

    :param hrtf_data: An array of HRTF measurements.
    :param directions: A list of tuples, each representing the (azimuth, elevation) of the HRTF measurement.
    :param max_order: The maximum order of spherical harmonics to compute.
    :return: An array of spherical harmonics coefficients.
    """
    # Number of measurements
    num_measurements = hrtf_data.shape[0]

    # Initialize matrix for spherical harmonics
    Y_matrix = np.zeros((num_measurements, (max_order + 1) ** 2), dtype=complex)

    for i, (azimuth, elevation) in enumerate(directions):
        for order in range(max_order + 1):
            for degree in range(-order, order + 1):
                Y_matrix[i, order ** 2 + order + degree] = sph_harm(degree, order, azimuth, elevation)

    # Compute spherical harmonics coefficients
    coeffs = np.linalg.lstsq(Y_matrix, hrtf_data, rcond=None)[0]

    return coeffs
    
def load_npy_data(filepath):
    # Load data from .npy file
    npy_data = np.load(filepath)
    # Convert NumPy array to PyTorch tensor
    tensor_data = torch.tensor(npy_data, dtype=torch.float)
    return tensor_data    

def calculate_sh_coefficients(sofa_file_path, max_order):
    # 从SOFA文件中读取HRIR数据
    directions, _ = read_sofa_hrir("/data/fyw/dataset/dataset/SONICOM/hrtf/P0199_FreeFieldComp_48kHz.sofa")
    hrtf_data = load_npy_data(sofa_file_path)
    print(hrtf_data.shape)
    hrtf_data = hrtf_data[:, 0, :]  # 假设只处理一个耳朵
    directions = directions[:, :2]

    # 计算球谐系数
    sh_coeffs = hrtf_to_spherical_harmonics(hrtf_data, directions, max_order)
    print(sh_coeffs.shape)
    return sh_coeffs

def process_sofa_file_in_mel(mel_npy_file,target_folder):
    output = calculate_sh_coefficients(mel_npy_file,7)
    base_name = os.path.basename(sofa_file)
    save_path = os.path.join(target_folder, base_name)
    np.save(save_path, output)



if __name__ == "__main__":
    # audio=np.random.randn(2,480)
    # print(dogfcc(audio,48000).shape)
    # mel_fliter("D:\hrtf\SONICOM\hrtf\P0021_FreeFieldComp_48kHz.sofa")
    # mel_npy_folder = "/data/fyw/dataset/dataset/chedar/hrtfs_erb64_log/"
    # target_folder = "/data/fyw/dataset/dataset/chedar/hrtfs_erb64_log_sh/"
    mel_npy_folder = "/data/fyw/dataset/dataset/SONICOM/hrtf_erb64_log/"
    target_folder = "/data/fyw/dataset/dataset/SONICOM/hrtf_erb64_log_sh/"
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # 使用glob找到所有满足模式的SOFA文件
    sofa_files = glob.glob(os.path.join(mel_npy_folder, "P*_FreeFieldComp_48kHz.npy"))
    for sofa_file in sofa_files:
        process_sofa_file_in_mel(sofa_file,target_folder)
