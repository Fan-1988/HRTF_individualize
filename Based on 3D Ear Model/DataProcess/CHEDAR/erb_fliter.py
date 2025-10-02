import numpy as np
import pysofaconventions as sofa
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import glob
from pysofaconventions import SOFAFile
from scipy.fft import fft
from pyfilterbank import gammatone


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
        
def apply_gammatone_filterbank(hrir_data, sample_rate):
    start_band = -5
    end_band = 18
    norm_freq = 2000
    density = 0.72
    a = gammatone.frequencies_gammatone_bank(start_band, end_band, norm_freq, density)
    a = np.array(a)
    # print(a, a.shape)
    freq_indices = np.round(512*(a/(sample_rate/2))).astype(int)  # Convert freqs to indices
    # print(freq_indices)
    gfb = gammatone.GammatoneFilterbank(samplerate=sample_rate, order=4, startband=start_band, endband=end_band,
                                        normfreq=norm_freq, density=density, bandwidth_factor=1.0,
                                        desired_delay_sec=0.02)
    filtered_output = np.zeros((hrir_data.shape[0],hrir_data.shape[1],int((end_band-start_band)/density)+1), dtype=np.float64)  # Ensure it's a floating-point array
    for i in range(hrir_data.shape[0]):
        for channel in range(hrir_data.shape[1]):
            hrir = np.pad(hrir_data[i, channel], (0, 1024 - hrir_data.shape[2]), 'constant')
            hrir = np.abs(np.fft.fft(hrir))
            filtered_signal = np.zeros(512)  # Or use the actual expected length of the filtered output
            for band, state in gfb.analyze(hrir):
                filtered_signal += hrir[:512]*np.abs(band[:512])  # Summing up signals from all bands
            # print(filtered_signal.shape)
            filtered_signal = 20*np.log10(filtered_signal)
            filtered_signal = filtered_signal[freq_indices]
            filtered_output[i, channel] = filtered_signal
    return filtered_output

def erb_fliter(file_path):
    _, hrir_data = read_sofa_hrir(file_path)
    file = sofa.SOFAFile(file_path, 'r')
    attributes = file.getGlobalAttributesAsDict()
    # print(attributes)
    sample_rate = int(file.getSamplingRate())
    file.close()

    filtered_hrir = apply_gammatone_filterbank(hrir_data, sample_rate)
    # paintline(filtered_hrir[0][0])  # Assuming this is the correct indexing
    print(filtered_hrir.shape)
    return(filtered_hrir)
        
def process_sofa_file_in_erb(sofa_file,target_folder):
    output = erb_fliter(sofa_file)
    base_name = os.path.basename(sofa_file)
    save_name = base_name.replace('.sofa', '.npy')
    save_path = os.path.join(target_folder, save_name)
    np.save(save_path, output)

from multiprocessing import Process
import multiprocessing as mp

if __name__ == "__main__":
    sofa_folder = "/data/fyw/dataset/dataset/chedar/hrtfs/"
    target_folder = "/data/fyw/dataset/dataset/chedar/hrtfs_erb64_log/"
    # sofa_folder = "/data/fyw/dataset/dataset/SONICOM/hrtf/"
    # target_folder = "/data/fyw/dataset/dataset/SONICOM/hrtf_erb64_log/"
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    mp.set_start_method('spawn')
    # 使用glob找到所有满足模式的SOFA文件
    sofa_files = glob.glob(os.path.join(sofa_folder, "P*_FreeFieldComp_48kHz.sofa"))
    with mp.Pool(processes=4) as pool:
        # Create a list of tasks for the pool
        pool.starmap(process_sofa_file_in_erb, [(sofa_file, target_folder) for sofa_file in sofa_files])

