import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import newShDataset
import torch.nn as nn
import time
import resnet
import resnet_combine2
from multiprocessing import Process
import multiprocessing as mp



def evaluate_test_set(model, freq_index, test_loader, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_count = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for voxel, hrtf, sh in test_loader:
            voxel = voxel.unsqueeze(1).float().to(device)
            # print(hrtf.shape)
            hrtf = hrtf[:, :, 1, :].squeeze().float().to(device)
            hrtf = hrtf[:, :, freq_index].squeeze().float().to(device)
            # print("hrtf shape", hrtf.shape)
            sh = sh.reshape(sh.shape[0], 2048).float().to(device)
            # print("mofel hrtf shape", hrtf.shape)
            # print("model input shape")
            # print(np.array(voxel.cpu()).shape)
            # 30,1,32,32,32
            output1, out2 = model(voxel)
            loss1 = criterion(output1, hrtf)
            loss2 = criterion(out2, sh)
            loss = loss1 * 0.5 + loss2 * 0.5
            total_loss += loss.item()

            # 计算每批次的MSE并累加
            mse = torch.mean((output1 - hrtf) ** 2).item()
            total_mse += mse * voxel.size(0)
            total_count += voxel.size(0)
    average_loss = total_loss / len(test_loader)
    mse_error = total_mse / total_count  # 计算整个测试集的平均MSE
    model.train()
    return average_loss, mse_error


def print_model_shape(model, input_size):
    model.eval()  # ????????????????
    with torch.no_grad():  # ?????????
        x = torch.randn(input_size)  # ?????????????????????????
        for name, layer in model.named_children():
            x = layer(x)
            print(f"{name}: {x.shape}")  # ??????????????????


def train_model(freq_index, gpu_id, only_split_and_print=False):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # OUTPUT_DIR_BASE = "/data1/fsj/precombine/chedar_newprint"
    batchsize = 8

    if only_split_and_print:
        # Define dataset config
        dataset_config = {
            "chedar": {
                "ply_path": "/data1/fsj/combine/CHEDAR/voxel/",
                "sofa_path": "/data1/fsj/combine/CHEDAR/hrtfs_erb64_log/",
                "sh_path": "/data1/fsj/combine/CHEDAR/hrtfs_erb64_log_sh/",
                "ply_pattern": "chedar_*_centered.npy",
                "sofa_pattern": "chedar_*_UV1m.npy",
                "sh_pattern": "chedar_*_UV1m.npy",
                "ply_file_name_format": "chedar_{}_centered.npy",
                "sofa_file_name_format": "chedar_{}_UV1m.npy",
                "sh_file_name_format": "chedar_{}_UV1m.npy",
                "hrtf_index": freq_index
            }
        }

        print("main func in dataset.py")
        dataset_names = ["chedar"]
        datasets = []

        for name in dataset_names:
            config = dataset_config[name]
            dataset = newShDataset(**config)
            datasets.append(dataset)

        combined_dataset = ConcatDataset(datasets)

        total_size = len(combined_dataset)
        train_size = int(total_size * 0.8)
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(
            combined_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(0)
        )

        # Print filenames of test dataset
        dataset_obj = datasets[0]
        print("\n=== Test Dataset 文件名列表 ===")
        for idx_in_subset, original_idx in enumerate(test_dataset.indices):
            filename = dataset_obj.ply_file_names[original_idx]
            print(f"Test Sample {idx_in_subset}: {filename}")
        print("=== 打印结束 ===\n")

        # Exit early without training
        sys.exit(0)

    # If not only split and print, proceed with training logic (model initialization, optimizer, scheduler, etc.)
    print(f"[GPU {gpu_id}] Starting training for freq_index {freq_index} on {device}")


# ----------------- run_train_model_on_gpu function -----------------
def run_train_model_on_gpu(range_start, range_end, gpu_id):
    torch.cuda.set_device(gpu_id)

    for i in range(range_start, range_end):
        try:
            train_model(i, gpu_id, only_split_and_print=True)  # Pass `only_split_and_print=True`
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory on GPU {gpu_id} while training frequency index {i}. Stopping training.")
            raise  # Re-raise the exception so the main process can catch it

# ----------------- Main Execution -----------------
if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []
    manager = mp.Manager()
    failed_indices = manager.list()

    gpu_ranges = [
        [0, 7, 0],
        [8, 15, 1],
        [16, 23, 2],
        [24, 31, 2]
    ]

    try:
        for range_start, range_end, gpu_id in gpu_ranges:
            p = Process(target=run_train_model_on_gpu, args=(range_start, range_end, gpu_id))
            processes.append(p)
            print(f"Starting process for frequencies {range_start} to {range_end}")
            p.start()

        for p in processes:
            p.join()

    except torch.cuda.OutOfMemoryError:
        print("Terminating all processes due to out of memory error.")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

        print("Training terminated due to out of memory error.")