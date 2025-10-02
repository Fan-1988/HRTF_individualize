# 我们方法实现代码
# 基本配置：
#   dataset_config：指定数据集路径和文件名匹配方法
#   gpu_ranges：指定训练的freq_index区间 和该区间训练的gpu 例如 如果我只有一块8Gb显存的gpu 我可以指定 两个gpu_ranges： [0,16,0]和 [16,32,0] 即代表在 0号gpu上同时训练两个线程
#       第一个线程依次训练 freq_index 0-15 第二个线程依次训练16-31 每个模型约占4Gb显存
# 其它配置：
#   需要注意修改两个任务目标损失值比率的代码在 weight1 和 weight2中定义； 这里的默认配置是正确给出的， 但是代码结构中对 损失比的调整写得有点弯弯绕绕， 如果要修改的话需要仔细阅读代码逻辑
#   诸如学习率配置等、命名很标准在 train_model 方法中初始那一块的定义中

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import ShDataset
import torch.nn as nn
import time
import resnet
import resnet_combine2


def subsample_points(feature, coord, num_points):
    current_num_points = coord.size(2)
    if current_num_points > num_points:
        indices = torch.randperm(current_num_points)[:num_points]
        return feature[:, indices, :], coord[:, :, indices]
    return feature, coord


def pad_points(feature, coord, num_points):
    current_num_points = coord.size(2)
    if current_num_points < num_points:
        padding_size = num_points - current_num_points
        padding_feature = torch.zeros(1, padding_size, 1, device=feature.device)
        padding_coord = torch.zeros(1, 3, padding_size, device=coord.device)
        return torch.cat([feature, padding_feature], dim=1), torch.cat([coord, padding_coord], dim=2)
    return feature, coord


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


def train_model(freq_index, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    OUTPUT_DIR_BASE = "/data1/fsj/precombine/chedaroutput"
    # OUTPUT_DIR_BASE = "/data1/fsj/precombine/sonioutput"
    batchsize = 8

    model_depth = 34
    n_input_channels = 1
    n_classes1 = 2522
    n_classes2 = 2048
    pretrain_path = r'/data1/fsj/myHRTF/Pretrain_get_param/param_result/feature_extraction_best_params.pth'
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"The pretrain path {pretrain_path} does not exist.")

    model = resnet_combine2.generate_model(model_depth=model_depth,
                                           pretrain_path=pretrain_path,
                                           n_input_channels=n_input_channels,
                                           conv1_t_size=3,
                                           conv1_t_stride=1,
                                           no_max_pool=True,
                                           n_classes=n_classes1,
                                           n_classes2=n_classes2
                                           )

    # ????????????????[batch_size, channels, depth, height, width]
    # input_size = (1, n_input_channels, 32, 32, 32)  # ???????

    # ?????????????????
    # print_model_shape(model, input_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.2)  # Start with a learning rate of 0.1
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)  # Decay the learning rate
    log_interval = 10
    log_file_name = "training_log.txt"
    print(" torch.cuda.is_available()")
    print(torch.cuda.is_available())
    print(device)
    print(f"[GPU {gpu_id}] Training model for freq_index {freq_index} on {device}")
    model.to(device)
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
        },
        # ?????????????...
        # "hutubs": {
        #     "ply_path": r"C:\Users\freeway\Desktop\HRTF_dataset\HUBTUB\3D ear meshes",
        #     "sofa_path": r"D:\hrtf\chader\\",
        #     "ply_pattern": "pp*_3DheadMesh.ply",
        #     "sofa_pattern": "pp*_HRIRs_measured.sofa",
        #     "ply_file_name_format": "pp{}__3DheadMesh.ply",
        #     "sofa_file_name_format": "pp{}_HRIRs_measured.sofa",
        #     "hrtf_index": freq_index
        # },
        # "sonicom": {
        #     "ply_path": "/data1/fsj/combine/SONICOM/npy/",
        #     "sofa_path": "/data1/fsj/combine/SONICOM/hrtf_erb64_log_selected/",
        #     "sh_path": "/data1/fsj/combine/SONICOM/hrtf_erb64_log_sh/",
        #     "ply_pattern": "adjusted_P*_watertight_centered.npy",
        #     "sofa_pattern": "P*_FreeFieldComp_48kHz.npy",
        #     "sh_pattern": "P*_FreeFieldComp_48kHz.npy",
        #     "ply_file_name_format": "adjusted_P{}_watertight_centered.npy",
        #     "sofa_file_name_format": "P{}_FreeFieldComp_48kHz.npy",
        #     "sh_file_name_format": "P{}_FreeFieldComp_48kHz.npy",
        #     "hrtf_index": freq_index
        # }
    }
    print("main func in dataset.py")
    dataset_names = ["sonicom"]
    datasets = []
    print(len(datasets))
    # ???????????????? CustomDataset ???
    for name in dataset_names:
        config = dataset_config[name]
        dataset = ShDataset(**config)
        datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)

    total_size = len(combined_dataset)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(0))

    # ?????????????????DataLoader
    if len(train_dataset) == 0:
        print("Dataset is empty. Check dataset configuration and data loading.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=2)

    # ?????????????
    # data_loader = DataLoader(combined_dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    # ?????????????

    # ?????????????
    print("init success")
    epoch = 0
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    num_epochs = 400  # 增加到400个epochs
    weight_adjust_start_epoch = 200  # 从第200个epoch开始调整权重
    weight_adjust_start2_epoch = 200  # 从第200个epoch开始调整权重
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        test_loss = 'N/A'

        # ???????????????????????????????

        # ??????????????
        output_dir = f"{OUTPUT_DIR_BASE}/index{freq_index}"
        log_file = f"{output_dir}/{log_file_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        weight1 = 0.8
        weight2 = 0.2
        # ?????????
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        epoch_start_time = time.time()  # ??????
        if epoch <= weight_adjust_start_epoch:
            weight1 = 0.8
            weight2 = 0.2
        elif weight_adjust_start_epoch < epoch:
            weight1 = 1.0
            weight2 = 0.0
        if epoch == weight_adjust_start_epoch:
            current_lr = scheduler.get_last_lr()[0]
            optimizer = torch.optim.Adam(
                [param for name, param in model.named_parameters() if 'fc2' not in name],
                lr=current_lr
            )
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.01)  # Decay the learning rate
            model.fc2.weight.requires_grad = False
            model.fc2.bias.requires_grad = False
        if epoch == weight_adjust_start2_epoch:
            current_lr = scheduler.get_last_lr()[0]
            optimizer = torch.optim.Adam(
                [param for name, param in model.named_parameters() if 'fc2' not in name],
                lr=current_lr
            )
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # Decay the learning rate
            model.fc2.weight.requires_grad = False
            model.fc2.bias.requires_grad = False

        for batch_idx, (voxel, hrtf, sh) in enumerate(train_loader):
            # ?????????��??
            voxel = voxel.unsqueeze(1).float().to(device)
            # print(hrtf.shape)
            hrtf = hrtf[:, :, 1, :].squeeze().float().to(device)
            hrtf = hrtf[:, :, freq_index].squeeze().float().to(device)
            # print("hrtf shape", hrtf.shape)
            sh = sh.reshape(sh.shape[0], 2048).float().to(device)
            # print("mofel hrtf shape", hrtf.shape)
            # print("model sh shape", sh.shape)
            optimizer.zero_grad()
            # print("model input shape")
            # print(np.array(voxel.cpu()).shape)
            # 30,1,32,32,32
            output1, out2 = model(voxel)
            optimizer.zero_grad()
            loss1 = criterion(output1, hrtf)
            loss2 = criterion(out2, sh)
            loss = loss1 * weight1 + loss2 * weight2
            print(loss1, " ", loss2)
            loss.backward()
            optimizer.step()
            train_loss += loss1.item() * voxel.size(0)  # ???????????????????��?��

        scheduler.step()
        train_loss /= len(train_loader.dataset)  # ?????????????

        # ???????????
        if epoch % 5 == 0:

            model_save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_model_path = os.path.join(output_dir, "best_train_model.pth")
                torch.save(model.state_dict(), best_train_model_path)
                print(f"Best training model saved to {best_train_model_path}")

            if epoch % log_interval == 0:
                test_loss, test_mse = evaluate_test_set(model, freq_index, test_loader, device)
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_model_path = os.path.join(output_dir, "best_test_model.pth")
                    torch.save(model.state_dict(), best_test_model_path)
                    print(f"Best testing model saved to {best_test_model_path}")

            # ??????
            current_lr = scheduler.get_last_lr()[0]  # This fetches the last learning rate used by the scheduler

            # Writing to the log file
            with open(log_file, "a") as file:
                if test_loss != 'N/A':
                    file.write(
                        f"Epoch: {epoch}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, LR: {current_lr}\n")
                else:
                    file.write(
                        f"Epoch: {freq_index}_{epoch}, Training Loss: {train_loss:.4f}, LR: {current_lr},weight1 {weight1:.4f}\n")

            print(
                f"Epoch: {epoch}, Training Loss: {float(train_loss):.4f}, LR: {current_lr}, Test Loss: {test_loss if test_loss != 'N/A' else 'N/A'}")
        if epoch % 100 == 0 and epoch != 0:
            if (train_loss > 10):
                optimizer = optim.Adam(model.parameters(), lr=0.2 * epoch / 100)  # Start with a learning rate of 0.1
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)  # Decay the learning rate

        # ??????????????
        epoch_time = time.time() - epoch_start_time  # Epoch????
        print(f"Epoch {epoch}: Total epoch time: {epoch_time:.4f} seconds.")
        torch.cuda.empty_cache()


from multiprocessing import Process
import multiprocessing as mp


# def run_train_model_on_gpu(range_start, range_end, gpu_id):
#     for i in range(range_start, range_end):
#
#         train_model(i, gpu_id)
def run_train_model_on_gpu(range_start, range_end, gpu_id):
    torch.cuda.set_device(gpu_id)

    for i in range(range_start, range_end):
        try:
            train_model(i, gpu_id)
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory on GPU {gpu_id} while training frequency index {i}. Stopping training.")
            raise  # 重新抛出异常，主进程可以捕获到这个异常并终止所有进程


if __name__ == '__main__':
    # input_shape = (1, 32, 32, 32)  # 1?????32x32x32?????
    #
    # a = torch.randn(1, 32, 16, 16, 16)
    #
    # # ????????????????��?64
    # b = a.repeat(1, 64 // 32, 1, 1, 1)
    #
    # print("?????????:", a.shape)
    # print("?????????????:", b.shape)
    #
    # a = torch.randn(1, 2,2 )
    #
    # # ????????????????��?64
    # b = a.repeat_interleave(2, dim=1)
    #
    # print("?????????:", a)c
    # print("?????????????:", b)

    # test_block_output_shape(input_shape)
    # test_CNN_output_shape(input_shape)
    # ????????��?
    mp.set_start_method('spawn')
    processes = []
    manager = mp.Manager()
    failed_indices = manager.list()

    gpu_ranges = [
        # (0, 2, 0),  # 第1份，使用GPU 0
        # (2, 4, 0),  # 第2份，使用GPU 0
        # (4, 6, 0),  # 第3份，使用GPU 0
        # (6, 8, 0),  # 第4份，使用GPU 0
        # (8, 10, 1),  # 第5份，使用GPU 1
        # (10, 12, 1),  # 第6份，使用GPU 1
        # (12, 14, 1),  # 第7份，使用GPU 1
        # (14, 16, 1),  # 第8份，使用GPU 1
        # (16, 18, 2),  # 第9份，使用GPU 2
        # (18, 20, 2),  # 第10份，使用GPU 2
        # (20, 22, 2),  # 第11份，使用GPU 2
        # (22, 24, 2),  # 第12份，使用GPU 2
        # (24, 26, 3),  # 第13份，使用GPU 3
        # (26, 28, 3),  # 第14份，使用GPU 3
        # (28, 30, 3),  # 第15份，使用GPU 3
        # (30, 32, 3)  # 第16份，使用GPU 3
        [2, 8, 2],
        [10, 16, 2],
        [18, 24, 2],
        [26, 32, 2]
    ]

    # run_train_model_on_gpu(0, 8, 0)
    # for range_start, range_end, gpu_id in gpu_ranges:
    #     p = Process(target=run_train_model_on_gpu, args=(range_start, range_end, gpu_id))
    #     processes.append(p)
    #     print(f"Starting process for frequencies {range_start} to {range_end}")
    #     p.start()

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

