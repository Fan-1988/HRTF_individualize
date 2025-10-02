import os
import numpy as np
import torch
import resnet_combine2


def create_model_and_infer(input_tensor, model_path, o1_pth, o2_pth, device='cuda:0'):
    model_depth = 34
    n_input_channels = 1
    n_classes1 = 2522
    n_classes2 = 2048

    model = resnet_combine2.generate_model(
        model_depth=model_depth,
        n_input_channels=n_input_channels,
        conv1_t_size=3,
        conv1_t_stride=1,
        no_max_pool=True,
        n_classes=n_classes1,
        n_classes2=n_classes2
    )

    #  加载模型权重
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} does not exist. Skipping...")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully from {model_path} on {device}.")

    # 3. 推理
    with torch.no_grad():
        output1, output2 = model(input_tensor)

    # 4. 将输出转到CPU并保存为Numpy文件
    output1_data = output1.cpu().numpy()
    output2_data = output2.cpu().numpy()

    os.makedirs(os.path.dirname(o1_pth), exist_ok=True)
    os.makedirs(os.path.dirname(o2_pth), exist_ok=True)

    np.save(o1_pth, output1_data)
    np.save(o2_pth, output2_data)

    print(f"Outputs saved to {o1_pth} and {o2_pth}.")

    # 5. 释放资源
    del model
    torch.cuda.empty_cache()


def main(input_file_path, device='cuda:0'):
    # 1. 加载一次输入数据
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file {input_file_path} does not exist.")

    input_data = np.load(input_file_path)
    print("Loaded input npy shape:", input_data.shape)

    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (B, C, ...)

    # 2. 循环推理
    for i in range(32):
        model_path = f"/data1/fsj/precombine/chedaroutput/index{i}/best_test_model.pth"
        o1_pth = f"/data1/fsj/precombine/new_0001/index{i}_hrtf.npy"
        o2_pth = f"/data1/fsj/precombine/new_0001/index{i}_sh.npy"

        print(f"\nProcessing model {model_path}...")
        create_model_and_infer(input_tensor, model_path, o1_pth, o2_pth, device)


test_samples = [
    "chedar_0438_centered.npy", "chedar_0827_centered.npy", "chedar_0393_centered.npy", "chedar_1001_centered.npy",
    "chedar_0540_centered.npy", "chedar_1242_centered.npy", "chedar_0474_centered.npy", "chedar_0151_centered.npy",
    "chedar_1121_centered.npy", "chedar_0429_centered.npy", "chedar_0312_centered.npy", "chedar_0345_centered.npy",
    "chedar_0575_centered.npy", "chedar_0646_centered.npy", "chedar_0872_centered.npy", "chedar_0965_centered.npy",
    "chedar_1197_centered.npy", "chedar_0574_centered.npy", "chedar_1194_centered.npy", "chedar_0858_centered.npy",
    "chedar_0055_centered.npy", "chedar_0544_centered.npy", "chedar_1149_centered.npy", "chedar_0657_centered.npy",
    "chedar_0399_centered.npy", "chedar_0468_centered.npy", "chedar_0985_centered.npy", "chedar_1137_centered.npy",
    "chedar_0756_centered.npy", "chedar_0048_centered.npy", "chedar_0318_centered.npy", "chedar_0836_centered.npy",
    "chedar_1080_centered.npy", "chedar_0276_centered.npy", "chedar_0247_centered.npy", "chedar_0786_centered.npy",
    "chedar_0545_centered.npy", "chedar_0837_centered.npy", "chedar_0883_centered.npy", "chedar_0565_centered.npy",
    "chedar_0065_centered.npy", "chedar_0075_centered.npy", "chedar_0303_centered.npy", "chedar_1061_centered.npy",
    "chedar_0279_centered.npy", "chedar_0035_centered.npy", "chedar_0133_centered.npy", "chedar_0042_centered.npy",
    "chedar_0625_centered.npy", "chedar_0593_centered.npy", "chedar_0483_centered.npy", "chedar_0794_centered.npy",
    "chedar_0194_centered.npy", "chedar_1201_centered.npy", "chedar_0931_centered.npy", "chedar_0880_centered.npy",
    "chedar_0452_centered.npy", "chedar_0286_centered.npy", "chedar_0387_centered.npy", "chedar_0140_centered.npy",
    "chedar_0908_centered.npy", "chedar_0251_centered.npy", "chedar_0125_centered.npy", "chedar_0484_centered.npy",
    "chedar_0206_centered.npy", "chedar_0039_centered.npy", "chedar_0712_centered.npy", "chedar_0353_centered.npy",
    "chedar_0370_centered.npy", "chedar_0331_centered.npy", "chedar_0209_centered.npy", "chedar_0141_centered.npy",
    "chedar_0512_centered.npy", "chedar_0333_centered.npy", "chedar_1011_centered.npy", "chedar_0877_centered.npy",
    "chedar_0671_centered.npy", "chedar_0329_centered.npy", "chedar_0024_centered.npy", "chedar_1083_centered.npy",
    "chedar_0842_centered.npy", "chedar_0768_centered.npy", "chedar_1094_centered.npy", "chedar_1103_centered.npy",
    "chedar_0355_centered.npy", "chedar_0920_centered.npy", "chedar_0911_centered.npy", "chedar_0884_centered.npy",
    "chedar_1228_centered.npy", "chedar_0879_centered.npy", "chedar_1185_centered.npy", "chedar_1081_centered.npy",
    "chedar_1213_centered.npy", "chedar_1058_centered.npy", "chedar_0530_centered.npy", "chedar_0123_centered.npy",
    "chedar_0487_centered.npy", "chedar_0677_centered.npy", "chedar_0528_centered.npy", "chedar_1045_centered.npy",
    "chedar_0339_centered.npy", "chedar_0201_centered.npy", "chedar_1117_centered.npy", "chedar_0860_centered.npy",
    "chedar_0938_centered.npy", "chedar_0573_centered.npy", "chedar_0155_centered.npy", "chedar_0959_centered.npy",
    "chedar_0498_centered.npy", "chedar_0792_centered.npy", "chedar_1246_centered.npy", "chedar_0548_centered.npy",
    "chedar_0192_centered.npy", "chedar_0309_centered.npy", "chedar_1178_centered.npy", "chedar_0850_centered.npy",
    "chedar_0236_centered.npy", "chedar_0337_centered.npy", "chedar_0633_centered.npy", "chedar_0758_centered.npy",
    "chedar_0930_centered.npy", "chedar_1115_centered.npy", "chedar_0313_centered.npy", "chedar_0882_centered.npy",
    "chedar_0662_centered.npy", "chedar_1179_centered.npy", "chedar_0204_centered.npy", "chedar_0660_centered.npy",
    "chedar_0282_centered.npy", "chedar_0767_centered.npy", "chedar_0859_centered.npy", "chedar_1163_centered.npy",
    "chedar_0762_centered.npy", "chedar_0776_centered.npy", "chedar_0046_centered.npy", "chedar_1170_centered.npy",
    "chedar_1181_centered.npy", "chedar_0463_centered.npy", "chedar_1093_centered.npy", "chedar_0811_centered.npy",
    "chedar_0732_centered.npy", "chedar_1063_centered.npy", "chedar_0659_centered.npy", "chedar_0609_centered.npy",
    "chedar_0674_centered.npy", "chedar_0202_centered.npy", "chedar_0268_centered.npy", "chedar_1128_centered.npy",
    "chedar_0744_centered.npy", "chedar_1029_centered.npy", "chedar_1186_centered.npy", "chedar_0726_centered.npy",
    "chedar_0535_centered.npy", "chedar_0478_centered.npy", "chedar_0271_centered.npy", "chedar_0736_centered.npy",
    "chedar_0026_centered.npy", "chedar_1108_centered.npy", "chedar_0234_centered.npy", "chedar_1226_centered.npy",
    "chedar_0100_centered.npy", "chedar_0588_centered.npy", "chedar_0258_centered.npy", "chedar_0138_centered.npy",
    "chedar_0398_centered.npy", "chedar_0569_centered.npy", "chedar_0232_centered.npy", "chedar_0143_centered.npy",
    "chedar_0857_centered.npy", "chedar_1162_centered.npy", "chedar_0645_centered.npy", "chedar_0902_centered.npy",
    "chedar_0620_centered.npy", "chedar_1169_centered.npy", "chedar_0425_centered.npy", "chedar_0485_centered.npy",
    "chedar_1006_centered.npy", "chedar_0119_centered.npy", "chedar_0073_centered.npy", "chedar_1233_centered.npy",
    "chedar_0246_centered.npy", "chedar_0322_centered.npy", "chedar_0241_centered.npy", "chedar_1147_centered.npy",
    "chedar_0506_centered.npy", "chedar_0696_centered.npy", "chedar_1195_centered.npy", "chedar_1077_centered.npy",
    "chedar_0467_centered.npy", "chedar_0706_centered.npy", "chedar_0458_centered.npy", "chedar_0404_centered.npy",
    "chedar_0443_centered.npy", "chedar_1153_centered.npy", "chedar_0084_centered.npy", "chedar_0519_centered.npy",
    "chedar_0898_centered.npy", "chedar_0027_centered.npy", "chedar_0702_centered.npy", "chedar_0631_centered.npy",
    "chedar_1079_centered.npy", "chedar_1097_centered.npy", "chedar_0152_centered.npy", "chedar_0159_centered.npy",
    "chedar_0319_centered.npy", "chedar_0775_centered.npy", "chedar_1155_centered.npy", "chedar_0501_centered.npy",
    "chedar_0085_centered.npy", "chedar_0373_centered.npy", "chedar_0906_centered.npy", "chedar_0606_centered.npy",
    "chedar_0473_centered.npy", "chedar_0014_centered.npy", "chedar_0607_centered.npy", "chedar_0678_centered.npy",
    "chedar_0694_centered.npy", "chedar_0244_centered.npy", "chedar_0564_centered.npy", "chedar_0068_centered.npy",
    "chedar_0356_centered.npy", "chedar_1232_centered.npy", "chedar_0818_centered.npy", "chedar_1016_centered.npy",
    "chedar_0228_centered.npy", "chedar_0270_centered.npy", "chedar_0296_centered.npy", "chedar_0252_centered.npy",
    "chedar_0647_centered.npy", "chedar_0489_centered.npy", "chedar_1216_centered.npy", "chedar_0742_centered.npy",
    "chedar_0196_centered.npy", "chedar_0098_centered.npy", "chedar_0721_centered.npy", "chedar_0868_centered.npy",
    "chedar_0727_centered.npy", "chedar_0954_centered.npy", "chedar_0118_centered.npy", "chedar_0471_centered.npy",
    "chedar_0492_centered.npy", "chedar_1238_centered.npy", "chedar_0681_centered.npy", "chedar_0324_centered.npy",
    "chedar_0102_centered.npy", "chedar_0025_centered.npy", "chedar_0275_centered.npy", "chedar_1009_centered.npy",
    "chedar_0245_centered.npy", "chedar_0394_centered.npy"
]

test_samples_1 = [
    "chedar_0438_centered.npy", "chedar_0827_centered.npy", "chedar_0393_centered.npy", "chedar_1001_centered.npy",
    "chedar_0540_centered.npy", "chedar_1242_centered.npy", "chedar_0474_centered.npy", "chedar_0151_centered.npy",
    "chedar_1121_centered.npy", "chedar_0429_centered.npy", "chedar_0312_centered.npy", "chedar_0345_centered.npy",
    "chedar_0575_centered.npy", "chedar_0646_centered.npy", "chedar_0872_centered.npy", "chedar_0965_centered.npy",
    "chedar_1197_centered.npy", "chedar_0574_centered.npy", "chedar_1194_centered.npy", "chedar_0858_centered.npy",
    "chedar_0055_centered.npy", "chedar_0544_centered.npy", "chedar_1149_centered.npy", "chedar_0657_centered.npy",
    "chedar_0399_centered.npy", "chedar_0468_centered.npy", "chedar_0985_centered.npy", "chedar_1137_centered.npy",
    "chedar_0756_centered.npy", "chedar_0048_centered.npy", "chedar_0318_centered.npy", "chedar_0836_centered.npy",
    "chedar_1080_centered.npy", "chedar_0276_centered.npy", "chedar_0247_centered.npy", "chedar_0786_centered.npy",
    "chedar_0545_centered.npy", "chedar_0837_centered.npy", "chedar_0883_centered.npy", "chedar_0565_centered.npy",
    "chedar_0065_centered.npy", "chedar_0075_centered.npy", "chedar_0303_centered.npy", "chedar_1061_centered.npy",
    "chedar_0279_centered.npy", "chedar_0035_centered.npy", "chedar_0133_centered.npy", "chedar_0042_centered.npy",
    "chedar_0625_centered.npy", "chedar_0593_centered.npy", "chedar_0483_centered.npy", "chedar_0794_centered.npy",
    "chedar_0194_centered.npy", "chedar_1201_centered.npy", "chedar_0931_centered.npy", "chedar_0880_centered.npy",
    "chedar_0452_centered.npy", "chedar_0286_centered.npy", "chedar_0387_centered.npy", "chedar_0140_centered.npy",
]


def all_main(input_dir, model_base_path, output_infer_path, device='cuda:0'):
    for npy_file in test_samples_1:
        input_file_path = os.path.join(input_dir, npy_file)
        print(f"\nProcessing input file: {input_file_path}")

        # Load input data
        input_data = np.load(input_file_path)
        print("Loaded input npy shape:", input_data.shape)

        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (B, C, ...)

        # Create a folder for each test sample (e.g., chedar_0647_infer)
        output_dir = os.path.join(output_infer_path, f"{os.path.splitext(npy_file)[0]}_infer")
        os.makedirs(output_dir, exist_ok=True)

        # Loop through the models for each test sample (32 models)
        for i in range(32):  # Assuming 32 models to process
            model_path = f"{model_base_path}/index{i}/best_test_model.pth"
            o1_pth = os.path.join(output_dir, f"index{i}_hrtf.npy")
            o2_pth = os.path.join(output_dir, f"index{i}_sh.npy")

            print(f"Processing model {model_path} for {npy_file}...")
            create_model_and_infer(input_tensor, model_path, o1_pth, o2_pth, device)


if __name__ == "__main__":
    # input_file_path = "/data1/fsj/combine/CHEDAR/voxel/chedar_0001_centered.npy"
    # device = 'cuda:3'
    # main(input_file_path, device)

    input_dir = "/data1/fsj/combine/CHEDAR/voxel"  # Directory containing the .npy files
    model_base_path = "/data1/fsj/precombine/chedaroutput"  # Base path where models are located
    device = 'cuda:0'  # Set the device
    output_infer_path = "/data1/fsj/precombine/all_test_infer"
    all_main(input_dir, model_base_path, output_infer_path, device)
