import open3d as o3d
import numpy as np
import os


def mirror_ply_across_x(input_file, output_file):
    # 读取 PLY 模型
    mesh = o3d.io.read_triangle_mesh(input_file)

    # 对所有顶点的 x 坐标取负，实现镜像
    vertices = np.asarray(mesh.vertices)
    vertices[:, 0] *= -1  # X轴镜像（你可以换轴看效果）

    # 法线方向也要对应镜像（否则渲染时会有问题）
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        normals[:, 0] *= -1  # 同样对X轴镜像

    # 保存新的镜像模型
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"保存镜像模型到：{output_file}")


def combine_hrtf_data(input_directory, output_file):
    """
    遍历模型推理得到的所有index{i}_hrtf.npy文件，整合其内容为一个 (32, 2522) 维度的npy文件，并保存到指定位置。
    """
    all_hrtf_data = []

    # 遍历目录下所有的.npy文件
    for i in range(32):
        filename = os.path.join(input_directory, f'index{i}_hrtf.npy')
        if os.path.exists(filename):
            # 读取npy文件
            hrtf_data = np.load(filename)
            # 确保每个文件的数据是(1, 2522)维度，并加入到all_hrtf_data中
            if hrtf_data.shape == (1, 2522):
                all_hrtf_data.append(hrtf_data[0])  # 去掉多余的第一维

    # 将所有HRTF数据合并成一个 (32, 2522) 的数组
    all_hrtf_data = np.array(all_hrtf_data)

    # 保存合并后的数据到新的npy文件
    np.save(output_file, all_hrtf_data)

    print(f'HRTF数据已成功整合并保存到 {output_file}！')


def all_combine_hrtf_data(input_directory, output_directory):
    """
    遍历A目录下的每个推理文件夹，将其中的32个index{i}_hrtf.npy文件整合为一个(32, 2522)的数组，
    并保存到B目录中。
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 遍历所有以 _centered_infer 结尾的子文件夹
    for folder_name in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder_name)

        if os.path.isdir(folder_path) and folder_name.endswith("_centered_infer"):
            all_hrtf_data = []

            # 读取32个index{i}_hrtf.npy文件
            for i in range(32):
                file_path = os.path.join(folder_path, f"index{i}_hrtf.npy")
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    if data.shape == (1, 2522):
                        all_hrtf_data.append(data[0])  # 去除第1维
                    else:
                        print(f" 警告: {file_path} 的维度不是 (1, 2522)，已跳过。")
                else:
                    print(f"缺失文件: {file_path}")

            if len(all_hrtf_data) == 32:
                combined_array = np.array(all_hrtf_data)  # shape: (32, 2522)

                # 构造输出文件名
                base_name = folder_name.replace("_centered_infer", "_infer_32")
                output_path = os.path.join(output_directory, f"{base_name}.npy")

                # 保存整合后的文件
                np.save(output_path, combined_array)
                print(f"✅ 保存成功: {output_path}")
            else:
                print(f"❌ 跳过：{folder_name} 中有效的 index{i}_hrtf.npy 文件数量不足 32。")


def split_hrtf_data(input_file, output_left_file, output_right_file):
    """
    将HRTF数据从 (2522, 2, 32) 的维度拆分为左右耳数据，并保存为新的 .npy 文件。

    参数:
    - input_file: 原始数据的文件路径 (npy格式)，形状为 (2522, 2, 32)
    - output_left_file: 左耳数据保存的文件路径
    - output_right_file: 右耳数据保存的文件路径
    """
    # 读取原始的 HRTF 数据
    hrtf_data = np.load(input_file)

    # 左耳数据, 维度: (2522, 32)
    left_ear_data = hrtf_data[:, 1, :]

    # 转置为 (32, 2522) 维度
    left_ear_data = left_ear_data.T  # 维度: (32, 2522)

    # 保存左耳和右耳的数据到新的 .npy 文件
    np.save(output_left_file, left_ear_data)


target_ids = [
    "0438", "0827", "0393", "1001", "0540", "1242", "0474", "0151", "1121", "0429",
    "0312", "0345", "0575", "0646", "0872", "0965", "1197", "0574", "1194", "0858",
    "0055", "0544", "1149", "0657", "0399", "0468", "0985", "1137", "0756", "0048",
    "0318", "0836", "1080", "0276", "0247", "0786", "0545", "0837", "0883", "0565",
    "0065", "0075", "0303", "1061", "0279", "0035", "0133", "0042", "0625", "0593",
    "0483", "0794", "0194", "1201", "0931", "0880", "0452", "0286", "0387", "0140",
    "0908", "0251", "0125", "0484", "0206", "0039", "0712", "0353", "0370", "0331",
    "0209", "0141", "0512", "0333", "1011", "0877", "0671", "0329", "0024", "1083",
    "0842", "0768", "1094", "1103", "0355", "0920", "0911", "0884", "1228", "0879",
    "1185", "1081", "1213", "1058", "0530", "0123", "0487", "0677", "0528", "1045",
    "0339", "0201", "1117", "0860", "0938", "0573", "0155", "0959", "0498", "0792",
    "1246", "0548", "0192", "0309", "1178", "0850", "0236", "0337", "0633", "0758",
    "0930", "1115", "0313", "0882", "0662", "1179", "0204", "0660", "0282", "0767",
    "0859", "1163", "0762", "0776", "0046", "1170", "1181", "0463", "1093", "0811",
    "0732", "1063", "0659", "0609", "0674", "0202", "0268", "1128", "0744", "1029",
    "1186", "0726", "0535", "0478", "0271", "0736", "0026", "1108", "0234", "1226",
    "0100", "0588", "0258", "0138", "0398", "0569", "0232", "0143", "0857", "1162",
    "0645", "0902", "0620", "1169", "0425", "0485", "1006", "0119", "0073", "1233",
    "0246", "0322", "0241", "1147", "0506", "0696", "1195", "1077", "0467", "0706",
    "0458", "0404", "0443", "1153", "0084", "0519", "0898", "0027", "0702", "0631",
    "1079", "1097", "0152", "0159", "0319", "0775", "1155", "0501", "0085", "0373",
    "0906", "0606", "0473", "0014", "0607", "0678", "0694", "0244", "0564", "0068",
    "0356", "1232", "0818", "1016", "0228", "0270", "0296", "0252", "0647", "0489",
    "1216", "0742", "0196", "0098", "0721", "0868", "0727", "0954", "0118", "0471",
    "0492", "1238", "0681", "0324", "0102", "0025", "0275", "1009", "0245", "0394"
]


def save_left_ear_data(
        base_input_dir: str,
        output_dir: str,
        target_ids: list
):

    os.makedirs(output_dir, exist_ok=True)

    for id_str in target_ids:
        input_file = os.path.join(base_input_dir, f"chedar_{id_str}_UV1m.npy")
        output_file = os.path.join(output_dir, f"uv1m_{id_str}.npy")

        # 加载数据并提取左耳
        hrtf_data = np.load(input_file)
        left_ear = hrtf_data[:, 1, :].T  # (32, 2522)

        np.save(output_file, left_ear)
        print(f"Saved left ear: {output_file}")


if __name__ == '__main__':
    # input_directory = r'E:\grad_project\new_0001'  # 替换为实际输入目录路径
    # output_file = r'E:\grad_project\infer_CHEDAR_3\new_0001_32.npy'  # 替换为实际结果文件路径和名称
    # combine_hrtf_data(input_directory, output_file)

    # ip = r"E:\grad_project\ground_truth\chedar_0438_UV1m.npy"
    # left = r"E:\grad_project\ground_truth\left_ear_chedar_00438_UV1m.npy"
    # right = r"E:\grad_project\ground_truth\right_ear_chedar_0438_UV1m.npy"
    # split_hrtf_data(ip, left, right)

    # input_dir = "/data1/fsj/precombine/all_test_infer/"
    # output_dir = "/data1/fsj/precombine/chedartrain/all_test_infer_32_250"
    # all_combine_hrtf_data(input_dir, output_dir)

    input_dir = "/data1/fsj/combine/CHEDAR/hrtfs_erb64_log"
    output_dir = "/data1/fsj/combine/CHEDAR/all_test_gtruth_left"
    save_left_ear_data(input_dir, output_dir, target_ids)
