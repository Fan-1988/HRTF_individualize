import torch

# 定义 .pth 文件路径
# pretrain_path = r'E:\NewDesktop\best_test_model.pth'
pretrain_path = r"E:\NewDesktop\feature_extraction_best_params.pth"
model_data = torch.load(pretrain_path,map_location='cpu')


# 加载 .pth 文件
# state_dict = torch.load(pretrain_path, map_location='cpu')

# 打印 .pth 文件的内容
# print("Keys in the state_dict:")
# for key in state_dict.keys():
#     print(key)


# 检查文件内容
for key, value in model_data.items():
    print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")