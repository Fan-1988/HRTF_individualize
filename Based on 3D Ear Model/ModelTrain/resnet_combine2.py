import math
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


# 各卷积层输入通道数的设定值

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=793,
                 n_classes2=2048):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.fc2 = nn.Linear(block_inplanes[3] * block.expansion, n_classes2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input size", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("first relu", x.shape)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        layer1 = x
        # print("layer1", x.shape)
        x = self.layer2(x)
        # print("layer2", x.shape)
        layer2 = x
        x = self.layer3(x)
        # print("layer3", x.shape)
        layer3 = x
        x = self.layer4(x)
        # print("layer4", x.shape)
        layer4 = x

        x = self.avgpool(x)
        # print("avg pool", x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out1 = self.fc(x)
        out2 = self.fc2(x)
        return out1, out2


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model

    # if pretrain_path:
    #     try:
    #         pretrain_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    #         # 获取当前模型的state_dict
    #         model_dict = model.state_dict()
    #         # 删除与fc层相关的权重项
    #         pretrain_dict = {k: v for k, v in pretrain_dict.itemds() if 'fc' not in k}
    #         # 更新当前模型的state_dict
    #         model_dict.update(pretrain_dict)
    #         # 加载更新后的state_dict
    #         model.load_state_dict(model_dict)
    #
    #         in_features = model.fc.in_features
    #         # new fc layer for different output classes1
    #         model.fc = nn.Linear(in_features, 793)
    #         torch.cuda.empty_cache()
    #         model.to(torch.device('cuda:2'))
    #         if next(model.parameters()).is_cuda:
    #             print(f"Model is successfully loaded onto GPU.")
    #         else:
    #             print(f"Model is not loaded onto GPU.")
    #
    #     except RuntimeError as e:
    #         print(f"Runtime error: {e}")
    #         # Clean up and exit
    #         torch.cuda.empty_cache()
    #         sys.exit("Stopping training due to error loading pre-trained model.")
    #     except Exception as e:
    #         print(f"Error loading pre-trained model from {pretrain_path}: {e}")
    #         # 停止训练
    #         sys.exit("Stopping training due to error loading pre-trained model.")


def count_model_parameters(model):
    """
    计算模型的参数量
    :param model: PyTorch 模型
    :return: 参数总量（单位：百万）
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6  # 转换为百万参数量


# 示例调用方式
if __name__ == "__main__":
    model_depth = 34
    n_input_channels = 1
    n_classes1 = 2522
    n_classes2 = 2048
    model = generate_model(model_depth=model_depth,
                           n_input_channels=n_input_channels,
                           conv1_t_size=3,
                           conv1_t_stride=1,
                           no_max_pool=True,
                           n_classes=n_classes1,
                           n_classes2=n_classes2
                           )
    num_params_million = count_model_parameters(model)
    print(f"ResNet-{model_depth} 参数量: {num_params_million:.2f}M")
