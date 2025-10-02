
import math
import os
from functools import partial
from mydataset import PretrainDatasetChedar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


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
                 n_classes=2522,
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
        # print("layer2 ", x.shape)
        layer2 = x
        x = self.layer3(x)
        # print("layer3", x.shape)
        layer3 = x
        x = self.layer4(x)
        # print("layer4", x.shape)
        layer4 = x

        x = self.avgpool(x)
        # print("avg pool", x.shape)
        # x = x.view(x.size(0), -1)
        # # print(x.shape)
        # out1 = self.fc(x)
        # out2 = self.fc2(x)
        return x


def generate_model(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model

def test_pretrain():
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print(f"using GPU {device}")
    else:
        device = torch.device("cpu")
        print("using cpu")
    model = PretrainModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    base_dir = '/data1/fsj/myHRTF/Pretrain_get_param'
    output_dir = os.path.join(base_dir, 'param_result')
    file_path = os.path.join(output_dir, 'feature_extraction_best_params.pth')
    # 数据加载
    mesh_dir = '/data1/fsj/myHRTF/3Dmeshvoxel'
    csv_file = '/data1/fsj/myHRTF/measures.csv'
    dataset = PretrainDatasetChedar(mesh_dir, csv_file)

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print('all data loaded')
    # Training loop with early stopping
    num_epochs = 10
    patience = 3
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for meshes, body_params in train_loader:
            meshes = meshes.to(device)
            body_params = body_params.to(device)

            optimizer.zero_grad()
            outputs = model(meshes)
            loss = 1 - cosine_similarity(outputs, body_params).mean()  # Assuming cosine_similarity is defined
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * meshes.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for meshes, body_params in val_loader:
                meshes = meshes.to(device)
                body_params = body_params.to(device)
                outputs = model(meshes)
                val_loss += (1 - cosine_similarity(outputs, body_params).mean()).item() * meshes.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.feature_extraction.state_dict(), file_path)
            print("Saved better model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early due to no improvement")
                break

class PretrainModel(nn.Module):
    def __init__(self):
        super(PretrainModel, self).__init__()
        self.feature_extraction = generate_model(n_input_channels=1, conv1_t_size=3, conv1_t_stride=1,
                                                 no_max_pool=True)
        self.fc1 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    test_pretrain()




