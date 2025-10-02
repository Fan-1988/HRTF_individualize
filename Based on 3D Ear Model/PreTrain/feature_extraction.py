import torch
import torch.nn as nn
import pdb

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResidualBlock(nn.Module):
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

class FeatureExtraction(nn.Module):
    def __init__(self, n_input_channels=1):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(ResidualBlock, 64, 3)
        self.layer2 = self._make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * ResidualBlock.expansion, 32)
        print("Model successfully initialized")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# test
if __name__== "__main__":
    torch.cuda.empty_cache()
    # print("Before running the model:")
    # print(torch.cuda.memory_summary())

    # pdb.set_trace()
    # 创建一个随机的输入张量，形状为 (batch_size, channels, depth, height, width)
    # input_tensor = torch.randn(1, 1, 32, 32, 32).cuda()
    # print(f'input tensor successfully created: {input_tensor.shape}')
    # model = FeatureExtraction(n_input_channels=1).cuda()
    # output = model(input_tensor)
    # print(f"Output shape: {output.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtraction(n_input_channels=1).to(device)
    input_tensor = torch.randn(1, 1, 32, 32, 32).to(device)
    print(f"Tensor is on device: {input_tensor.device}")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")