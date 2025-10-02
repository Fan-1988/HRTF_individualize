import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Channel and Spatial Attention Module 部分
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)


class UNet_h(nn.Module):
    def __init__(self):
        super(UNet_h, self).__init__()

        # 人体参数处理
        self.fc1 = nn.Linear(8, 8 * 129)  # Expand from (1, 1, 8) to (129, 1, 8)
        self.fc2 = nn.Linear(8, 8 * 440)  # Further expand from (129, 1, 8) to (129, 1250, 8)

        # Initial Conv1x1 layers for feature fusion
        self.conv1x1_1 = nn.Conv2d(9, 9, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(9, 8, kernel_size=1)

        # Attention Module (CBAM)
        self.cbam = CBAM(channels=8)

        self.conv1x1_3 = nn.Conv2d(8, 1, kernel_size=1)

        # Encoding layers
        self.conv1 = self._conv_block(8, 64, kernel_size=(33, 5), stride=(2, 1), padding=(16, 2))
        self.conv2 = self._conv_block(64, 128, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.conv3 = self._conv_block(128, 256, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.conv4 = self._conv_block(256, 512, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.conv5 = self._conv_block(512, 1024, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))

        # Decoding layers
        self.deconv1 = self._deconv_block(1024, 512, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2),output_padding=(1, 0))
        self.deconv2 = self._deconv_block(512, 256, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.deconv3 = self._deconv_block(256, 128, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2),output_padding=(1, 0))
        self.deconv4 = self._deconv_block(128, 64, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2),output_padding=(1, 0))
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=(33, 5), stride=(2, 1), padding=(16, 2),output_padding=(1, 0))
        #cipic只需要最后一个加outputpadding,hutubs加1345


    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ELU()
        )

    def _deconv_block(self, in_channels, out_channels, kernel_size, stride, padding,output_padding=(0,0)):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding),
            nn.ELU()
        )

    def forward(self, generic_hrtf, anthropometric):
        batch_size = anthropometric.size(0)  # 获取 batch size

        anthropometric = anthropometric.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8)

        # Further expand to (129, 1250, 8)
        anthropometric = anthropometric.view(batch_size, 1, 1, 8)  # Shape: (batch_size, 1, 1, 8)
        anthropometric = self.fc1(anthropometric).view(batch_size, 129, 1, 8)
        anthropometric = self.fc2(anthropometric).view(batch_size, 129, 440, 8) #1250 for cipic

        # (129,1250,8)-> (8,129,1250)
        anthropometric = anthropometric.permute(0, 3, 2, 1)

        # Concatenate anthropometric parameters and HRTF (9,1250,129)
        x = torch.cat((generic_hrtf, anthropometric), dim=1)

        # Initial Conv1x1 layers before CBAM
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)

        # Apply CBAM (1,8,1250,129)
        x = self.cbam(x)

        # Encoding path
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        # Decoding path with skip connections
        d1 = self.deconv1(e5)
        d2 = self.deconv2(d1 + e4)
        d3 = self.deconv3(d2 + e3)
        d4 = self.deconv4(d3 + e2)
        output = F.softplus(self.deconv5(d4+e1))

        return output

class UNet_c(nn.Module):
    def __init__(self):
        super(UNet_c, self).__init__()

        # 人体参数处理
        self.fc1 = nn.Linear(8, 8 * 129)  # Expand from (1, 1, 8) to (129, 1, 8)
        self.fc2 = nn.Linear(8, 8 * 1250)  # Further expand from (129, 1, 8) to (129, 1250, 8)

        # Initial Conv1x1 layers for feature fusion
        self.conv1x1_1 = nn.Conv2d(9, 9, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(9, 8, kernel_size=1)

        # Attention Module (CBAM)
        self.cbam = CBAM(channels=8)

        self.conv1x1_3 = nn.Conv2d(8, 1, kernel_size=1)

        # Encoding layers
        self.conv1 = self._conv_block(8, 64, kernel_size=(33, 5), stride=(2, 1), padding=(16, 2))
        self.conv2 = self._conv_block(64, 128, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.conv3 = self._conv_block(128, 256, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.conv4 = self._conv_block(256, 512, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.conv5 = self._conv_block(512, 1024, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))

        # Decoding layers
        self.deconv1 = self._deconv_block(1024, 512, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.deconv2 = self._deconv_block(512, 256, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.deconv3 = self._deconv_block(256, 128, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.deconv4 = self._deconv_block(128, 64, kernel_size=(33, 5), stride=(2, 2), padding=(16, 2))
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=(33, 5), stride=(2, 1), padding=(16, 2),output_padding=(1, 0))


    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ELU()
        )

    def _deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ELU()
        )

    def forward(self, generic_hrtf, anthropometric):
        batch_size = anthropometric.size(0)  # 获取 batch size

        anthropometric = anthropometric.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8)

        # Further expand to (129, 1250, 8)
        anthropometric = anthropometric.view(batch_size, 1, 1, 8)  # Shape: (batch_size, 1, 1, 8)
        anthropometric = self.fc1(anthropometric).view(batch_size, 129, 1, 8)
        anthropometric = self.fc2(anthropometric).view(batch_size, 129, 1250, 8)

        # (129,1250,8)-> (8,129,1250)
        anthropometric = anthropometric.permute(0, 3, 2, 1)

        # Concatenate anthropometric parameters and HRTF (9,1250,129)
        x = torch.cat((generic_hrtf, anthropometric), dim=1)

        # Initial Conv1x1 layers before CBAM
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)

        # Apply CBAM (1,8,1250,129)
        x = self.cbam(x)

        # Encoding path
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        # Decoding path with skip connections
        d1 = self.deconv1(e5)
        d2 = self.deconv2(d1 + e4)
        d3 = self.deconv3(d2 + e3)
        d4 = self.deconv4(d3 + e2)
        output = F.softplus(self.deconv5(d4+e1))

        return output



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # 人体参数处理部分
        self.fc1 = nn.Linear(8, 8 * 129)  # Expand from (1, 1, 8) to (129, 1, 8)
        self.fc2 = nn.Linear(8, 8 * 1250)  # Further expand from (129, 1, 8) to (129, 1250, 8)

        # MLP 网络部分
        self.fc3 = nn.Linear(1250 * 129 * 8, 1024)  # 输入维度 (batch_size, 1250, 129, 8)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 1250 * 129)  # 输出维度为 1250 * 129

        # Initial Conv1x1 layers for feature fusion
        self.conv1x1_1 = nn.Conv2d(9, 9, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(9, 8, kernel_size=1)

        # Attention Module (CBAM)
        self.cbam = CBAM(channels=8)

        self.conv1x1_3 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, generic_hrtf, anthropometric):
        batch_size = anthropometric.size(0)

        anthropometric = anthropometric.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8)

        # Further expand to (129, 1250, 8)
        anthropometric = anthropometric.view(batch_size, 1, 1, 8)  # Shape: (batch_size, 1, 1, 8)
        anthropometric = self.fc1(anthropometric).view(batch_size, 129, 1, 8)
        anthropometric = self.fc2(anthropometric).view(batch_size, 129, 1250, 8)

        # (129,1250,8) -> (8, 129, 1250)
        anthropometric = anthropometric.permute(0, 3, 2, 1)

        # Concatenate anthropometric parameters and HRTF (9,1250,129)
        x = torch.cat((generic_hrtf, anthropometric), dim=1)

        # Initial Conv1x1 layers before CBAM
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)

        # Apply CBAM (1,8,1250,129)
        x = self.cbam(x)

        # Flatten for MLP
        x = x.view(batch_size, -1)  # Flatten the tensor for MLP, shape: (batch_size, 1250 * 129 * 8)

        # Pass through MLP
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        # Reshape back to (batch_size, 1, 1250, 129)
        output = x.view(batch_size, 1, 1250, 129)

        return output


if __name__=="__main__":
    batch_size = 4  # 设置批大小
    hrtf = torch.randn(batch_size, 1,1250,129)
    anthropometric = torch.randn(batch_size, 8)  # 随机生成 (batch_size, 8) 的人体参数

    # 初始化模型
    # model = UNet_h()
    model = MLP()

    # 前向传播
    output = model(hrtf, anthropometric)
    print("Output shape:", output.shape)
