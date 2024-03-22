import torch
import torch.nn as nn
import torch.nn.functional as F
L2 = 1.0e-5
alpha = 0.2

def conv3d(in_channels,out_channels, kernel_size, stride, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

class Conv3dBn(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, strides, padding, activation=True):
        super(Conv3dBn, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x

class TransConv3dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, activation=True):
        super(TransConv3dBn, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x
class Hourglass(nn.Module):
    def __init__(self, filters):
        super(Hourglass, self).__init__()

        self.conv1 = Conv3dBn(filters, filters, 3, 1, 1, True)
        self.conv2 = Conv3dBn(filters, filters, 3, 1, 1, True)
        self.conv3 = Conv3dBn(filters, 2 * filters, 3, 2, 1, True)
        self.conv4 = Conv3dBn(2 * filters, 2 * filters, 3, 1, 1, True)
        self.conv5 = Conv3dBn(2 * filters, 2 * filters, 3, 2, 1, True)
        self.conv6 = Conv3dBn(2 * filters, 2 * filters, 3, 1, 1, True)
        self.conv7 = TransConv3dBn(2 * filters, 2 * filters, 4, 2, 1, True)
        self.conv8 = TransConv3dBn(2 * filters, filters, 4, 2, 1, True)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        if x1.shape[2] % 2 != 0:
            x1 = nn.functional.pad(x1, (0, 1, 0, 1, 0, 0))
        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        if x2.shape[2] % 2 != 0:
            x2 = nn.functional.pad(x2, (0, 1, 0, 1, 0, 0))
        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x4 = self.conv7(x3)
        x4 += x2
        x5 = self.conv8(x4)
        if x1.shape[2] != x5.shape[2]:
            x1 = nn.functional.pad(x1, (0, 0, 1, 1, 1, 1))
        x5 += x1

        return x5  # [N, C,D,H ,W] # differen for tensorflow
class FeatureFusion(nn.Module):
    def __init__(self, infeatures, units):
        super(FeatureFusion, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')  # 使用Upsample实现上采样
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1,1,1))  # 在PyTorch中，全局平均池化层使用AdaptiveAvgPool3d
        self.fc1 = nn.Linear(infeatures, units)  # 在PyTorch中，全连接层使用Linear
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(units, units)  # 再次使用全连接层
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数
   
    def forward(self, inputs):
        x1 = self.upsample(inputs[0])
        x2 = torch.add(x1, inputs[1])
        v = self.avg_pool3d(x2)[:,:,0,0,0]
        v = self.fc1(v)
        v = F.relu(v)
        v = self.fc2(v)
        v = self.sigmoid(v)
        v1 = 1.0 - v
        v = v.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x2)
        v1 = v1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x2)
        x3 = torch.mul(x1, v)
        x4 = torch.mul(inputs[1], v1)
        x = torch.add(x3, x4)

        return x