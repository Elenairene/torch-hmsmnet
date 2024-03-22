import torch
import torch.nn as nn


L2 = 1.0e-5


def conv2d(inchannels, filters, kernel_size, strides, padding, dilation_rate):
    return nn.Conv2d(in_channels= inchannels, out_channels=filters, kernel_size=kernel_size,
                     stride=strides, padding=padding, dilation=dilation_rate)


def conv2d_bn(inchannels,filters, kernel_size, strides, padding, dilation_rate, activation):
    conv = nn.Conv2d(in_channels=inchannels, out_channels=filters, kernel_size=kernel_size,
                     stride=strides, padding=padding, dilation=dilation_rate, bias=False)
    bn = nn.BatchNorm2d(filters)
    relu = nn.ReLU()

    layers = [conv, bn]
    if activation:
        layers.append(relu)

    return nn.Sequential(*layers)

class AvgPoolConv(nn.Module):
    def __init__(self, pool_size, inchannels, filters):
        super(AvgPoolConv, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        self.conv = nn.Conv2d(inchannels, filters, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inchannels, filters, dilation_rate, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2d_bn(inchannels, filters, 3, 1, padding, dilation_rate, True)
        self.conv2 = conv2d_bn(inchannels, filters, 3, 1, padding, dilation_rate, False)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        x = self.relu(x)
        return x

def make_blocks(inchannels, filters, dilation_rate, num):
    blocks = []
    for _ in range(num):
        blocks.append(BasicBlock(inchannels, filters, dilation_rate))
    return nn.Sequential(*blocks)

def make_blocks2(inchannels, filters, dilation_rate, num):
    blocks = []
    for _ in range(num):
        blocks.append(BasicBlock(inchannels, filters, dilation_rate, padding=2))
    return nn.Sequential(*blocks)
def make_blocks4(inchannels, filters, dilation_rate, num):
    blocks = []
    for _ in range(num):
        blocks.append(BasicBlock(inchannels, filters, dilation_rate, padding=4))
    return nn.Sequential(*blocks)
class FeatureExtraction(nn.Module):
    def __init__(self, filters):
        super(FeatureExtraction, self).__init__()
        self.conv0_1 = conv2d_bn(3, filters, 5, 2, 2, 1, True)
        self.conv0_2 = conv2d_bn(filters, 2 * filters, 5, 2, 2, 1, True)
        self.conv1_0 = make_blocks(2 * filters, 2 * filters, 1, 4)
        self.conv1_1 = make_blocks2(2 * filters,2 * filters,  2, 2)
        self.conv1_2 = make_blocks4(2 * filters, 2 * filters, 4, 2)
        self.conv1_3 = make_blocks(2 * filters, 2 * filters, 1, 2)
        self.branch0 = AvgPoolConv(1, 2 * filters, filters)
        self.branch1 = AvgPoolConv(2, 2 * filters, filters)
        self.branch2 = AvgPoolConv(4, 2 * filters , filters)

    def forward(self, inputs):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return [x0, x1, x2]  # [1/4, 1/8, 1/16]