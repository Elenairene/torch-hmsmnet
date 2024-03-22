import torch
import torch.nn as nn
import torch.nn.functional as F

from feature import conv2d, conv2d_bn
class conv_bn_act(nn.Module):
    def __init__(self, inchannels, filters, kernel_size, strides, padding, dilation_rate):
        super(conv_bn_act, self).__init__()
        self.conv = nn.Conv2d(inchannels, filters, kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation_rate, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class Refinement(nn.Module):
    def __init__(self, filters):
        super(Refinement, self).__init__()
        self.conv1 = conv_bn_act(6, filters, 3, 1, 1, 1)
        self.conv2 = conv_bn_act(filters, filters, 3, 1, 1,  1)
        self.conv3 = conv_bn_act(filters, filters, 3, 1, 2, 2)
        self.conv4 = conv_bn_act(filters, filters, 3, 1, 3, 3)
        self.conv5 = conv_bn_act(filters, filters, 3, 1, 1, 1)
        self.conv6 = conv2d(filters, 1, 3, 1, 1, 1)

    def forward(self, inputs):
        # inputs: [disparity, rgb, gx, gy]
        assert len(inputs) == 4

        scale_factor = inputs[1].shape[2] / inputs[0].shape[2]
        disp = F.interpolate(inputs[0], size=(inputs[1].shape[2], inputs[1].shape[3]), mode='bilinear', align_corners=True)
        disp = disp * scale_factor

        concat = torch.cat((disp, inputs[1], inputs[2], inputs[3]), dim=1)
        delta = self.conv1(concat)
        delta = self.conv2(delta)
        delta = self.conv3(delta)
        delta = self.conv4(delta)
        delta = self.conv5(delta)
        delta = self.conv6(delta)
        disp_final = disp + delta

        return disp_final