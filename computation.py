import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0, input_channels=32):
        super(Estimation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.conv = nn.Conv3d(input_channels, 1, kernel_size=3, stride=1, padding=1)  

    def forward(self, inputs):
        x = self.conv(inputs)     # [N, 1, D, H, W]
        x = x.squeeze(1)  # [N, 1, D, H, W]
        x = x.permute(0, 2, 3, 1)  # [N, H, W, D]
        assert x.shape[-1] == self.max_disp - self.min_disp
        candidates = torch.linspace(float(self.min_disp), float(self.max_disp - 1), int(self.max_disp - self.min_disp)).cuda()
        probabilities = F.softmax(-1.0 * x, dim=-1)
        disparities = torch.sum(candidates * probabilities, dim=-1, keepdim=True)
        return disparities