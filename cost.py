import torch
import torch.nn as nn
import torch.nn.functional as F

class CostConcatenation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(CostConcatenation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def forward(self, inputs): # N C, H ,W
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(F.pad(torch.cat((inputs[0][:, :, :,:i], inputs[1][:, :, :,-i:]), dim=1), (0, -i)))
            elif i > 0:
                cost_volume.append(F.pad(torch.cat((inputs[0][:, :, :,i:], inputs[1][:, :, :,:-i]), dim=1), (i, 0)))
            else:
                cost_volume.append(torch.cat((inputs[0], inputs[1]), dim=1))
        cost_volume = torch.stack(cost_volume, 2)
        return cost_volume #N C(D) H W


class CostDifference(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(CostDifference, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def forward(self, inputs):
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(F.pad(inputs[0][:, :, :i, :] - inputs[1][:, :, -i:, :], (0, 0, 0, -i)))
            elif i > 0:
                cost_volume.append(F.pad(inputs[0][:, :, i:, :] - inputs[1][:, :, :-i, :], (0, 0, i, 0)))
            else:
                cost_volume.append(inputs[0] - inputs[1])
        cost_volume = torch.stack(cost_volume, 1)
        return cost_volume