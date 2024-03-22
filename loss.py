import torch.nn.functional as F
import torch
import numpy as np
def loss_epe(est, gt, min_disp=None, max_disp=None):
    
    # zeros = np.zeros_like(gt, 'int32')
    # ones = np.ones_like(gt, 'int32')
    mask = (gt != -999.0) &(~torch.isnan(gt))
    # mask1 = np.where(gt >= max_disp, zeros, ones)
    # mask2 = np.where(gt < min_disp, zeros, ones)
    mask = mask & (gt<max_disp) &(gt>=min_disp)
    # mask = mask1 & mask2
    # 我们将被测试的图像超出范围的部分也去掉 正式测试时应该删掉！
    '''
    mask3 = np.where(est >= max_disp, zeros, ones)
    mask4 = np.where(est < min_disp, zeros, ones)
    mask = mask & mask3 & mask4'''
    # 本部分结束
    if mask.sum()==0:
        error = torch.tensor(0.0)
    else:
        error = F.smooth_l1_loss(gt[mask],est[mask], size_average=True)
    # nums = np.sum(mask)
    # epe = error / nums

    return error#, nums, epe