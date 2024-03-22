import glob
import cv2
import numpy as np


def compute_epe(est_path, gt_path, min_disp, max_disp):
    est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask1 = np.where(gt >= max_disp, zeros, ones)
    mask2 = np.where(gt < min_disp, zeros, ones)
    mask = mask1 & mask2
    # 我们将被测试的图像超出范围的部分也去掉 正式测试时应该删掉！
    '''
    mask3 = np.where(est >= max_disp, zeros, ones)
    mask4 = np.where(est < min_disp, zeros, ones)
    mask = mask & mask3 & mask4'''
    # 本部分结束
    error = np.sum(np.abs(est - gt) * mask)
    nums = np.sum(mask)
    epe = error / nums

    return error, nums, epe


def compute_d1(est_path, gt_path, min_disp, max_disp):
    est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask1 = np.where(gt >= max_disp, zeros, ones)
    mask2 = np.where(gt < min_disp, zeros, ones)
    mask = mask1 & mask2
#我们将被测试的图像超出范围的部分也去掉 正式测试时应该删掉！
    '''
    mask3 = np.where(est >=max_disp, zeros, ones)
    mask4 = np.where(est<min_disp, zeros, ones)
    mask= mask&mask3&mask4'''

#本部分结束
    err_map = np.abs(est - gt) * mask
    err_mask = err_map > 3
    err_disps = np.sum(err_mask.astype('float32'))
    nums = np.sum(mask)
    d1 = err_disps / nums

    return err_disps, nums, d1


def evaluate(est_path, gt_path, min_disp, max_disp):
    error, nums, epe = compute_epe(est_path, gt_path, min_disp, max_disp)
    print('Sum of absolute error: %f, num of valid pixels: %d, end-point-error: %f'
          % (error, int(nums), epe))
    err_disps, nums, d1 = compute_d1(est_path, gt_path, min_disp, max_disp)
    print('Num of error disparities: %d, num of valid pixels: %d, d1: %f'
          % (int(err_disps), int(nums), d1))

def readfromtxt(list_filename):
    with open(list_filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    splits = [line.split() for line in lines]
    est_paths = [x[2].split('/')[-1].replace('DSP','RGB') for x in splits]
    gt_paths = [x[2].split('/')[-1] for x in splits]
    return est_paths, gt_paths

def evaluate_all(est_dir, gt_dir, min_disp, max_disp, testlist=None):
    if testlist == None:
        est_paths = glob.glob(est_dir + '/*.tif')
        gt_paths = glob.glob(gt_dir + '/OMA*DSP.tif')
    else:
        est_paths, gt_paths = readfromtxt(testlist)
        est_paths = [est_dir+x for x in est_paths]
        gt_paths = [gt_dir + x for x in gt_paths ]
    est_paths.sort()
    gt_paths.sort()
    assert len(est_paths) == len(gt_paths)

    total_error, total_nums ,total_epe = 0, 0, 0
    for est_path, gt_path in zip(est_paths, gt_paths):
        error, nums, epe = compute_epe(est_path, gt_path, min_disp, max_disp)
        total_error += error
        total_nums += nums
        total_epe += epe
    print('total test tiff number:', len(est_paths))
    print('\nEnd-point-error: %f' % (total_error / total_nums))
    print('\n epe divided by pic nums', total_epe/len(est_paths))
    total_err_disps, total_nums,total_d1 = 0, 0,0
    for est_path, gt_path in zip(est_paths, gt_paths):
        err_disps, nums, d1 = compute_d1(est_path, gt_path, min_disp, max_disp)
        total_err_disps += err_disps
        total_nums += nums
        total_d1 += d1
    print('\nD1: %f' % (total_err_disps / total_nums))
    print('\n d1 divided by pic nums ', total_d1/len(est_paths))
if __name__ == '__main__':
    est_dir = "D:\Dataset\hmsmnet-igarss\\128128\OMA\\"
    gt_dir = "D:\Dataset\Track2-Truth\\"
    testlist = "D:\\Dataset\\Track2-RGB-ALL\\igarss_test.txt"
    min_disp, max_disp  = [-128,128]
    evaluate_all(est_dir, gt_dir , min_disp, max_disp)
    pass