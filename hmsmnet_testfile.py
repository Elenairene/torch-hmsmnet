import torch
import torch.nn as nn
import torch.nn.functional as F
from feature import FeatureExtraction
from cost import CostConcatenation
from aggregation import Hourglass, FeatureFusion
from computation import Estimation
from refinement import Refinement
from data_reader import read_disp, read_left, read_right,read_batch
import time, os , glob
from PIL import Image
import numpy as np
class HMSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None
        self.feature_extraction = FeatureExtraction(filters=16)
        self.cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        self.cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        self.cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        self.hourglass0 = Hourglass(filters=16)
        self.hourglass1 = Hourglass(filters=16)
        self.hourglass2 = Hourglass(filters=16)
        self.estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        self.fusion1 = FeatureFusion(units=16)
        self.hourglass3 = Hourglass(filters=16)
        self.estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        self.fusion2 = FeatureFusion(units=16)
        self.hourglass4 = Hourglass(filters=16)
        self.estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        self.refiner = Refinement(filters=32)

    def forward(self,input): #[left_image_ori, right_image_ori, gx_ori, gy_ori]
        # Define input layers
        [left_image, right_image, gx, gy] = input
        # left_image = nn.Input(shape=(self.height, self.width, self.channel))
        # right_image = nn.Input(shape=(self.height, self.width, self.channel))
        # gx = nn.Input(shape=(self.height, self.width, self.channel))
        # gy = nn.Input(shape=(self.height, self.width, self.channel))

                
        # Feature extraction
        l0, l1, l2 = self.feature_extraction(left_image)
        r0, r1, r2 = self.feature_extraction(right_image)

        # Cost concatenation
        cost_volume0 = self.cost0([l0, r0])
        cost_volume1 = self.cost1([l1, r1])
        cost_volume2 = self.cost2([l2, r2])

        # Hourglass and estimation for each scale
        
        agg_cost0 = self.hourglass0(cost_volume0)
        agg_cost1 = self.hourglass1(cost_volume1)
        agg_cost2 = self.hourglass2(cost_volume2)

        # Cut dim operations
                
        # Resize agg_cost0 if the sizes are not the same
        if current_size != target_size:
            if current_size > target_size:
                cut_dim = int((current_size - target_size * 2) / 2)  # calculate cut_dim
                agg_cost0 = agg_cost0[:, :, cut_dim - 1:-1 - cut_dim, cut_dim - 1:-1 - cut_dim, :]  # perform the cut operation
            else:
                cut_dim = -int((current_size - target_size * 2) / 2)  # calculate cut_dim
                agg_cost0 = F.pad(agg_cost0, (cut_dim, cut_dim, cut_dim, cut_dim))  # perform the padding operation

        # Assume agg_cost2 is processed using hourglass2
        # Calculate the current and target sizes
        current_size = agg_cost1.size(2)
        target_size = agg_cost2.size(2)

        # Resize agg_cost2 if the sizes are not the same
        if current_size != target_size:
            if current_size > target_size:
                cut_dim = int((current_size / 2 - target_size) / 2)  # calculate cut_dim
                agg_cost2 = F.pad(agg_cost2, (cut_dim, cut_dim, cut_dim, cut_dim))  # perform the padding operation
            else:
                cut_dim = -int((current_size / 2 - target_size) / 2)  # calculate cut_dim
                agg_cost2 = agg_cost2[:, :, cut_dim - 1:-1 - cut_dim, cut_dim - 1:-1 - cut_dim, :]  # perform the cut operation
        # Estimation and feature fusion
        disparity2 = self.estimator2(agg_cost2)

        fusion_cost1 = self.fusion1([agg_cost2, agg_cost1])
        agg_fusion_cost1 = self.hourglass3(fusion_cost1)

        disparity1 = self.estimator1(agg_fusion_cost1)

        fusion_cost2 = self.fusion2([agg_fusion_cost1, agg_cost0])
        agg_fusion_cost2 = self.hourglass4(fusion_cost2)

        disparity0 = self.estimator0(agg_fusion_cost2)

        # Refinement
        final_disp = self.refiner([disparity0, left_image, gx, gy])

        return [disparity2, disparity1, disparity0, final_disp]#, disparitiesr

    def predict(self, left_dir, right_dir, output_dir, weights, testlist=None, mean=None, std = None,indexi=None):
        self.model.load_weights(weights)
        # lefts = os.listdir(left_dir)
        # rights = os.listdir(right_dir)
        lefts = []
        rights = []
        if testlist == None:
            if 'png' in left_dir or 'tif' in left_dir:
                lefts = [left_dir]
                rights = [right_dir]
            else:
                lefts += [img.split('\\')[-1] for img in sorted(glob.glob('%s/'% (left_dir)+indexi+'*Left_*.tif' ))]
                rights += [img.split('\\')[-1] for img in sorted(glob.glob('%s/'% (right_dir)+indexi+'*Right_*.tif' ))]
        else:
            lefts, rights, disparitygt = self.readfromtxt(testlist) #disparityrgt
        lefts.sort()
        rights.sort()
        assert len(lefts) == len(rights)
        # t0 = time.time()
        whole = len(lefts)
        num = 0
        total_error, total_nums, total_epe = 0,0,0 
        total_err_disps, total_nums, total_d1 = 0,0,0
        for left, right, dispname in zip(lefts, rights, disparitygt): #disprname
            num = num + 1
            print(left)
            if 'png' in left_dir:
                left_image_ori, gx_ori, gy_ori = read_left(left_dir,mean,std)
                right_image_ori = read_right(right_dir,mean,std)
            elif testlist == None:
                left_image_ori, gx_ori, gy_ori = read_left(os.path.join(left_dir, left),mean,std)
                right_image_ori = read_right(os.path.join(right_dir, right),mean,std)
            else:
                left_image_ori, gx_ori, gy_ori = read_left(left,mean,std)
                right_image_ori = read_right( right,mean,std)
                disp = read_disp(dispname)
                # dispr =  read_disp(disprname)
            left_image_ori = np.expand_dims(left_image_ori, 0)
            gx_ori = np.expand_dims(gx_ori, 0)
            gy_ori = np.expand_dims(gy_ori, 0)
            right_image_ori = np.expand_dims(right_image_ori, 0)

            disparity = self.forward([left_image_ori, right_image_ori, gx_ori, gy_ori])
            # compute epe d1 # left
            error, nums, epe = compute_epe(disparity[-1][0,:,:,0], disp[-1][:,:,0], max_disp= self.max_disp,min_disp=self.min_disp)
            if ~np.isnan(error):
                total_error += error
                total_nums += nums
                total_epe += epe
                print("epe:" + str(error / nums))
                err_disps, nums, d1 = compute_d1(disparity[-1][0,:,:,0], disp[-1][:,:,0], min_disp=self.min_disp, max_disp=self.max_disp)
                total_err_disps += err_disps
                total_nums += nums
                total_d1 += d1
                print("d1:" + str(err_disps / nums))
            else:
                print("nan")
            #compute end
            # compute epe d1 # right
            name = left.replace('Left', 'disparity')
            name = name.split('.')[-2]
            disparity = disparity[-1][0,:,:,0]
            #print(output_dir + name + '.tif')
            disparity = (disparity - np.min(disparity)) * 255.0 / (
                        np.max(disparity) - np.min(disparity))
            disparity = Image.fromarray(disparity)
            if disparity.mode == "F":
                disparity = disparity.convert('L')
        # t1 = time.time()
        # print('Total time: ', t1 - t0)
        print('\nEnd-point-error: %f' % (total_error / total_nums))
        print('\n epe divided by pic nums', total_epe/len(lefts))
        print('\nD1: %f' % (total_err_disps / total_nums))
        print('\n d1 divided by pic nums ', total_d1/len(lefts))

def compute_epe(est, gt, min_disp=None, max_disp=None):
    
    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask = (gt != -999.0) &(~np.isnan(gt))
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
    error = np.sum(np.abs(est - gt)[mask])
    nums = np.sum(mask)
    epe = error / nums

    return error, nums, epe


def compute_d1(est, gt, min_disp=None, max_disp=None):
    # est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    # gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    # mask1 = np.where(gt >= max_disp, zeros, ones)
    # mask2 = np.where(gt < min_disp, zeros, ones)
    # mask = mask1 & mask2
    mask = (gt!=-999.0)&(~np.isnan(gt))
    mask = mask & (gt>=min_disp) & (gt< max_disp)
#我们将被测试的图像超出范围的部分也去掉 正式测试时应该删掉！
    '''
    mask3 = np.where(est >=max_disp, zeros, ones)
    mask4 = np.where(est<min_disp, zeros, ones)
    mask= mask&mask3&mask4'''

#本部分结束
    err_map = np.abs(est - gt)[mask]
    err_mask = err_map > 3
    err_disps = np.sum(err_mask.astype('float32'))
    nums = np.sum(mask)
    d1 = err_disps / nums

    return err_disps, nums, d1

if __name__ == '__main__':
    # # predict
    #mean = [360.026,341.076,352.272]
    #std = [176.613,161.453,197.528]
    left_dir = ["D:\Dataset\Track2-RGB-ALL\JAX_166_022_002_LEFT_RGB.tif"]  # 'D:\\Dataset\\GF7_WLMQ\\22\\2\\'
    right_dir = ["D:\Dataset\Track2-RGB-ALL\JAX_166_022_002_RIGHT_RGB.tif"]  # 'D:\\Dataset\\GF7_WLMQ\\22\\2\\'

    # left_dir = "D:\Dataset\WHU-Stereo\experimentaldata\withgroundtruth\\test\left\\"
    # right_dir = "D:\Dataset\WHU-Stereo\experimentaldata\withgroundtruth\\test\\right\\"
    outs = ["D:\Dataset\\RVL_SatStereo\\disp_hmsmnet\\"]
    index = ["OMA","JAX"]
    test = ["D:\Dataset\WHU-Stereo\experimentaldata\\train_whu_dis.txt"]
    # output_dir = "D:\Dataset\WHU-Stereo\experimentaldata\withgroundtruth\\test\hmsmnet-lr\\"
    for i in range(len(outs)):
        output_dir = outs[i]
        testlist = test[i]
        meani = None
        stdi = None
        #indexi = index[i]
        weights = 'D:\pythonwork\WHU-Stereo-master\HMSMNet\HMSM-Net.h5'
        net = HMSMNet(1024, 1024, 1, -128.0, 64.0)
        # net.build_model()
        net.predict(left_dir[i], right_dir[i], output_dir, weights, testlist = testlist, mean=meani,std=stdi)
    # left_dir = 'the directory of left images'
    # right_dir = 'the directory of right images'
    # output_dir = 'the directory to save results'
    # weights = 'the weight file'
    # net = HMSMNet(1024, 1024, 1, -128.0, 64.0)
    # net.build_model()
    # net.predict(left_dir, right_dir, output_dir, weights)

    # # evaluation
    # gt_dir = 'the directory of ground truth labels'
    # evaluate_all(output_dir, gt_dir, -128.0, 64.0)

    pass