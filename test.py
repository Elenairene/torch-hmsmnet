import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] ='2'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
from feature import FeatureExtraction
from cost import CostConcatenation
from aggregation import Hourglass, FeatureFusion
from computation import Estimation
from refinement import Refinement
from data_reader import read_disp, read_left, read_right,read_batch, CustomDataset
import time , glob,  gc, argparse
from PIL import Image
import numpy as np
from utils import *
from loss import *
from tensorboardX import SummaryWriter

class HMSMNet(nn.Module):
    def __init__(self, height, width, channel, min_disp, max_disp):
        super(HMSMNet, self).__init__()
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
        self.hourglass0 = Hourglass(filters=32)
        self.hourglass1 = Hourglass(filters=32)
        self.hourglass2 = Hourglass(filters=32)
        self.estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16,input_channels=32)
        self.fusion1 = FeatureFusion(units=32, infeatures=32)
        self.hourglass3 = Hourglass(filters=32)
        self.estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, input_channels=32)
        self.fusion2 = FeatureFusion(units=32, infeatures=32)
        self.hourglass4 = Hourglass(filters=32)
        self.estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, input_channels=32)
        self.refiner = Refinement(filters=64)

    def forward(self,input): #[left_image_ori, right_image_ori, gx_ori, gy_ori]
        # Define input layers
        
        [left_image, right_image, gx, gy] = input
        bs, c,_ ,_ = left_image.shape
        if c!=3:
            left_image = left_image.repeat(1,3,1,1)
            right_image = right_image.repeat(1,3,1,1)
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
        current_size=agg_cost0.size(2)
        target_size = agg_cost1.size(2)
        # Resize agg_cost0 if the sizes are not the same
        if current_size != target_size:
            if current_size > target_size:
                cut_dim = int((current_size - target_size * 2) / 2)  # calculate cut_dim
                if cut_dim!=0:
                    agg_cost0 = agg_cost0[:, :, cut_dim - 1:-1 - cut_dim, cut_dim - 1:-1 - cut_dim, :]  # perform the cut operation
            else:
                cut_dim = -int((current_size - target_size * 2) / 2)  # calculate cut_dim
                if cut_dim !=0 :
                    agg_cost0 = F.pad(agg_cost0, (cut_dim, cut_dim, cut_dim, cut_dim))  # perform the padding operation

        # Assume agg_cost2 is processed using hourglass2
        # Calculate the current and target sizes
        current_size = agg_cost1.size(2)
        target_size = agg_cost2.size(2)

        # Resize agg_cost2 if the sizes are not the same
        if current_size != target_size:
            if current_size > target_size:
                cut_dim = int((current_size / 2 - target_size) / 2)  # calculate cut_dim
                if cut_dim!=0:
                    agg_cost2 = F.pad(agg_cost2, (cut_dim, cut_dim, cut_dim, cut_dim))  # perform the padding operation
            else:
                cut_dim = -int((current_size / 2 - target_size) / 2)  # calculate cut_dim
                if cut_dim!=0:
                    agg_cost2 = agg_cost2[:, :, cut_dim - 1:-1 - cut_dim, cut_dim - 1:-1 - cut_dim, :]  # perform the cut operation
        # Estimation and feature fusion
        disparity2 = self.estimator2(agg_cost2).permute(0,3,1,2)

        fusion_cost1 = self.fusion1([agg_cost2, agg_cost1])
        agg_fusion_cost1 = self.hourglass3(fusion_cost1)

        disparity1 = self.estimator1(agg_fusion_cost1).permute(0,3,1,2)

        fusion_cost2 = self.fusion2([agg_fusion_cost1, agg_cost0])
        agg_fusion_cost2 = self.hourglass4(fusion_cost2)

        disparity0 = self.estimator0(agg_fusion_cost2).permute(0,3,1,2)

        # Refinement
        final_disp = self.refiner([disparity0, left_image, gx, gy])

        return [disparity2, disparity1, disparity0, final_disp]#, disparitiesr

    # def predict(self, left_dir, right_dir, output_dir, testlist=None, mean=None, std = None,indexi=None):
       
    #     whole = len(lefts)
    #     num = 0
    #     total_error, total_nums, total_epe = 0,0,0 
    #     total_err_disps, total_nums, total_d1 = 0,0,0
    #     for left, right, dispname in zip(lefts, rights, disparitygt): #disprname
    #         num = num + 1
    #         print(left)
            

    #         disparity = self.forward([left_image_ori, right_image_ori, gx_ori, gy_ori])
    #         # compute epe d1 # left
    #         error, nums, epe = compute_epe(disparity[-1][0,:,:,0], disp[-1][:,:,0], max_disp= self.max_disp,min_disp=self.min_disp)
    #         if ~np.isnan(error):
    #             total_error += error
    #             total_nums += nums
    #             total_epe += epe
    #             print("epe:" + str(error / nums))
    #             err_disps, nums, d1 = compute_d1(disparity[-1][0,:,:,0], disp[-1][:,:,0], min_disp=self.min_disp, max_disp=self.max_disp)
    #             total_err_disps += err_disps
    #             total_nums += nums
    #             total_d1 += d1
    #             print("d1:" + str(err_disps / nums))
    #         else:
    #             print("nan")
    #         #compute end
    #         # compute epe d1 # right
    #         name = left.replace('Left', 'disparity')
    #         name = name.split('.')[-2]
    #         disparity = disparity[-1][0,:,:,0]
    #         #print(output_dir + name + '.tif')
    #         disparity = (disparity - np.min(disparity)) * 255.0 / (
    #                     np.max(disparity) - np.min(disparity))
    #         disparity = Image.fromarray(disparity)
    #         if disparity.mode == "F":
    #             disparity = disparity.convert('L')
    #     # t1 = time.time()
    #     # print('Total time: ', t1 - t0)
    #     print('\nEnd-point-error: %f' % (total_error / total_nums))
    #     print('\n epe divided by pic nums', total_epe/len(lefts))
    #     print('\nD1: %f' % (total_err_disps / total_nums))
    #     print('\n d1 divided by pic nums ', total_d1/len(lefts))

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')

parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--log_freq', type=int, default=1, help='log freq')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

#DDP
parser.add_argument('--local_rank', type=int, default=-1)

# parse arguments, set seeds
args = parser.parse_args()


# DDP #dis
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

#set seed
set_random_seed(args.seed)




# DDP
# model, optimizer
def hmsmnet(h,w,c,min, max):
    return HMSMNet(h,w,c,min,max)
model = hmsmnet(1024, 1024, 1, -128.0, 64.0)
# model = nn.DataParallel(model)
model.cuda()
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

#conver model to dist
if is_distributed:
    print("Dist Train, Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        find_unused_parameters=True
        # this should be removed if we update BatchNorm stats
        # broadcast_buffers=False,
    )
else:
    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)



if args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])



# dataset, dataloader
test_dataset = CustomDataset(args.testlist, False)

# DDP
if is_distributed:
    
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                       rank=dist.get_rank())

    
    TestImgLoader = torch.utils.data.DataLoader(test_dataset, args.test_batch_size, sampler=test_sampler, num_workers=1,
                                                drop_last=False, pin_memory=True)

else:
    
    TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=4, drop_last=False)
def upsample_auto(disp, disp_target):
    bs,_,h,w = disp.shape
    _,_,ht,wt = disp_target.shape

    scale_factor = ht / h
    disp = F.interpolate(disp, size=(ht, wt), mode='bilinear', align_corners=True)
    disp = disp * scale_factor
    return disp
def test():
  
    # testing
    avg_test_scalars = AverageMeterDict()
    #bestepoch = 0
    #error = 100
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        scalar_outputs = test_sample(sample)
        if (not is_distributed) or (dist.get_rank() == 0):
            if scalar_outputs["epe"]>=0.0:
                # save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
            if batch_idx % args.log_freq == 0:
                print(' Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format( 
                                                                                        batch_idx,
                                                                                        len(TestImgLoader), scalar_outputs["loss"],
                                                                                        time.time() - start_time))
            del scalar_outputs

    if (not is_distributed) or (dist.get_rank() == 0):
        avg_test_scalars = avg_test_scalars.mean()
        print("avg_test_scalars", avg_test_scalars)
            

def test_sample(sample):
    global model
    model.eval()
    left, right,disp = sample["left"].cuda(), sample["right"].cuda(), sample["disp"].cuda()
    gx, gy = sample["gx"].cuda(), sample["gy"].cuda()
    with torch.no_grad():
        [disparity2, disparity1, disparity0, final_disp] = model([left, right,gx ,gy])
    disp2, disp1, disp0, final_disp = upsample_auto(disparity2, disp),  upsample_auto(disparity1, disp), \
    upsample_auto( disparity0, disp), upsample_auto(final_disp, disp)
    weight_loss = [0.5, 0.7, 1.0 , 0.6]
    fullloss = weight_loss[0]*loss_epe(disp2, disp, -args.maxdisp, args.maxdisp) + \
        weight_loss[1]* loss_epe(disp1, disp, -args.maxdisp, args.maxdisp)+ \
            weight_loss[2] *loss_epe(disp0, disp, -args.maxdisp, args.maxdisp) + \
            weight_loss[3]* loss_epe(final_disp, disp, -args.maxdisp, args.maxdisp)
    
    _, nums, epe = compute_epe(final_disp[:,0,:,:].detach(), disp[:,0,:,:].detach(), max_disp= args.maxdisp,min_disp=-args.maxdisp)
    _,_,d1 = compute_d1(final_disp[:,0,:,:].detach(), disp[:,0,:,:].detach(), max_disp= args.maxdisp,min_disp=-args.maxdisp)
    if nums ==0:
        epe = -1.0
 
    scalar_outputs={"epe":epe,
                    "d1":d1, 
    "loss": fullloss}
    return tensor2float(scalar_outputs)
    #error, nums, epe = compute_epe(disparity[-1][0,:,:,0], disp[-1][:,:,0], max_disp= self.max_disp,min_disp=self.min_disp)

def compute_epe(est, gt, min_disp=None, max_disp=None):
    if gt.device.type=='cuda':
        gt = gt.cpu()
    if est.device.type=='cuda':
        est = est.cpu()
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
    error = torch.sum(torch.abs(est - gt)[mask])
    nums = torch.sum(mask)
    epe = error / nums

    return error, nums, epe


def compute_d1(est, gt, min_disp=None, max_disp=None):
    # est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    # gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt.device.type=='cuda':
        gt = gt.cpu()
    if est.device.type=='cuda':
        est = est.cpu()
    # zeros = np.zeros_like(gt, 'int32')
    # ones = np.ones_like(gt, 'int32')
    # mask1 = np.where(gt >= max_disp, zeros, ones)
    # mask2 = np.where(gt < min_disp, zeros, ones)
    # mask = mask1 & mask2
    mask = (gt!=-999.0)&(~torch.isnan(gt))
    mask = mask & (gt>=min_disp) & (gt< max_disp)
#我们将被测试的图像超出范围的部分也去掉 正式测试时应该删掉！
    '''
    mask3 = np.where(est >=max_disp, zeros, ones)
    mask4 = np.where(est<min_disp, zeros, ones)
    mask= mask&mask3&mask4'''

#本部分结束
    err_map = torch.abs(est - gt)[mask]
    err_mask = err_map > 3
    err_disps = torch.sum(err_mask)
    nums = torch.sum(mask)
    d1 = err_disps / nums

    return err_disps, nums, d1

if __name__ == '__main__':
    test()