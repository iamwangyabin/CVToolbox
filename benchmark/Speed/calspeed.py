import torch
import time

import torch
from torch import nn
from torch.nn import functional as f
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def cal_speed(det,des):
    with torch.no_grad():
        input_sample = torch.randn((1, 3, 320, 240),device = device)
        for i in range(10):
            det(input_sample)
        total_time = 0
        for i in range(1000):
            torch.cuda.synchronize()
            time0 = time.time()
            det(input_sample)
            torch.cuda.synchronize()
            time1 = time.time()
            total_time += (time1 - time0)
        print(total_time/1000)
        patch_sample = torch.randn((1000, 1, 32, 32),device = device)
        for i in range(10):
            des(patch_sample)
        total_time = 0
        for i in range(1000):
            torch.cuda.synchronize()
            time0 = time.time()
            des(patch_sample)
            torch.cuda.synchronize()
            time1 = time.time()
            total_time += (time1 - time0)
        print(total_time/1000)

#
# def cal_our1060():
#     import sys
#     sys.path.extend(['/home/wang/workspace/FasterNet/NASNet_0.1'])
#     from model.det import RFDet
#     from model.des import HardNetNeiMask
#     from model.network import Network
#     from config import cfg
#     from hpatch_dataset import *
#     # CUDA
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     # Creating CNN model
#     det = RFDet(cfg.TRAIN.score_com_strength,cfg.TRAIN.scale_com_strength,cfg.TRAIN.NMS_THRESH,cfg.TRAIN.NMS_KSIZE,cfg.TRAIN.TOPK,
#         cfg.MODEL.GAUSSIAN_KSIZE,cfg.MODEL.GAUSSIAN_SIGMA,cfg.MODEL.KSIZE, cfg.MODEL.padding,cfg.MODEL.dilation,cfg.MODEL.scale_list,)
#     des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
#     model = Network(det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK)
#     model.cuda()
#     det.cuda()
#     des.cuda()
#     cal_speed(det, des)



def cal_lfnet():
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, use_bias=True, downsample=None):
            super(BasicBlock, self).__init__()
            self.bn0 = nn.BatchNorm2d(inplanes)
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride, padding=2, bias=use_bias)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=use_bias)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.bn0(x)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class DetectorModel(torch.nn.Module):
        def __init__(self, num_block=3, num_channels=16, conv_ksize=5,
                     use_bias=True, min_scale=2 ** -3, max_scale=1, num_scales=3):
            self.inplanes = num_channels
            self.num_blocks = num_block
            self.min_scale = min_scale
            self.max_scale = max_scale
            self.num_scales = num_scales
            super(DetectorModel, self).__init__()
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=conv_ksize, stride=1, padding=2,
                                   bias=use_bias)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.layer = BasicBlock(self.inplanes, self.inplanes, stride=1, use_bias=True)
            self.soft_conv = nn.Conv2d(16, 1, kernel_size=conv_ksize, stride=1, padding=2,
                                       bias=use_bias)
            self.ori_layer = nn.Conv2d(self.inplanes, 2, kernel_size=conv_ksize, stride=1, padding=2,
                                       bias=True)
            if self.num_scales == 1:
                self.scale_factors = [1.0]
            else:
                scale_log_factors = np.linspace(np.log(self.max_scale), np.log(self.min_scale), self.num_scales)
                self.scale_factors = np.exp(scale_log_factors)

        def forward(self, x):
            x = self.conv1(x)
            for i in range(self.num_blocks):
                x = self.layer(x)
            x = self.bn1(x)
            score_maps_list = []
            base_height_f = x.shape[2]
            base_width_f = x.shape[3]
            for i, s in enumerate(self.scale_factors):
                feat_height = (base_height_f * s + 0.5).astype(np.uint32)
                feat_width = (base_width_f * s + 0.5).astype(np.uint32)
                rs_feat_maps = torch.nn.functional.interpolate(x, [feat_height, feat_width])
                score_maps = self.soft_conv(rs_feat_maps)
                score_maps_list.append(score_maps)
            ori_maps = self.ori_layer(x)
            norm = ori_maps.norm(p=2, dim=1, keepdim=True)
            ori_maps = ori_maps.div(norm.expand_as(ori_maps))
            return score_maps_list,ori_maps

    class Descriptor(nn.Module):
        def __init__(self):
            super(Descriptor, self).__init__()
            nn.Sequential(

            )

        def forward(self, x):

            return out

    det = DetectorModel().cuda()
    des = Descriptor().cuda()

    cal_speed(det,des)

if __name__ == '__main__':
    cale
    if
    cal_lfnet()
