import torch
import sys

sys.path.extend(['/home/wang/workspace/FasterNet/NASNet_0.1'])

from model.det import RFDet
from model.des import HardNetNeiMask
from model.network import Network
from config import cfg
from hpatch_dataset import *

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Creating CNN model
det = RFDet(
    cfg.TRAIN.score_com_strength,
    cfg.TRAIN.scale_com_strength,
    cfg.TRAIN.NMS_THRESH,
    cfg.TRAIN.NMS_KSIZE,
    cfg.TRAIN.TOPK,
    cfg.MODEL.GAUSSIAN_KSIZE,
    cfg.MODEL.GAUSSIAN_SIGMA,
    cfg.MODEL.KSIZE,
    cfg.MODEL.padding,
    cfg.MODEL.dilation,
    cfg.MODEL.scale_list,
)
des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
model = Network(
    det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
)
model.cuda()

from thop import profile, clever_format

flop, para = profile(det, (torch.randn((1, 1, 640, 480), device=device),))
'''
flop
364512000.0
para
1377.0
'''
flop2, para2 = profile(des, (torch.randn((1000, 1, 32, 32), device=device),))
'''
flop2
16663040000.0
para2
334848.0
'''
clever_format([flop2, para2], "%.3f")

import math
import torch
import torch.nn as nn

def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)

def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return (x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W))

class IRFBlock(nn.Module):
    def __init__(self, input_depth, output_depth, expansion, stride, kernel=3, width_divisor=1, shuffle_type=None,
                 pw_group=1):
        super(IRFBlock, self).__init__()
        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth
        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)
        # pw
        self.pw = nn.Sequential(
            nn.Conv2d(input_depth, mid_depth, kernel_size=1, stride=1, padding=0, bias=not 1, groups=pw_group),
            nn.BatchNorm2d(mid_depth),
            nn.ReLU(inplace=True)
        )
        # dw
        self.dw = nn.Sequential(
            nn.Conv2d(mid_depth, mid_depth, kernel_size=kernel, stride=stride, padding=(kernel // 2), bias=not 1,
                      groups=mid_depth),
            nn.BatchNorm2d(mid_depth),
            nn.ReLU(inplace=True)
        )
        # pw-linear
        self.pwl = nn.Sequential(
            nn.Conv2d(mid_depth, output_depth, kernel_size=1, stride=1, padding=0, bias=not 1, groups=pw_group),
            nn.BatchNorm2d(output_depth),
            nn.ReLU(inplace=True)
        )
        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        return y


class HardNetNeiMask(nn.Module):
    def __init__(self):
        super(HardNetNeiMask, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            #######################################################################################################
            ## Stage 1: Identity(32,32,2)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=0, groups=1, bias=not 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ## Stage 2: Identity(32,32,1)
            # pass
            ## Stage 3: Identity(32,64,2)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0, groups=1, bias=not 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ## Stage 4: IRF(64,64,exp = 1,stride = 1,ksize = 5)
            IRFBlock(64, 64, 1, 1, kernel=5),
            ## Stage 5:
            IRFBlock(64, 128, 3, 2, kernel=3),
            ## Stage 6:
            IRFBlock(128, 128, 1, 1, kernel=5, shuffle_type="mid", pw_group=2),
            #######################################################################################################
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature
