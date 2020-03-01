from thop import profile, clever_format

import torch
import sys
sys.path.extend(['/home/wang/workspace/RFSLAM_offline/RFNET/rfnet/'])

from config import cfg
from model.rf_des import HardNetNeiMask
from model.rf_det_so import RFDetSO
from model.rf_net_so import RFNetSO

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

det = RFDetSO(cfg.TRAIN.score_com_strength, cfg.TRAIN.scale_com_strength, cfg.TRAIN.NMS_THRESH, cfg.TRAIN.NMS_KSIZE,
              1024, cfg.MODEL.GAUSSIAN_KSIZE, cfg.MODEL.GAUSSIAN_SIGMA, cfg.MODEL.KSIZE, cfg.MODEL.padding,
              cfg.MODEL.dilation, cfg.MODEL.scale_list,
              )
des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
model = RFNetSO(
    det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, 512,
)
model = model.to(device)


flop, para = profile(det, (torch.randn((1, 1, 640, 480),device = device), ))

'''
flop
6829056000.0 
para
Out[4]: 21890.0
'''
flop2, para2 = profile(des, (torch.randn((1000, 1, 32, 32),device = device), ))
'''
flop2
39551488.0
39551488000.0
para2
1334560.0
'''
clever_format([flop2, para2], "%.3f")


