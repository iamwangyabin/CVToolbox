import torch
from torch import nn
from torch.nn import functional as f
import numpy as np

# build resnet blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, use_bias=True,downsample=None):
        super(BasicBlock, self).__init__()
        self.bn0=nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,padding=2, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride,padding=2, bias=use_bias)
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
    def __init__(self, num_block=3, num_channels=16,conv_ksize=5,
                 use_bias=True, min_scale=2**-3, max_scale=1, num_scales=3):

        self.inplanes = num_channels
        self.num_blocks=num_block
        self.min_scale = min_scale
        self.max_scale=max_scale
        self.num_scales=num_scales

        super(DetectorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=conv_ksize, stride=1, padding=2,
                               bias=use_bias)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer=BasicBlock(self.inplanes, self.inplanes, stride=1, use_bias=True)
        self.soft_conv=nn.Conv2d(16, 1, kernel_size=conv_ksize, stride=1, padding=2,
                               bias=use_bias)
        self.ori_layer=nn.Conv2d(self.inplanes,2,kernel_size=conv_ksize, stride=1, padding=2,
                                bias=True )
#         ori_b_init=torch.nn.init.constant(np.array([1,0], dtype=np.float32))
#         self.ori_layer.bias.data.fill_(ori_b_init)
        if self.num_scales == 1:
            self.scale_factors = [1.0]
        else:
            scale_log_factors = np.linspace(np.log(self.max_scale), np.log(self.min_scale), self.num_scales)
            self.scale_factors = np.exp(scale_log_factors)
        
    def forward(self, x):
        x=self.conv1(x)
        for i in range(self.num_blocks):
            x=self.layer(x)
            print(i)
        x=self.bn1(x)
        score_maps_list = []
        base_height_f = x.shape[2]
        base_width_f = x.shape[3]
        for i, s in enumerate(self.scale_factors):
            feat_height = (base_height_f * s + 0.5).astype(np.uint32)
            feat_width = (base_width_f * s + 0.5).astype(np.uint32)
            rs_feat_maps=torch.nn.functional.interpolate(x,[feat_height, feat_width])
            score_maps = self.soft_conv(rs_feat_maps)
            score_maps_list.append(score_maps)
#         ori_b_init=torch.nn.init.constant(np.array([1,0], dtype=np.float32))
#         self.ori_layer.bias.data.fill_(ori_b_init)
        ori_maps=self.ori_layer(x)
        norm = ori_maps.norm(p=2, dim=1, keepdim=True)
        ori_maps = ori_maps.div(norm.expand_as(ori_maps))
    
        endpoints={}
        endpoints['ori_maps'] = ori_maps
        endpoints['scale_factors'] = self.scale_factors
        return score_maps_list,endpoints
