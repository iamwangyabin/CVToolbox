from thop import profile, clever_format
from torchscope import scope
import torch
import torch.nn as nn
import torch.nn.functional as F


def L2Norm(input, dim=-1):
    input = input / torch.norm(input, p=2, dim=dim, keepdim=True)
    return input


class RFDet(nn.Module):
    def __init__(self):
        super(RFDet, self).__init__()
        ksize = 3
        padding = 1
        dilation = 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 3 RF
        self.insnorm1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s3 = nn.InstanceNorm2d(1, affine=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 5 RF
        self.insnorm2 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s5 = nn.InstanceNorm2d(1, affine=True)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 7 RF
        self.insnorm3 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s7 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s7 = nn.InstanceNorm2d(1, affine=True)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 9 RF
        self.insnorm4 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s9 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s9 = nn.InstanceNorm2d(1, affine=True)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 11 RF
        self.insnorm5 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s11 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s11 = nn.InstanceNorm2d(1, affine=True)

        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 13 RF
        self.insnorm6 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s13 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s13 = nn.InstanceNorm2d(1, affine=True)

        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 15 RF
        self.insnorm7 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s15 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s15 = nn.InstanceNorm2d(1, affine=True)

        self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 17 RF
        self.insnorm8 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s17 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s17 = nn.InstanceNorm2d(1, affine=True)

        self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation)  # 19 RF
        self.insnorm9 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s19 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s19 = nn.InstanceNorm2d(1, affine=True)

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                                dilation=dilation)  # 21 RF
        self.insnorm10 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s21 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s21 = nn.InstanceNorm2d(1, affine=True)

        self.conv_o3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o7 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o9 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o11 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o13 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o15 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o17 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o19 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o21 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.scale_list = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0])

    def forward(self, photos):
        # Extract score map in scale space from 3 to 21
        score_featmaps_s3 = F.leaky_relu(self.insnorm1(self.conv1(photos)))
        score_map_s3 = self.insnorm_s3(self.conv_s3(score_featmaps_s3)).permute(
            0, 2, 3, 1
        )
        orint_map_s3 = (
            L2Norm(self.conv_o3(score_featmaps_s3), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        score_featmaps_s5 = F.leaky_relu(self.insnorm2(self.conv2(score_featmaps_s3)))
        score_map_s5 = self.insnorm_s5(self.conv_s5(score_featmaps_s5)).permute(
            0, 2, 3, 1
        )
        orint_map_s5 = (
            L2Norm(self.conv_o5(score_featmaps_s5), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s5 = score_featmaps_s5 + score_featmaps_s3

        score_featmaps_s7 = F.leaky_relu(self.insnorm3(self.conv3(score_featmaps_s5)))
        score_map_s7 = self.insnorm_s7(self.conv_s7(score_featmaps_s7)).permute(
            0, 2, 3, 1
        )
        orint_map_s7 = (
            L2Norm(self.conv_o7(score_featmaps_s7), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s7 = score_featmaps_s7 + score_featmaps_s5

        score_featmaps_s9 = F.leaky_relu(self.insnorm4(self.conv4(score_featmaps_s7)))
        score_map_s9 = self.insnorm_s9(self.conv_s9(score_featmaps_s9)).permute(
            0, 2, 3, 1
        )
        orint_map_s9 = (
            L2Norm(self.conv_o9(score_featmaps_s9), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s9 = score_featmaps_s9 + score_featmaps_s7

        score_featmaps_s11 = F.leaky_relu(self.insnorm5(self.conv5(score_featmaps_s9)))
        score_map_s11 = self.insnorm_s11(self.conv_s11(score_featmaps_s11)).permute(
            0, 2, 3, 1
        )
        orint_map_s11 = (
            L2Norm(self.conv_o11(score_featmaps_s11), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s11 = score_featmaps_s11 + score_featmaps_s9

        score_featmaps_s13 = F.leaky_relu(self.insnorm6(self.conv6(score_featmaps_s11)))
        score_map_s13 = self.insnorm_s13(self.conv_s13(score_featmaps_s13)).permute(
            0, 2, 3, 1
        )
        orint_map_s13 = (
            L2Norm(self.conv_o13(score_featmaps_s13), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s13 = score_featmaps_s13 + score_featmaps_s11

        score_featmaps_s15 = F.leaky_relu(self.insnorm7(self.conv7(score_featmaps_s13)))
        score_map_s15 = self.insnorm_s15(self.conv_s15(score_featmaps_s15)).permute(
            0, 2, 3, 1
        )
        orint_map_s15 = (
            L2Norm(self.conv_o15(score_featmaps_s15), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s15 = score_featmaps_s15 + score_featmaps_s13

        score_featmaps_s17 = F.leaky_relu(self.insnorm8(self.conv8(score_featmaps_s15)))
        score_map_s17 = self.insnorm_s17(self.conv_s17(score_featmaps_s17)).permute(
            0, 2, 3, 1
        )
        orint_map_s17 = (
            L2Norm(self.conv_o17(score_featmaps_s17), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s17 = score_featmaps_s17 + score_featmaps_s15

        score_featmaps_s19 = F.leaky_relu(self.insnorm9(self.conv9(score_featmaps_s17)))
        score_map_s19 = self.insnorm_s19(self.conv_s19(score_featmaps_s19)).permute(
            0, 2, 3, 1
        )
        orint_map_s19 = (
            L2Norm(self.conv_o19(score_featmaps_s19), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s19 = score_featmaps_s19 + score_featmaps_s17

        score_featmaps_s21 = F.leaky_relu(
            self.insnorm10(self.conv10(score_featmaps_s19))
        )
        score_map_s21 = self.insnorm_s21(self.conv_s21(score_featmaps_s21)).permute(
            0, 2, 3, 1
        )
        orint_map_s21 = (
            L2Norm(self.conv_o21(score_featmaps_s21), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        score_maps = torch.cat(
            (
                score_map_s3,
                score_map_s5,
                score_map_s7,
                score_map_s9,
                score_map_s11,
                score_map_s13,
                score_map_s15,
                score_map_s17,
                score_map_s19,
                score_map_s21,
            ),
            -1,
        )  # (B, H, W, C)

        orint_maps = torch.cat(
            (
                orint_map_s3,
                orint_map_s5,
                orint_map_s7,
                orint_map_s9,
                orint_map_s11,
                orint_map_s13,
                orint_map_s15,
                orint_map_s17,
                orint_map_s19,
                orint_map_s21,
            ),
            -2,
        )  # (B, H, W, 10, 2)
        return score_maps, orint_maps


class L2Net(nn.Module):
    def __init__(self):
        super(L2Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def input_norm(x):
        eps = 1e-8
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + eps
        return (
                       x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
               ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature


import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, use_bias=True):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride, padding=2, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class LFDet(torch.nn.Module):
    def __init__(self, num_block=3, num_channels=16, conv_ksize=5,
                 use_bias=True, min_scale=2 ** -3, max_scale=1, num_scales=3):
        self.inplanes = num_channels
        self.num_blocks = num_block
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_scales = num_scales
        super(LFDet, self).__init__()
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
            print(i)
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

        endpoints = {}
        endpoints['ori_maps'] = ori_maps
        endpoints['scale_factors'] = self.scale_factors
        return score_maps_list, endpoints


class LFDes(nn.Module):
    def __init__(self):
        super(LFDes, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 16, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


from operations import IRFBlock


class OurAdet(nn.Module):
    def __init__(self):
        super(OurAdet, self).__init__()
        self.features = []
        self.features.append(IRFBlock(input_depth=1, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features.append(IRFBlock(input_depth=16, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features = nn.Sequential(*self.features)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.insnorm = nn.InstanceNorm2d(1, affine=True)
        self.conv_ori = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.insnorm_ori = nn.InstanceNorm2d(2, affine=True)

    def forward(self, photos):
        H, W = photos.shape[2:4]
        feature_map = self.features(photos)
        ori_map = self.insnorm_ori(
            self.conv_ori(torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)(feature_map)))
        ori_map = L2Norm(torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(ori_map), dim=1).permute(0,2,3,1)
        feature_map = F.leaky_relu(self.insnorm(self.conv(feature_map)))
        score_maps = torch.cat((feature_map,), 1, )
        score_maps = score_maps.permute(0, 2, 3, 1)
        return score_maps


class OurAdes(nn.Module):
    def __init__(self):
        super(OurAdes, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            IRFBlock(input_depth=32, output_depth=64, expansion=1, stride=1, kernel=3, se=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            IRFBlock(input_depth=64, output_depth=128, expansion=1, stride=1, kernel=3, se=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def input_norm(x):
        eps = 1e-8
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + eps
        return (
                       x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
               ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature


class OurBdet(nn.Module):
    def __init__(self):
        super(OurBdet, self).__init__()
        self.features = []
        self.features.append(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        self.features.append(IRFBlock(input_depth=16, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features.append(IRFBlock(input_depth=16, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features = nn.Sequential(*self.features)
        self.conv_scale = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.insnorm_scale = nn.InstanceNorm2d(1, affine=True)
        self.conv_ori = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.insnorm_ori = nn.InstanceNorm2d(2, affine=True)

    def forward(self, photos):
        H, W = photos.shape[2:4]
        feature_map = self.features(photos)
        feature_map_2 = torch.nn.Upsample(scale_factor=0.75, mode='bilinear', align_corners=True)(feature_map)
        feature_map_3 = torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)(feature_map)
        feature_map_1 = F.leaky_relu(self.insnorm_scale(self.conv_scale(feature_map)))
        feature_map_2 = F.leaky_relu(self.insnorm_scale(self.conv_scale(feature_map_2)))
        feature_map_3 = F.leaky_relu(self.insnorm_scale(self.conv_scale(feature_map_3)))
        ori_map = self.insnorm_ori(
            self.conv_ori(torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)(feature_map)))
        ori_map = L2Norm(torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(ori_map), dim=1).permute(0,2,3,1)
        score_maps = torch.cat(
            (
                feature_map_1,
                torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(feature_map_2),
                torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(feature_map_3),
            ),
            1,
        )  # (B, C, H, W)
        score_maps = score_maps.permute(0, 2, 3, 1)
        return ori_map


class OurBdes(nn.Module):
    def __init__(self):
        super(OurBdes, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            IRFBlock(input_depth=32, output_depth=64, expansion=1, stride=1, kernel=3, se=False),
            IRFBlock(input_depth=64, output_depth=64, expansion=1, stride=2, kernel=3, se=False),
            IRFBlock(input_depth=64, output_depth=128, expansion=1, stride=1, kernel=3, se=False),
            IRFBlock(input_depth=128, output_depth=128, expansion=1, stride=2, kernel=3, se=False),
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def input_norm(x):
        eps = 1e-8
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + eps
        return (
                       x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
               ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature


class OurCdet(nn.Module):
    def __init__(self):
        super(OurCdet, self).__init__()
        self.features = []
        self.features.append(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        self.features.append(IRFBlock(input_depth=16, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features.append(IRFBlock(input_depth=16, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features.append(IRFBlock(input_depth=16, output_depth=16, expansion=1, stride=1, kernel=3, se=False))
        self.features = nn.Sequential(*self.features)
        self.conv_scale = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.insnorm_scale = nn.InstanceNorm2d(1, affine=True)
        self.conv_ori = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.insnorm_ori = nn.InstanceNorm2d(2, affine=True)
        self.scale_list = torch.tensor([16.0, 32.0, 64.0])

    def forward(self, photos):
        H, W = photos.shape[2:4]
        feature_map = self.features(photos)
        feature_map_2 = torch.nn.Upsample(scale_factor=0.75, mode='bilinear', align_corners=True)(feature_map)
        feature_map_3 = torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)(feature_map)
        feature_map_1 = F.leaky_relu(self.insnorm_scale(self.conv_scale(feature_map)))
        feature_map_2 = F.leaky_relu(self.insnorm_scale(self.conv_scale(feature_map_2)))
        feature_map_3 = F.leaky_relu(self.insnorm_scale(self.conv_scale(feature_map_3)))
        ori_map = self.insnorm_ori(
            self.conv_ori(torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)(feature_map)))
        ori_map = L2Norm(torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(ori_map), dim=1).permute(0,2,3,1)
        score_maps = torch.cat(
            (
                feature_map_1,
                torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(feature_map_2),
                torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(feature_map_3),
            ),
            1,
        )
        score_maps = score_maps.permute(0, 2, 3, 1)
        return ori_map


class OurCdes(nn.Module):
    def __init__(self):
        super(OurCdes, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            IRFBlock(input_depth=32, output_depth=64, expansion=1, stride=1, kernel=1, se=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            IRFBlock(input_depth=64, output_depth=64, expansion=1, stride=1, kernel=1, se=False),
            IRFBlock(input_depth=64, output_depth=128, expansion=1, stride=1, kernel=3, se=False),
            IRFBlock(input_depth=128, output_depth=128, expansion=1, stride=1, kernel=3, se=False),
            IRFBlock(input_depth=128, output_depth=128, expansion=1, stride=2, kernel=3, se=False),
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def input_norm(x):
        eps = 1e-8
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + eps
        return (
                       x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
               ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature


lfdet = LFDet()#.cuda()
lfdes = LFDes()#.cuda()
rfdet = RFDet()#.cuda()
rfdes = L2Net()#.cuda()
l2net = L2Net()#.cuda()
print("lfdet")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(lfdet, input_size=(3, 640, 480))
print("lfdes")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(lfdes, input_size=(1, 32, 32))
print("rfdet")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(rfdet, input_size=(1, 640, 480))
print("rfdes")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(rfdes, input_size=(1, 32, 32))
print("l2net")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(l2net, input_size=(1, 32, 32))

# lfdetflop, lfdetpara = profile(lfdet, (torch.randn(1, 3, 640, 480).cuda(),))
# lfdesflop, lfdespara = profile(lfdes, (torch.randn(1000, 1, 32, 32).cuda(),))
# rfdetflop, rfdetpara = profile(rfdet, (torch.randn(1, 1, 640, 480).cuda(),))
# rfdesflop, rfdespara = profile(rfdes, (torch.randn(1000, 1, 32, 32).cuda(),))
# l2netflop, l2netpara = profile(l2net, (torch.randn(1000, 1, 32, 32).cuda(),))
#
# print("lfdet\t" + str(lfdetflop) + "\t||\t" + str(lfdetpara))
# print("lfdes\t" + str(lfdesflop) + "\t||\t" + str(lfdespara))
# print("rfdet\t" + str(rfdetflop) + "\t||\t" + str(rfdetpara))
# print("rfdes\t" + str(rfdesflop) + "\t||\t" + str(rfdespara))
# print("l2net\t" + str(l2netflop) + "\t||\t" + str(l2netpara))

Adet = OurAdet()#.cuda()
Ades = OurAdes()#.cuda()
Bdet = OurBdet()#.cuda()
Bdes = OurBdes()#.cuda()
Cdet = OurCdet()#.cuda()
Cdes = OurCdes()#.cuda()
print("A")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(Adet, input_size=(1, 640, 480))
scope(Ades, input_size=(1, 32, 32))
print("B")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(Bdet, input_size=(1, 640, 480))
scope(Bdes, input_size=(1, 32, 32))
print("C")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
scope(Cdet, input_size=(1, 640, 480))
scope(Cdes, input_size=(1, 32, 32))


# Adetflop, Adetpara = profile(Adet, (torch.randn(1, 1, 640, 480) ,))                       #.cuda()
# Adesflop, Adespara = profile(Ades, (torch.randn(1000, 1, 32, 32),))                       #.cuda()
# Bdetflop, Bdetpara = profile(Bdet, (torch.randn(1, 1, 640, 480) ,))                       #.cuda()
# Bdesflop, Bdespara = profile(Bdes, (torch.randn(1000, 1, 32, 32),))                       #.cuda()
# Cdetflop, Cdetpara = profile(Cdet, (torch.randn(1, 1, 640, 480) ,))                       #.cuda()
# Cdesflop, Cdespara = profile(Cdes, (torch.randn(1000, 1, 32, 32),))                       #.cuda()
#
# print("Adet\t" + str(Adetflop) + "\t||\t" + str(Adetpara))
# print("Ades\t" + str(Adesflop) + "\t||\t" + str(Adespara))
# print("Bdet\t" + str(Bdetflop) + "\t||\t" + str(Bdetpara))
# print("Bdes\t" + str(Bdesflop) + "\t||\t" + str(Bdespara))
# print("Cdet\t" + str(Cdetflop) + "\t||\t" + str(Cdetpara))
# print("Cdet\t" + str(Cdesflop) + "\t||\t" + str(Cdespara))




# import time
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
#
# def cal_speed_lf(det, des):
#     with torch.no_grad():
#         input_sample = torch.randn((1, 3, 640, 480), device=device)
#         total_time = 0
#         for i in range(100):
#             torch.cuda.synchronize()
#             time0 = time.time()
#             det(input_sample)
#             torch.cuda.synchronize()
#             time1 = time.time()
#             total_time += (time1 - time0)
#         print(total_time / 1000)
#         patch_sample = torch.randn((1000, 1, 32, 32), device=device)
#         total_time = 0
#         for i in range(100):
#             torch.cuda.synchronize()
#             time0 = time.time()
#             des(patch_sample)
#             torch.cuda.synchronize()
#             time1 = time.time()
#             total_time += (time1 - time0)
#         print(total_time / 1000)
#
#
# def cal_speed(det, des):
#     with torch.no_grad():
#         input_sample = torch.randn((1, 1, 640, 480), device=device)
#         total_time = 0
#         for i in range(100):
#             torch.cuda.synchronize()
#             time0 = time.time()
#             det(input_sample)
#             torch.cuda.synchronize()
#             time1 = time.time()
#             total_time += (time1 - time0)
#         print(total_time / 1000)
#         patch_sample = torch.randn((1000, 1, 32, 32), device=device)
#         total_time = 0
#         for i in range(100):
#             torch.cuda.synchronize()
#             time0 = time.time()
#             des(patch_sample)
#             torch.cuda.synchronize()
#             time1 = time.time()
#             total_time += (time1 - time0)
#         print(total_time / 1000)
#
# cal_speed_lf(lfdet, lfdes)
# cal_speed(rfdet, rfdes)
# cal_speed(rfdes, l2net)
# cal_speed(Adet, Ades)
# cal_speed(Bdet, Bdes)
# cal_speed(Cdet, Cdes)

# 0.003097022533416748
# 0.00046321439743041994
# 0.004447117328643799
# 0.0016262781620025634
# 0.0011806464195251464
# 0.001807570457458496
# 0.0014498653411865235
# 0.0008102757930755616
# 0.002014974355697632
# 0.0010907046794891358
# 0.00262319016456604
# 0.0012487895488739013

