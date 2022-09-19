import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_utils import filter_border, nms, topk_map, get_gauss_filter_weight, soft_nms_3d, soft_max_and_argmax_1d
from utils.math_utils import MSD, distance_matrix_vector, L2Norm


class RFDet(nn.Module):
    def __init__(
            self,
            score_com_strength,
            scale_com_strength,
            nms_thresh,
            nms_ksize,
            topk,
            gauss_ksize,
            gauss_sigma,
            ksize,
            padding,
            dilation,
            scale_list,
            SCORE_W,
            PAIR_W,
    ):
        super(RFDet, self).__init__()

        self.score_com_strength = score_com_strength
        self.scale_com_strength = scale_com_strength
        self.NMS_THRESH = nms_thresh
        self.NMS_KSIZE = nms_ksize
        self.TOPK = topk
        self.GAUSSIAN_KSIZE = gauss_ksize
        self.GAUSSIAN_SIGMA = gauss_sigma
        self.SCORE_W = SCORE_W
        self.PAIR_W = PAIR_W

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 3 RF
        self.insnorm1 = nn.InstanceNorm2d(16, affine=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 5 RF
        self.insnorm2 = nn.InstanceNorm2d(16, affine=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 7 RF
        self.insnorm3 = nn.InstanceNorm2d(16, affine=True)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 9 RF
        self.insnorm4 = nn.InstanceNorm2d(16, affine=True)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 11 RF
        self.insnorm5 = nn.InstanceNorm2d(16, affine=True)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 13 RF
        self.insnorm6 = nn.InstanceNorm2d(16, affine=True)
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 15 RF
        self.insnorm7 = nn.InstanceNorm2d(16, affine=True)
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 17 RF
        self.insnorm8 = nn.InstanceNorm2d(16, affine=True)
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                               dilation=dilation, )  # 19 RF
        self.insnorm9 = nn.InstanceNorm2d(16, affine=True)
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ksize, stride=1, padding=padding,
                                dilation=dilation, )  # 21 RF
        self.insnorm10 = nn.InstanceNorm2d(16, affine=True)

        self.conv_s3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s3 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s5 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s7 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s7 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s9 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s9 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s11 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s11 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s13 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s13 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s15 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s15 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s17 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s17 = nn.InstanceNorm2d(1, affine=True)
        self.conv_s19 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s19 = nn.InstanceNorm2d(1, affine=True)
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

        self.apply(self.weights_init)
        self.conv_o3.apply(self.convO_init)
        self.conv_o5.apply(self.convO_init)
        self.conv_o7.apply(self.convO_init)
        self.conv_o9.apply(self.convO_init)
        self.conv_o11.apply(self.convO_init)
        self.conv_o13.apply(self.convO_init)
        self.conv_o15.apply(self.convO_init)
        self.conv_o17.apply(self.convO_init)
        self.conv_o19.apply(self.convO_init)
        self.conv_o21.apply(self.convO_init)

        self.scale_list = torch.tensor(scale_list)

    def forward(self, photos):
        score_featmaps_s3 = F.leaky_relu(self.insnorm1(self.conv1(photos)))
        score_map_s3 = self.insnorm_s3(self.conv_s3(score_featmaps_s3)).permute(0, 2, 3, 1)
        orint_map_s3 = self.L2Norm(self.conv_o3(score_featmaps_s3)).permute(0, 2, 3, 1).unsqueeze(-2)

        score_featmaps_s5 = F.leaky_relu(self.insnorm2(self.conv2(score_featmaps_s3)))
        score_map_s5 = self.insnorm_s5(self.conv_s5(score_featmaps_s5)).permute(0, 2, 3, 1)

        orint_map_s5 = self.L2Norm(self.conv_o5(score_featmaps_s5)).permute(0, 2, 3, 1).unsqueeze(-2)
        score_featmaps_s5 = score_featmaps_s5 + score_featmaps_s3

        score_featmaps_s7 = F.leaky_relu(self.insnorm3(self.conv3(score_featmaps_s5)))
        score_map_s7 = self.insnorm_s7(self.conv_s7(score_featmaps_s7)).permute(0, 2, 3, 1)
        orint_map_s7 = self.L2Norm(self.conv_o7(score_featmaps_s7)).permute(0, 2, 3, 1).unsqueeze(-2)

        score_featmaps_s7 = score_featmaps_s7 + score_featmaps_s5

        score_featmaps_s9 = F.leaky_relu(self.insnorm4(self.conv4(score_featmaps_s7)))
        score_map_s9 = self.insnorm_s9(self.conv_s9(score_featmaps_s9)).permute(0, 2, 3, 1)
        orint_map_s9 = self.L2Norm(self.conv_o9(score_featmaps_s9)).permute(0, 2, 3, 1).unsqueeze(-2)

        score_featmaps_s9 = score_featmaps_s9 + score_featmaps_s7

        score_featmaps_s11 = F.leaky_relu(self.insnorm5(self.conv5(score_featmaps_s9)))
        score_map_s11 = self.insnorm_s11(self.conv_s11(score_featmaps_s11)).permute(0, 2, 3, 1)
        orint_map_s11 = self.L2Norm(self.conv_o11(score_featmaps_s11)).permute(0, 2, 3, 1).unsqueeze(-2)
        score_featmaps_s11 = score_featmaps_s11 + score_featmaps_s9

        score_featmaps_s13 = F.leaky_relu(self.insnorm6(self.conv6(score_featmaps_s11)))
        score_map_s13 = self.insnorm_s13(self.conv_s13(score_featmaps_s13)).permute(0, 2, 3, 1)
        orint_map_s13 =  self.L2Norm(self.conv_o13(score_featmaps_s13)).permute(0, 2, 3, 1).unsqueeze(-2)
        score_featmaps_s13 = score_featmaps_s13 + score_featmaps_s11

        score_featmaps_s15 = F.leaky_relu(self.insnorm7(self.conv7(score_featmaps_s13)))
        score_map_s15 = self.insnorm_s15(self.conv_s15(score_featmaps_s15)).permute(0, 2, 3, 1 )
        orint_map_s15 = self.L2Norm(self.conv_o15(score_featmaps_s15)).permute(0, 2, 3, 1).unsqueeze(-2)

        score_featmaps_s15 = score_featmaps_s15 + score_featmaps_s13

        score_featmaps_s17 = F.leaky_relu(self.insnorm8(self.conv8(score_featmaps_s15)))
        score_map_s17 = self.insnorm_s17(self.conv_s17(score_featmaps_s17)).permute(0, 2, 3, 1)
        orint_map_s17 = self.L2Norm(self.conv_o17(score_featmaps_s17)).permute(0, 2, 3, 1).unsqueeze(-2)
        score_featmaps_s17 = score_featmaps_s17 + score_featmaps_s15

        score_featmaps_s19 = F.leaky_relu(self.insnorm9(self.conv9(score_featmaps_s17)))
        score_map_s19 = self.insnorm_s19(self.conv_s19(score_featmaps_s19)).permute(0, 2, 3, 1 )
        orint_map_s19 = self.L2Norm(self.conv_o19(score_featmaps_s19)).permute(0, 2, 3, 1).unsqueeze(-2)
        score_featmaps_s19 = score_featmaps_s19 + score_featmaps_s17

        score_featmaps_s21 = F.leaky_relu(self.insnorm10(self.conv10(score_featmaps_s19)) )
        score_map_s21 = self.insnorm_s21(self.conv_s21(score_featmaps_s21)).permute(0, 2, 3, 1)
        orint_map_s21 = self.L2Norm(self.conv_o21(score_featmaps_s21)).permute(0, 2, 3, 1).unsqueeze(-2)

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

    def process(self, im1w_score):
        """
        nms(n), topk(t), gaussian kernel(g) operation
        :param im1w_score: warped score map
        :return: processed score map, topk mask, topk value
        """
        im1w_score = filter_border(im1w_score)

        # apply nms to im1w_score
        nms_mask = nms(im1w_score, thresh=self.NMS_THRESH, ksize=self.NMS_KSIZE)
        im1w_score = im1w_score * nms_mask
        topk_value = im1w_score

        # apply topk to im1w_score
        topk_mask = topk_map(im1w_score, self.TOPK)
        im1w_score = topk_mask.to(torch.float) * im1w_score

        # apply gaussian kernel to im1w_score
        psf = im1w_score.new_tensor(
            get_gauss_filter_weight(self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIGMA)[
            None, None, :, :
            ]
        )
        im1w_score = F.conv2d(
            input=im1w_score.permute(0, 3, 1, 2),
            weight=psf,
            stride=1,
            padding=self.GAUSSIAN_KSIZE // 2,
        ).permute(
            0, 2, 3, 1
        )  # (B, H, W, 1)

        """
        apply tf.clamp to make sure all value in im1w_score isn't greater than 1
        but this won't happend in correct way
        """
        im1w_score = im1w_score.clamp(min=0.0, max=1.0)

        return im1w_score, topk_mask, topk_value

    @staticmethod
    def loss(left_score, im1gt_score, im1visible_mask):
        im1_score = left_score

        l2_element_diff = (im1_score - im1gt_score) ** 2
        # visualization numbers
        Nvi = torch.clamp(im1visible_mask.sum(dim=(3, 2, 1)), min=2.0)
        loss = (
                torch.sum(l2_element_diff * im1visible_mask, dim=(3, 2, 1)) / (Nvi + 1e-8)
        ).mean()

        return loss

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(
                m.weight.data, gain=nn.init.calculate_gain("leaky_relu")
            )
            try:
                nn.init.xavier_uniform_(m.bias.data)
            except:
                pass

    @staticmethod
    @torch.jit.script
    def L2Norm(input):
        norm = input.pow(2).sum(dim=1, keepdim=True).sqrt()
        input = input / norm
        return input

    @staticmethod
    def convO_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight.data)
            try:
                nn.init.ones_(m.bias.data)
            except:
                pass

    def criterion(self, endpoint):
        im1_score = endpoint["im1_score"]
        im1_gtsc = endpoint["im1_gtsc"]
        im1_visible = endpoint["im1_visible"]
        im2_score = endpoint["im2_score"]
        im2_gtsc = endpoint["im2_gtsc"]
        im2_visible = endpoint["im2_visible"]
        im1_lpreddes = endpoint["im1_lpreddes"]
        im1_rpreddes = endpoint["im1_rpreddes"]
        im2_lpreddes = endpoint["im2_lpreddes"]
        im2_rpreddes = endpoint["im2_rpreddes"]

        im1_scloss = self.loss(im1_score, im1_gtsc, im1_visible)
        im2_scloss = self.loss(im2_score, im2_gtsc, im2_visible)
        score_loss = (im1_scloss + im2_scloss) / 2.0 * self.SCORE_W

        im1_pairloss = distance_matrix_vector(im1_lpreddes, im1_rpreddes).diag().mean()
        im2_pairloss = distance_matrix_vector(im2_lpreddes, im2_rpreddes).diag().mean()
        pair_loss = (im1_pairloss + im2_pairloss) / 2.0 * self.PAIR_W
        det_loss = score_loss + pair_loss
        return det_loss.mean()
