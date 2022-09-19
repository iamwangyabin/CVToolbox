import os
import time
import torch
import random
import argparse
import numpy as np
from torch import autograd
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from hpatch_dataset import HpatchDataset, Grayscale, Normalize, Rescale, LargerRescale, RandomCrop, ToTensor
from utils.eval_utils import eval_model, getAC
from utils.common_utils import gct, prettydict
from utils.train_utils import parse_batch, parse_unsqueeze, mgpu_merge, writer_log, ExponentialLR, SgdLR
from utils.math_utils import MSD, distance_matrix_vector, L2Norm
from utils.image_utils import warp, filter_border, soft_nms_3d, soft_max_and_argmax_1d
from utils.net_utils import pair

from utils.common_utils import imgBatchXYZ, transXYZ_2_to_1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def Lr_Schechuler(lr_schedule, optimizer, epoch, cfg):
    if lr_schedule == "exp":
        ExponentialLR(optimizer, epoch, cfg)
    elif lr_schedule == "sgd":
        SgdLR(optimizer, cfg)


def select_optimizer(optim, param, lr, wd):
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            param, lr=lr, momentum=0.9, dampening=0.9, weight_decay=wd
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(param, lr=lr, weight_decay=wd)
    else:
        raise Exception(f"Not supported optimizer: {optim}")
    return optimizer


def create_optimizer(optim, model, lr, wd, mgpu=False):
    if mgpu:
        param = model.module.parameters()
    else:
        param = model.parameters()
    optimizer = select_optimizer(optim, param, lr, wd)
    return optimizer


def gt_scale_orin(im2_scale, im2_orin, homo12, homo21):
    B, H, W, C = im2_scale.size()
    im2_cos, im2_sin = im2_orin.squeeze().chunk(chunks=2, dim=-1)  # (B, H, W, 1)
    # im2_tan = im2_sin / im2_cos

    # each centX, centY, centZ is (B, H, W, 1)
    centX, centY, centZ = imgBatchXYZ(B, H, W).to(im2_scale.device).chunk(3, dim=3)

    """get im1w scale maps"""
    half_scale = im2_scale // 2
    centXYZ = torch.cat((centX, centY, centZ), dim=3)  # (B, H, W, 3)
    upXYZ = torch.cat((centX, centY - half_scale, centZ), dim=3)
    bottomXYZ = torch.cat((centX, centY + half_scale, centZ), dim=3)
    rightXYZ = torch.cat((centX + half_scale, centY, centZ), dim=3)
    leftXYZ = torch.cat((centX - half_scale, centY, centZ), dim=3)

    centXYw = transXYZ_2_to_1(centXYZ, homo21)  # (B, H, W, 2) (x, y)
    centXw, centYw = centXYw.chunk(chunks=2, dim=-1)  # (B, H, W, 1)
    centXYw = centXYw.long()
    upXYw = transXYZ_2_to_1(upXYZ, homo21).long()
    rightXYw = transXYZ_2_to_1(rightXYZ, homo21).long()
    bottomXYw = transXYZ_2_to_1(bottomXYZ, homo21).long()
    leftXYw = transXYZ_2_to_1(leftXYZ, homo21).long()

    upScale = MSD(upXYw, centXYw)
    rightScale = MSD(rightXYw, centXYw)
    bottomScale = MSD(bottomXYw, centXYw)
    leftScale = MSD(leftXYw, centXYw)
    centScale = (upScale + rightScale + bottomScale + leftScale) / 4  # (B, H， W, 1)

    """get im1w orintation maps"""
    offset_x, offset_y = im2_scale * im2_cos, im2_scale * im2_sin  # (B, H, W, 1)
    offsetXYZ = torch.cat((centX + offset_x, centY + offset_y, centZ), dim=3)
    offsetXYw = transXYZ_2_to_1(offsetXYZ, homo21)  # (B, H, W, 2) (x, y)
    offsetXw, offsetYw = offsetXYw.chunk(chunks=2, dim=-1)  # (B, H, W, 1)
    offset_ww, offset_hw = offsetXw - centXw, offsetYw - centYw  # (B, H, W, 1)
    offset_rw = (offset_ww ** 2 + offset_hw ** 2 + 1e-8).sqrt()
    # tan = offset_hw / (offset_ww + 1e-8)  # (B, H, W, 1)
    cos_w = offset_ww / (offset_rw + 1e-8)  # (B, H, W, 1)
    sin_w = offset_hw / (offset_rw + 1e-8)  # (B, H, W, 1)
    # atan_w = np.arctan(tan.cpu().detach())  # (B, H, W, 1)

    # get left scale by transXYZ_2_to_1
    map_xy_2_to_1 = transXYZ_2_to_1(centXYZ, homo12).round().long()  # (B, H, W, 2)
    x, y = map_xy_2_to_1.chunk(2, dim=3)  # each x and y is (B, H, W, 1)
    x = x.clamp(min=0, max=W - 1)
    y = y.clamp(min=0, max=H - 1)

    # (B, H, W, 1)
    im1w_scale = centScale[
        torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
    ].view(im2_scale.size())

    # (B, H, W, 1, 2)
    im1w_cos = cos_w[
        torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
    ].view(im2_cos.size())
    im1w_sin = sin_w[
        torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
    ].view(im2_sin.size())
    im1w_orin = torch.cat((im1w_cos[:, None], im1w_sin[:, None]), dim=-1)
    im1w_orin = L2Norm(im1w_orin, dim=-1).to(im2_orin.device)

    return im1w_scale, im1w_orin

def get_all_endpoints(det, des, batch):
    im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = batch
    score_maps, orint_maps = det(im1_data)
    im1_rawsc, im1_scale, im1_orin = handle_det_out(score_maps, orint_maps, det.scale_list, det.score_com_strength,
                                                    det.scale_com_strength)
    score_maps, orint_maps = det(im2_data)
    im2_rawsc, im2_scale, im2_orin = handle_det_out(score_maps, orint_maps, det.scale_list, det.score_com_strength,
                                                    det.scale_com_strength)
    im1_gtscale, im1_gtorin = gt_scale_orin(im2_scale, im2_orin, homo12, homo21)
    im2_gtscale, im2_gtorin = gt_scale_orin(im1_scale, im1_orin, homo21, homo12)
    im2_score = filter_border(im2_rawsc)
    im1w_score = warp(im2_score, homo12)
    im1_visiblemask = warp(
        im2_score.new_full(im2_score.size(), fill_value=1, requires_grad=True),
        homo12, )
    im1_gtsc, im1_topkmask, im1_topkvalue = det.process(im1w_score)

    im1_score = filter_border(im1_rawsc)
    im2w_score = warp(im1_score, homo21)
    im2_visiblemask = warp(
        im2_score.new_full(im1_score.size(), fill_value=1, requires_grad=True),
        homo21, )
    im2_gtsc, im2_topkmask, im2_topkvalue = det.process(im2w_score)
    im1_score = det.process(im1_rawsc)[0]
    im2_score = det.process(im2_rawsc)[0]
    im1_ppair, im1_limc, im1_rimcw = pair(
        im1_topkmask,
        im1_topkvalue,
        im1_scale,
        im1_orin,
        im1_info,
        im1_raw,
        homo12,
        im2_gtscale,
        im2_gtorin,
        im2_info,
        im2_raw,
        cfg.PATCH.SIZE,
    )
    im2_ppair, im2_limc, im2_rimcw = pair(
        im2_topkmask,
        im2_topkvalue,
        im2_scale,
        im2_orin,
        im2_info,
        im2_raw,
        homo21,
        im1_gtscale,
        im1_gtorin,
        im1_info,
        im1_raw,
        cfg.PATCH.SIZE,
    )
    im1_lpatch, im1_rpatch = im1_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)
    im2_lpatch, im2_rpatch = im2_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)

    im1_lpatch = des.input_norm(im1_lpatch)
    im2_lpatch = des.input_norm(im2_lpatch)
    im1_rpatch = des.input_norm(im1_rpatch) 
    im2_rpatch = des.input_norm(im2_rpatch)
    im1_lpdes, im1_rpdes = des(im1_lpatch), des(im1_rpatch)
    im2_lpdes, im2_rpdes = des(im2_lpatch), des(im2_rpatch)
    im1_predpair, _, _ = pair(
        im1_topkmask,
        im1_topkvalue,
        im1_scale,
        im1_orin,
        im1_info,
        im1_raw,
        homo12,
        im2_scale,
        im2_orin,
        im2_info,
        im2_raw,
        cfg.PATCH.SIZE,
    )
    im2_predpair, _, _ = pair(
        im2_topkmask,
        im2_topkvalue,
        im2_scale,
        im2_orin,
        im2_info,
        im2_raw,
        homo21,
        im1_scale,
        im1_orin,
        im1_info,
        im1_raw,
        cfg.PATCH.SIZE,
    )
    # each is (N, 32, 32)
    im1_lpredpatch, im1_rpredpatch = im1_predpair.chunk(chunks=2, dim=1)
    im2_lpredpatch, im2_rpredpatch = im2_predpair.chunk(chunks=2, dim=1)
    im1_lpredpatch = des.input_norm(im1_lpredpatch)
    im2_lpredpatch = des.input_norm(im2_lpredpatch)
    im1_rpredpatch = des.input_norm(im1_rpredpatch)
    im2_rpredpatch = des.input_norm(im2_rpredpatch)

    im1_lpreddes, im1_rpreddes = des(im1_lpredpatch), des(im1_rpredpatch)
    im2_lpreddes, im2_rpreddes = des(im2_lpredpatch), des(im2_rpredpatch)
    endpoint = {
        "im1_score": im1_score,
        "im1_gtsc": im1_gtsc,
        "im1_visible": im1_visiblemask,
        "im2_score": im2_score,
        "im2_gtsc": im2_gtsc,
        "im2_visible": im2_visiblemask,
        "im1_lpreddes": im1_lpreddes,
        "im1_rpreddes": im1_rpreddes,
        "im2_lpreddes": im2_lpreddes,
        "im2_rpreddes": im2_rpreddes,
        "im1_limc": im1_limc,  #
        "im1_rimcw": im1_rimcw,  #
        "im2_limc": im2_limc,  #
        "im2_rimcw": im2_rimcw,  #
        "im1_lpdes": im1_lpdes,  #
        "im1_rpdes": im1_rpdes,  #
        "im2_lpdes": im2_lpdes,  #
        "im2_rpdes": im2_rpdes,  #
    }
    return endpoint

def handle_det_out(score_maps, orint_maps, scale_list, score_com_strength, scale_com_strength):
    scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=3.0)
    score_map, scale_map, orint_map = soft_max_and_argmax_1d(
        input=scale_probs,
        orint_maps=orint_maps,
        dim=-1,
        scale_list=scale_list,
        keepdim=True,
        com_strength1=score_com_strength,
        com_strength2=scale_com_strength,
    )
    return score_map, scale_map, orint_map

def parse_parms():
    parser = argparse.ArgumentParser(description="Test a DualDet Network")
    parser.add_argument(
        "--resume", default="", type=str, help="latest checkpoint (default: none)"
    )
    parser.add_argument(
        "--ver", default="", type=str, help="model version(defualt: none)"
    )
    parser.add_argument(
        "--save", default="", type=str, help="source code save path(defualt: none)"
    )
    parser.add_argument(
        "--det-step", default=1, type=int, help="train detection step(defualt: 1)"
    )
    parser.add_argument(
        "--des-step", default=2, type=int, help="train descriptor step(defualt: 2)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    from config import cfg
    from model.rf_det import RFDet
    from model.rf_des import HardNetNeiMask

    args = parse_parms()
    cfg.TRAIN.SAVE = args.save
    cfg.TRAIN.DET = args.det_step
    cfg.TRAIN.DES = args.des_step
    print(f"{gct()} : Called with args:{args}")
    print(f"{gct()} : Using config:")
    prettydict(cfg)

    ###############################################################################
    # Set the random seed manually for reproducibility
    ###############################################################################
    print(f"{gct()} : Prepare for repetition")
    device = torch.device("cuda" if cfg.PROJ.USE_GPU else "cpu")
    mgpu = True if cfg.PROJ.USE_GPU and torch.cuda.device_count() > 1 else False
    seed = cfg.PROJ.SEED
    if cfg.PROJ.USE_GPU:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
        if mgpu:
            print(f"{gct()} : Train with {torch.cuda.device_count()} GPUs")
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ###############################################################################
    # Build the model
    ###############################################################################
    print(f"{gct()} : Build the model")
    det = RFDet(cfg.TRAIN.score_com_strength,
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
                cfg.LOSS.SCORE,
                cfg.LOSS.PAIR,
                )
    des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)

    if mgpu:
        det = torch.nn.DataParallel(det)
        des = torch.nn.DataParallel(des)
    det = det.to(device=device)
    des = des.to(device=device)

    ###############################################################################
    # Load train data
    ###############################################################################
    PPT = [cfg.PROJ.TRAIN_PPT, (cfg.PROJ.TRAIN_PPT + cfg.PROJ.EVAL_PPT)]

    print(f"{gct()} : Loading traning data")
    train_data = DataLoader(
        HpatchDataset(
            data_type="train",
            PPT=PPT,
            use_all=cfg.PROJ.TRAIN_ALL,
            csv_file=cfg[cfg.PROJ.TRAIN]["csv"],
            root_dir=cfg[cfg.PROJ.TRAIN]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.TRAIN]["MEAN"], std=cfg[cfg.PROJ.TRAIN]["STD"]
                    ),
                    LargerRescale((960, 1280)),
                    RandomCrop((720, 960)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    ###############################################################################
    # Load evaluation data
    ###############################################################################
    print(f"{gct()} : Loading evaluation data")
    val_data = DataLoader(
        HpatchDataset(
            data_type="eval",
            PPT=PPT,
            use_all=cfg.PROJ.EVAL_ALL,
            csv_file=cfg[cfg.PROJ.EVAL]["csv"],
            root_dir=cfg[cfg.PROJ.EVAL]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.EVAL]["MEAN"], std=cfg[cfg.PROJ.EVAL]["STD"]
                    ),
                    Rescale((960, 1280)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    ###############################################################################
    # Load test data
    ###############################################################################
    print(f"{gct()} : Loading testing data")
    test_data = DataLoader(
        HpatchDataset(
            data_type="test",
            PPT=PPT,
            use_all=cfg.PROJ.TEST_ALL,
            csv_file=cfg[cfg.PROJ.TEST]["csv"],
            root_dir=cfg[cfg.PROJ.TEST]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.TEST]["MEAN"], std=cfg[cfg.PROJ.TEST]["STD"]
                    ),
                    Rescale((960, 1280)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    ###############################################################################
    # Build the optimizer
    ###############################################################################
    det_optim = create_optimizer(
        optim=cfg.TRAIN.DET_OPTIMIZER,
        model=det,
        lr=cfg.TRAIN.DET_LR,
        wd=cfg.TRAIN.DET_WD,
        mgpu=mgpu,
    )
    des_optim = create_optimizer(
        optim=cfg.TRAIN.DES_OPTIMIZER,
        model=des,
        lr=cfg.TRAIN.DES_LR,
        wd=cfg.TRAIN.DES_WD,
        mgpu=mgpu,
    )
    train_writer = SummaryWriter(f"{args.save}/log/train")
    test_writer = SummaryWriter(f"{args.save}/log/test")

    ###############################################################################
    # Training function
    ###############################################################################
    def train():
        start_time = time.time()
        for i_batch, sample_batched in enumerate(train_data, 1):
            det.train()
            des.train()
            batch = parse_batch(sample_batched, device)
            with autograd.detect_anomaly():
                for des_train in range(0, cfg.TRAIN.DES):
                    des.zero_grad()
                    des_optim.zero_grad()
                    im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = batch
                    score_maps,orint_maps = det(im1_data)
                    im1_rawsc, im1_scale, im1_orin = handle_det_out(score_maps, orint_maps, det.scale_list, det.score_com_strength, det.scale_com_strength)
                    score_maps,orint_maps = det(im2_data)
                    im2_rawsc, im2_scale, im2_orin = handle_det_out(score_maps, orint_maps, det.scale_list, det.score_com_strength, det.scale_com_strength)
                    im1_gtscale, im1_gtorin = gt_scale_orin(im2_scale, im2_orin, homo12, homo21)
                    im2_gtscale, im2_gtorin = gt_scale_orin(im1_scale, im1_orin, homo21, homo12)
                    im2_score = filter_border(im2_rawsc)
                    im1w_score = warp(im2_score, homo12)
                    im1_gtsc, im1_topkmask, im1_topkvalue = det.process(im1w_score)
                    im1_score = filter_border(im1_rawsc)
                    im2w_score = warp(im1_score, homo21)
                    im2_gtsc, im2_topkmask, im2_topkvalue = det.process(im2w_score)
                    im1_ppair, im1_limc, im1_rimcw = pair(
                        im1_topkmask,
                        im1_topkvalue,
                        im1_scale,
                        im1_orin,
                        im1_info,
                        im1_raw,
                        homo12,
                        im2_gtscale,
                        im2_gtorin,
                        im2_info,
                        im2_raw,
                        cfg.PATCH.SIZE,
                    )
                    im2_ppair, im2_limc, im2_rimcw = pair(
                        im2_topkmask,
                        im2_topkvalue,
                        im2_scale,
                        im2_orin,
                        im2_info,
                        im2_raw,
                        homo21,
                        im1_gtscale,
                        im1_gtorin,
                        im1_info,
                        im1_raw,
                        cfg.PATCH.SIZE,
                    )
                    im1_lpatch, im1_rpatch = im1_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)
                    im2_lpatch, im2_rpatch = im2_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)
                    im1_lpatch = des.input_norm(im1_lpatch)
                    im2_lpatch = des.input_norm(im2_lpatch)
                    im1_rpatch = des.input_norm(im1_rpatch)   
                    im2_rpatch = des.input_norm(im2_rpatch)
                    im1_lpdes, im1_rpdes = des(im1_lpatch), des(im1_rpatch)
                    im2_lpdes, im2_rpdes = des(im2_lpatch), des(im2_rpatch)
                    endpoint = {
                        "im1_limc": im1_limc,
                        "im1_rimcw": im1_rimcw,
                        "im2_limc": im2_limc,
                        "im2_rimcw": im2_rimcw,
                        "im1_lpdes": im1_lpdes,
                        "im1_rpdes": im1_rpdes,
                        "im2_lpdes": im2_lpdes,
                        "im2_rpdes": im2_rpdes,
                    }

                    desloss = (
                        des.module.criterion(endpoint)
                        if mgpu
                        else des.criterion(endpoint)
                    )
                    desloss.backward()
                    des_optim.step()
                for det_train in range(0, cfg.TRAIN.DET):
                    det.zero_grad()
                    det_optim.zero_grad()
                    im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = batch

                    score_maps,orint_maps = det(im1_data)
                    im1_rawsc, im1_scale, im1_orin = handle_det_out(score_maps, orint_maps, det.scale_list, det.score_com_strength, det.scale_com_strength)
                    score_maps,orint_maps = det(im2_data)
                    im2_rawsc, im2_scale, im2_orin = handle_det_out(score_maps, orint_maps, det.scale_list, det.score_com_strength, det.scale_com_strength)

                    im2_score = filter_border(im2_rawsc)
                    im1w_score = warp(im2_score, homo12)
                    im1_visiblemask = warp(
                        im2_score.new_full(im2_score.size(), fill_value=1, requires_grad=True),
                        homo12, )
                    im1_gtsc, im1_topkmask, im1_topkvalue = det.process(im1w_score)

                    im1_score = filter_border(im1_rawsc)
                    im2w_score = warp(im1_score, homo21)
                    im2_visiblemask = warp(
                        im2_score.new_full(im1_score.size(), fill_value=1, requires_grad=True),
                        homo21, )
                    im2_gtsc, im2_topkmask, im2_topkvalue = det.process(im2w_score)

                    im1_score = det.process(im1_rawsc)[0]
                    im2_score = det.process(im2_rawsc)[0]
                    im1_predpair, _, _ = pair(
                        im1_topkmask,
                        im1_topkvalue,
                        im1_scale,
                        im1_orin,
                        im1_info,
                        im1_raw,
                        homo12,
                        im2_scale,
                        im2_orin,
                        im2_info,
                        im2_raw,
                        cfg.PATCH.SIZE,
                    )
                    im2_predpair, _, _ = pair(
                        im2_topkmask,
                        im2_topkvalue,
                        im2_scale,
                        im2_orin,
                        im2_info,
                        im2_raw,
                        homo21,
                        im1_scale,
                        im1_orin,
                        im1_info,
                        im1_raw,
                        cfg.PATCH.SIZE,
                    )
                    # each is (N, 32, 32)
                    im1_lpredpatch, im1_rpredpatch = im1_predpair.chunk(chunks=2, dim=1)
                    im2_lpredpatch, im2_rpredpatch = im2_predpair.chunk(chunks=2, dim=1)
                    im1_lpredpatch = des.input_norm(im1_lpredpatch)
                    im2_lpredpatch = des.input_norm(im2_lpredpatch)
                    im1_rpredpatch = des.input_norm(im1_rpredpatch)
                    im2_rpredpatch = des.input_norm(im2_rpredpatch)
                    im1_lpreddes, im1_rpreddes = des(im1_lpredpatch), des(im1_rpredpatch)
                    im2_lpreddes, im2_rpreddes = des(im2_lpredpatch), des(im2_rpredpatch)
                    endpoint = {
                        "im1_score": im1_score,
                        "im1_gtsc": im1_gtsc,
                        "im1_visible": im1_visiblemask,
                        "im2_score": im2_score,
                        "im2_gtsc": im2_gtsc,
                        "im2_visible": im2_visiblemask,
                        "im1_lpreddes": im1_lpreddes,
                        "im1_rpreddes": im1_rpreddes,
                        "im2_lpreddes": im2_lpreddes,
                        "im2_rpreddes": im2_rpreddes,
                    }

                    detloss = (
                        det.module.criterion(endpoint)
                        if mgpu
                        else det.criterion(endpoint)
                    )
                    detloss.backward()
                    det_optim.step()

            Lr_Schechuler(cfg.TRAIN.DET_LR_SCHEDULE, det_optim, epoch, cfg)
            Lr_Schechuler(cfg.TRAIN.DES_LR_SCHEDULE, des_optim, epoch, cfg)

            # log
            if i_batch % cfg.TRAIN.LOG_INTERVAL == 0 and i_batch > 0:
                elapsed = time.time() - start_time
                det.eval()
                des.eval()
                with torch.no_grad():
                    parsed_trainbatch = parse_unsqueeze(train_data.dataset[0], device)
                    endpoint = get_all_endpoints(det,des,parsed_trainbatch)
                    detloss = (
                        det.module.criterion(endpoint)
                        if mgpu
                        else det.criterion(endpoint)
                    )
                    desloss = (
                        des.module.criterion(endpoint)
                        if mgpu
                        else des.criterion(endpoint)
                    )
                    PLT_SCALAR = {}
                    PLT = {"scalar": PLT_SCALAR}
                    PLT_SCALAR["pair_loss"] = detloss
                    PLT_SCALAR["hard_loss"] = desloss
                    PLTS = PLT["scalar"]
                    PLTS["Accuracy"] = getAC(endpoint["im1_lpdes"], endpoint["im1_rpdes"])
                    PLTS["det_lr"] = det_optim.param_groups[0]["lr"]
                    PLTS["des_lr"] = des_optim.param_groups[0]["lr"]
                    if mgpu:
                        mgpu_merge(PLTS)
                    iteration = (epoch - 1) * len(train_data) + (i_batch - 1)
                    writer_log(train_writer, PLT["scalar"], iteration)

                    pstring = (
                        "epoch {:2d} | {:4d}/{:4d} batches | ms {:4.02f} | "
                        "pair {:05.03f} | des {:05.03f} |".format(
                            epoch,
                            i_batch,
                            len(train_data) // cfg.TRAIN.BATCH_SIZE,
                            elapsed / cfg.TRAIN.LOG_INTERVAL,
                            PLTS["pair_loss"],
                            PLTS["hard_loss"],
                        )
                    )
                    # eval log
                    parsed_valbatch = parse_unsqueeze(val_data.dataset[0], device)
                    ept = get_all_endpoints(det,des,parsed_valbatch)
                    detloss = (
                        det.module.criterion(endpoint)
                        if mgpu
                        else det.criterion(endpoint)
                    )
                    desloss = (
                        des.module.criterion(endpoint)
                        if mgpu
                        else des.criterion(endpoint)
                    )
                    PLT_SCALAR = {}
                    PLT = {"scalar": PLT_SCALAR}
                    PLT_SCALAR["pair_loss"] = detloss
                    PLT_SCALAR["hard_loss"] = desloss
                    PLTS = PLT["scalar"]
                    PLTS["Accuracy"] = getAC(ept["im1_lpdes"], ept["im1_rpdes"])
                    writer_log(test_writer, PLT["scalar"], iteration)
                    print(f"{gct()} | {pstring}")
                    start_time = time.time()

    ###############################################################################
    # evaluate function
    # ###############################################################################
    def evaluate(data_source):
        det.eval()
        des.eval()
        PreNN, PreNNT, PreNNDR = 0, 0, 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_source, 1):
                batch = parse_batch(sample_batched, device)

                TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR = eval_model(
                    det.module if mgpu else det,
                    des.module if mgpu else des,
                    batch,
                    cfg.MODEL.DES_THRSH,
                    cfg.MODEL.COO_THRSH,
                    cfg.TRAIN.TOPK,
                    cfg.PATCH.SIZE,
                )
                PreNN += TPNN / PNN
                PreNNT += TPNNT / PNNT
                PreNNDR += TPNNDR / PNNDR
        length = len(data_source)
        PreNN, PreNNT, PreNNDR = (PreNN / length, PreNNT / length, PreNNDR / length)
        meanms = (PreNN + PreNNT + PreNNDR) / 3
        checkpoint_name = (
            f"NN_{PreNN:.3f}_NNT_{PreNNT:.3f}_NNDR_{PreNNDR:.3f}_MeanMS_{meanms:.3f}"
        )
        return checkpoint_name, meanms


    args.start_epoch = 0
    print(f"{gct()} : Start training")
    best_ms = None
    best_f = None
    start_epoch = args.start_epoch + 1
    end = cfg.TRAIN.EPOCH_NUM
    for epoch in range(start_epoch, end):
        epoch_start_time = time.time()
        train()
        checkpoint, val_ms = evaluate(val_data)

        # Save the model if the match score is the best we've seen so far.
        if not best_ms or val_ms >= best_ms:
            det_state = {
                "epoch": epoch,
                "state_dict": det.state_dict(),
                "det_optim": det_optim.state_dict(),
            }
            det_filname = f"{args.save}/model/e{epoch:03d}_{checkpoint}_det.pth.tar"
            torch.save(det_state, det_filname)
            des_state = {
                "epoch": epoch,
                "state_dict": des.state_dict(),
                "des_optim": des_optim.state_dict(),
            }
            des_filename = f"{args.save}/model/e{epoch:03d}_{checkpoint}_des.pth.tar"
            torch.save(des_state, des_filename)
            best_ms = val_ms
            # best_f = filename

        print("-" * 96)
        print(
            "| end of epoch {:3d} | time: {:5.02f}s | val ms {:5.03f} | best ms {:5.03f} | ".format(
                epoch, (time.time() - epoch_start_time), val_ms, best_ms
            )
        )
        print("-" * 96)

    # # Load the best saved model.
    # with open(best_f, "rb") as f:
    #     det.load_state_dict(torch.load(f)["state_dict"])

    # Run on test data.
    _, test_ms = evaluate(test_data)
    print("=" * 96)
    print("| End of training | test ms {:5.03f}".format(test_ms))
    print("=" * 96)
