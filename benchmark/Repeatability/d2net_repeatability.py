import cv2
import torch
import random
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
sys.path.extend(['/home/wang/workspace/RFSLAM_offline/RFNET/rfnet/'])
sys.path.extend(['/home/wang/d2-net'])

from hpatch_dataset import (
    HpatchDataset,
    Grayscale,
    Normalize,
    Rescale,
    ToTensor,
)
from config import cfg
from utils.math_utils import pairwise_distances
from utils.train_utils import parse_batch

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

def parse_batch_np(one_batch, mean, std):
    im1_data_ = np.squeeze(one_batch['im1'].cpu().detach().numpy()) * std + mean
    im1_info_ = one_batch['im1_info'].float()
    homo12_ = np.squeeze(one_batch['homo12'].cpu().detach().numpy())
    im2_data_ = np.squeeze(one_batch['im2'].cpu().detach().numpy()) * std + mean
    im2_info_ = one_batch['im2_info'].float()
    homo21_ = np.squeeze(one_batch['homo21'].cpu().detach().numpy())
    im1_raw_ = (one_batch['im1_raw'] * std + mean).float()
    im2_raw_ = (one_batch['im2_raw'] * std + mean).float()
    return im1_data_, im1_info_, homo12_, im2_data_, im2_info_, homo21_, im1_raw_, im2_raw_


def topk(kps, scores, k_=cfg.TRAIN.TOPK):
    kpr = np.array([kp for kp in scores])
    kpc = np.array([[kp[0], kp[1]] for kp in kps])
    idx = np.argsort(kpr)
    if len(kpr) >= k_:
        idx = idx[len(kpr) - k_:]
    return kpc[idx], idx


def ptCltoCr(kpc, homo):
    ones = np.ones_like(kpc)
    kpc = np.concatenate((kpc, ones), axis=-1)[:, :3]
    kpcwhomo = np.matmul(homo, kpc.transpose())
    kpcw = kpcwhomo.transpose()
    kpcw = kpcw / np.expand_dims(kpcw[:, 2], axis=-1)
    kpcw = kpcw[:, :2]
    return kpcw


def caluseful(kp1c_, kp2c_, homo12_, im2_data_, coo_t=5.0):
    kp2_ = torch.from_numpy(kp2c_).float()
    kp1w = torch.from_numpy(ptCltoCr(kp1c_, homo12_)).float()

    maxh, maxw = np.shape(im2_data_)  # (1280 960)
    visible = kp1w[:, 0].lt(maxw) * kp1w[:, 1].lt(maxh)
    useful_ = visible.sum().item()

    coo_dist_matrix = pairwise_distances(kp1w, kp2_)
    visible = visible.unsqueeze(-1).repeat(1, coo_dist_matrix.size(1))

    repeats_ = coo_dist_matrix.le(coo_t)
    repeatable_ = (repeats_ * visible).sum(dim=1).gt(0).sum().item()

    return repeatable_, max(useful_, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=1024, type=int)  # topk
    parser.add_argument('--data', default='e', type=str)  # dataset
    args = parser.parse_args()

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Creating CNN model
    model = D2Net(
        model_file='/home/wang/d2-net/models/d2_tf.pth',
        use_relu=True,
        use_cuda=use_cuda
    )

    random.seed(cfg.PROJ.SEED)
    torch.manual_seed(cfg.PROJ.SEED)
    np.random.seed(cfg.PROJ.SEED)

    root_dir = '/home/wang/workspace/RFSLAM_offline/RFNET/data/'
    csv_file = None
    seq = None
    a = None
    if args.data == 'v':
        csv_file = 'hpatch_view.csv'
        root_dir += 'hpatch_v_sequence'
        seq = 'view'
        a = False
    elif args.data == 'i':
        csv_file = 'hpatch_illum.csv'
        root_dir += 'hpatch_i_sequence'
        seq = 'illu'
        a = True
    elif args.data == 'e':
        csv_file = 'EFDataset.csv'
        root_dir += 'EFDataset'
        seq = 'ef'
        a = True
    else:
        print(f'cannot find {args.data}')
        exit(-1)

    mean=cfg[seq]["MEAN"]
    std=cfg[seq]["STD"]
    data_loader = DataLoader(
        HpatchDataset(
            data_type="test",
            PPT=[0.8,0.9],
            use_all=a,
            csv_file=csv_file,
            root_dir=root_dir,
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(mean=mean, std=std),
                    Rescale((960, 1280)),
                    Rescale((480, 640)),
                    ToTensor()
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    useful_list = []
    repeat_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader, 1):
            im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = parse_batch(sample_batched, device)

            # (angle, class_id, octave, pt, response, size)
            keypoints1, scores1, descriptors1 = process_multiscale(
                im1_data.repeat(1,3,1,1),
                model,
                scales=[1]
            )
            keypoints2, scores2, descriptors2 = process_multiscale(
                im2_data.repeat(1,3,1,1),
                model,
                scales=[1]
            )

            kp1c,_ = topk(keypoints1,scores1,args.k)
            kp2c,_ = topk(keypoints2,scores2,args.k)
            im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = parse_batch_np(sample_batched, mean, std)
            repeatable, useful = caluseful(kp1c, kp2c, homo12, im2_data)
            useful_list.append(useful), repeat_list.append(repeatable)

    usefuls = np.array(useful_list)
    repeats = np.array(repeat_list)

    repeatability = repeats.sum() / usefuls.sum()

    print(f'\n\tdata: {args.data} len:{len(data_loader)} k:{args.k}\trepeats:{repeats.sum()}\trepeatability {repeatability:.05f}')