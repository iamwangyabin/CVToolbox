import numpy as np
import os
import torch
from tqdm import tqdm
import sys
import cv2
sys.path.extend(['/home/wang/workspace/RFSLAM_offline/RFNET/rfnet/'])

from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.train_utils import parse_batch
from hpatch_dataset import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def distance_matrix_vector(anchor, positive):
    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)
    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = y.transpose(0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = x.transpose(0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    eps = 1e-8
    return torch.sqrt(dist.clamp(min=eps, max=np.inf))


def ptCltoCr(leftC, homolr, clamp, maxh, maxw):
    # projective transform im1_C back to im2 called im2_Cw
    B = 1
    leftC_homo = leftC.clone()
    leftC_homo[:, 3] = leftC_homo[:, 3] + 1  # (B*topk, 4) (b, y, x, 1)
    leftC_homo = leftC_homo[:, 1:]  # (B*topk, 3) (y, x, 1)
    leftC_homo = leftC_homo.index_select(
        1, leftC_homo.new_tensor([1, 0, 2])
    )  # (B*topk, 3) [[x], [y], [1]]
    leftC_homo = leftC_homo.view(B, -1, 3)  # (B, topk, 3)
    leftC_homo = leftC_homo.permute(0, 2, 1)  # (B, 3, topk)

    rightC_homo = torch.matmul(homolr, leftC_homo.float())  # (B, 3, topk) (x, y, h)
    rightC_homo = rightC_homo.permute(0, 2, 1)  # (B, topk, 3) (x, y, h)
    # (B, topk, 3) (x, y, h) to 1
    rightC_homo = rightC_homo / (torch.unsqueeze(rightC_homo[:, :, 2], -1) + 1e-8)
    rightC_homo = rightC_homo.round().long()
    if clamp:
        rightC_homo[:, :, 0] = rightC_homo[:, :, 0].clamp(min=0, max=maxw - 1)
        rightC_homo[:, :, 1] = rightC_homo[:, :, 1].clamp(min=0, max=maxh - 1)

    topk = rightC_homo.size(1)
    batch_v = (
        torch.arange(B, device=rightC_homo.device).view(B, 1, 1).repeat(1, topk, 1)
    )  # (B, topk, 1)
    # (B, topk, 4) (B, x, y, h)
    rightC_homo = torch.cat((batch_v, rightC_homo), -1)
    rightC_homo = rightC_homo.contiguous().view(-1, 4)  # (B*topk, 4) (B, x, y, h)
    rightC_homo = rightC_homo.index_select(
        1, rightC_homo.new_tensor([0, 2, 1, 3])
    )  # (B*topk, 4) (B, y, x, h)
    rightC_homo[:, 3] = rightC_homo[:, 3] - 1  # (B*topk, 4) (B, y, x, 0)
    return rightC_homo


def nearest_neighbor_match_score(des1, des2, kp1w, kp2, visible, COO_THRSH):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    nn_value, nn_idx = des_dist_matrix.min(dim=-1)
    nn_kp2 = kp2.index_select(dim=0, index=nn_idx)
    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correct_match_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(visible.sum().item(), 1)

    return correct_matches, predict_matches

def nearest_neighbor_threshold_match_score(
        des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH
):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    nn_value, nn_idx = des_dist_matrix.min(dim=-1)
    predict_label = nn_value.lt(DES_THRSH) * visible
    nn_kp2 = kp2.index_select(dim=0, index=nn_idx)
    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible
    correct_match_label = predict_label * correspondences_label
    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)

    return correct_matches, predict_matches


def threshold_match_score(des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    visible = visible.unsqueeze(-1).repeat(1, des_dist_matrix.size(1))
    predict_label = des_dist_matrix.lt(DES_THRSH) * visible
    coo_dist_matrix = pairwise_distances(kp1w[:, 1:3].float(), kp2[:, 1:3].float())
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible
    correct_match_label = predict_label * correspondences_label
    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)
    correspond_matches = max(correspondences_label.sum().item(), 1)
    return correct_matches, predict_matches, correspond_matches


def nearest_neighbor_distance_ratio_match(des1, des2, kp2, threshold):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    sorted, indices = des_dist_matrix.sort(dim=-1)
    Da, Db, Ia = sorted[:, 0], sorted[:, 1], indices[:, 0]
    DistRatio = Da / Db
    predict_label = DistRatio.lt(threshold)
    nn_kp2 = kp2.index_select(dim=0, index=Ia.view(-1))
    return predict_label, nn_kp2


def nearest_neighbor_distance_ratio_match_score(
        des1, des2, kp1w, kp2, visible, COO_THRSH, threshold=0.7
):
    predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(
        des1, des2, kp2, threshold
    )
    predict_label = predict_label * visible
    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible
    correct_match_label = predict_label * correspondences_label
    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)
    return correct_matches, predict_matches

def eval_sift(model,parsed_batch, DES_THRSH, COO_THRSH):
    im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = parsed_batch

    im1_data1 = im1_data.squeeze().permute(1, 2, 0).cpu().to(torch.uint8).numpy()
    im2_data2 = im2_data.squeeze().permute(1, 2, 0).cpu().to(torch.uint8).numpy()

    kps1, des1 = model.detectAndCompute(im1_data1, None)
    kps2, des2 = model.detectAndCompute(im2_data2, None)

    kp1 = []
    kp2 = []
    for i in kps1:
        temp = [0, int(i.pt[1]), int(i.pt[0]), 0]
        kp1.append(temp)
    for i in kps2:
        temp = [0, int(i.pt[1]), int(i.pt[0]), 0]
        kp2.append(temp)

    kp1 = torch.tensor(kp1,device = device)
    kp2 = torch.tensor(kp2,device = device)
    des1 = torch.tensor(des1,device = device)
    des2 = torch.tensor(des2,device = device)

    kp1w = ptCltoCr(kp1, homo12, clamp=False, maxh=im1_data.shape[2], maxw=im1_data.shape[3])
    _, _, maxh, maxw = im2_data.size()
    visible = kp1w[:, 2].lt(maxw) * kp1w[:, 1].lt(maxh)
    TPNN, PNN = nearest_neighbor_match_score(des1, des2, kp1w, kp2, visible, COO_THRSH)
    TPNNT, PNNT = nearest_neighbor_threshold_match_score(
        des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH
    )
    TPNNDR, PNNDR = nearest_neighbor_distance_ratio_match_score(
        des1, des2, kp1w, kp2, visible, COO_THRSH
    )
    return TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR


lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def get_dataloader(data):
    data_loader = []
    PPT = [cfg.PROJ.TRAIN_PPT, (cfg.PROJ.TRAIN_PPT + cfg.PROJ.EVAL_PPT)]
    use_all = {"view": False, "illu": True, "ef": True}

    if data == "view":
        data_loader = DataLoader(
            HpatchDataset(data_type="test", PPT=PPT, use_all=use_all[data], csv_file=cfg[data]["csv"],
                          root_dir="/home/wang/workspace/RFSLAM_offline/RFNET/data/hpatch_v_sequence",
                          transform=transforms.Compose(
                              [Rescale((960, 1280)),
                               Rescale((480, 640)), ToTensor()]), ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    elif data == 'illu':
        data_loader = DataLoader(
            HpatchDataset(data_type="test", PPT=PPT, use_all=use_all[data], csv_file=cfg[data]["csv"],
                          root_dir="/home/wang/workspace/RFSLAM_offline/RFNET/data/hpatch_i_sequence",
                          transform=transforms.Compose(
                              [Rescale((960, 1280)),
                               Rescale((480, 640)), ToTensor()]), ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        pass

    return data_loader

def benchmark_features():
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    model = cv2.xfeatures2d.SURF_create()

    with torch.no_grad():
        data_loader = get_dataloader("view")
        for i_batch, sample_batched in tqdm(enumerate(data_loader, 1)):
            batch = parse_batch(sample_batched, device)
            for thr in rng:
                TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR = eval_sift(model, batch, 1, thr)
                PreNN = TPNN / PNN
                PreNNT = TPNNT / PNNT
                PreNNDR = TPNNDR / PNNDR
                meanms = (PreNN + PreNNT + PreNNDR) / 3
                v_err[thr] += meanms
        n_v = i_batch

    with torch.no_grad():
        data_loader = get_dataloader("illu")
        for i_batch, sample_batched in tqdm(enumerate(data_loader, 1)):
            batch = parse_batch(sample_batched, device)
            for thr in rng:
                TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR = eval_sift(model, batch, 1, thr)
                PreNN = TPNN / PNN
                PreNNT = TPNNT / PNNT
                PreNNDR = TPNNDR / PNNDR
                meanms = (PreNN + PreNNT + PreNNDR) / 3
                i_err[thr] += meanms
        n_i = i_batch

    return i_err, v_err, n_i, n_v

if __name__ == '__main__':
    method = 'surf'
    output_file = os.path.join('./cache', method + '.npy')
    np.save(output_file, benchmark_features())

