# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import sys
sys.path.extend(['/home/wang/workspace/FasterNet/NASNet_0.1'])

from model.det import RFDet
from model.des import HardNetNeiMask
from model.network import Network
from config import cfg
# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model_file = "/home/wang/workspace/CVToolbox/benchmark/evaluationMMA/model/e068_NN_0.429_NNT_0.612_NNDR_0.869_MeanMS_0.637.pth.tar"

# Creating CNN model
det = RFDet(
        cfg.TRAIN.score_com_strength,
        cfg.TRAIN.scale_com_strength,
        cfg.TRAIN.NMS_THRESH,
        cfg.TRAIN.NMS_KSIZE,
        512,
        cfg.MODEL.GAUSSIAN_KSIZE,
        cfg.MODEL.GAUSSIAN_SIGMA,
        cfg.MODEL.KSIZE,
        cfg.MODEL.padding,
        cfg.MODEL.dilation,
        cfg.MODEL.scale_list,
    )
des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
model = Network(
    det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, 512
)
model = model.to(device=device)
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint["state_dict"])


img1_path = "./material/4_1.png"
img2_path = "./material/4_2.png"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
width = img1.shape[1]
kp1, des1, _, _,_,_= model.detectAndCompute(img1_path, device, (img1.shape[0], img1.shape[1]))
kp2, des2, _, _,_,_= model.detectAndCompute(img2_path, device, (img2.shape[0], img2.shape[1]))

def to_cv2_kp(kp):
    return cv2.KeyPoint(kp[2], kp[1], 0)

kp1 = list(map(to_cv2_kp, kp1))
kp2 = list(map(to_cv2_kp, kp2))

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1.cpu().detach().numpy(), trainDescriptors=des2.cpu().detach().numpy(), k=2)
good = [m for (m, n) in matches if m.distance < 0.8 * n.distance]
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(0, 0, 255),
                   matchesMask=matchesMask,
                   flags=2)

nonmatchesMask=[1 if(j==0) else 0 for j in matchesMask]
draw_params2 = dict(matchColor=(255, 0, 0),
                    singlePointColor=(0, 0, 255),
                    matchesMask=nonmatchesMask,
                    flags=2)

outImg = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params2)
outImg = cv2.drawMatches(outImg[:, :width, :], kp1, outImg[:, width:, :], kp2, good, None, **draw_params)
cv2.drawKeypoints(outImg[:, :width, :],kp1,outImg[:, :width, :])
cv2.drawKeypoints(outImg[:, width:, :],kp2,outImg[:, width:, :])
plt.imshow(outImg), plt.show()
plt.imsave("Our4.png",outImg)