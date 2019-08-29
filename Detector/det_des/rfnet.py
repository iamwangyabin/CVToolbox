# coding=utf-8
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def get_kp(path):
    img_kpts = np.genfromtxt(path)
    img_kpts_list = []
    for i in img_kpts:
        temp = cv2.KeyPoint(i[1], i[0], 0)
        img_kpts_list.append(temp)
    return img_kpts_list


material_path = "/home/wang/workspace/CVToolbox/Detector/det_des/material"
RFNET_DBOW_path="/home/wang/workspace/data/rgbd_dataset_freiburg1_desk/RFNet_DBOW"
img_path = os.listdir(material_path)

img1 = cv2.imread(os.path.join(material_path, img_path[0]))
img2 = cv2.imread(os.path.join(material_path, img_path[1]))

img1_feats_path =os.path.join(RFNET_DBOW_path,img_path[0][:-4]+"_feats.txt")
img1_kpts_path =os.path.join(RFNET_DBOW_path,img_path[0][:-4]+"_kpts.txt")
des1=np.genfromtxt(img1_feats_path).astype(np.float32)
kp1=get_kp(img1_kpts_path)

img2_feats_path =os.path.join(RFNET_DBOW_path,img_path[1][:-4]+"_feats.txt")
img2_kpts_path =os.path.join(RFNET_DBOW_path,img_path[1][:-4]+"_kpts.txt")
des2=np.genfromtxt(img2_feats_path).astype(np.float32)
kp2=get_kp(img2_kpts_path)

# 提取并计算特征点
bf = cv2.BFMatcher(cv2.NORM_L1)
# knn筛选结果
matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
good = [m for (m, n) in matches if m.distance < 0.8 * n.distance]

# RANSAC
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=(0, 0, 255),
                   matchesMask=matchesMask,
                   flags=2)  # draw only inliers

vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(vis), plt.show()

# Draw all points
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
plt.imshow(img3), plt.show()


img1_kp=img1
cv2.drawKeypoints(img1,kp1,img1_kp,[255,0,0])
plt.imshow(img1_kp), plt.show()

img2_kp=img2
cv2.drawKeypoints(img2,kp2,img2_kp,[255,0,0])
plt.imshow(img2_kp), plt.show()
