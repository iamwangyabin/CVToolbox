# coding=utf-8
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

material_path = "/home/wang/workspace/CVToolbox/Detector/det_des/material"
img_path = os.listdir(material_path)

img1 = cv2.imread(os.path.join(material_path, img_path[0]))
img2 = cv2.imread(os.path.join(material_path, img_path[1]))

# 最大特征点数,需要修改，5000太大。
orb = cv2.ORB_create(5000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 提取并计算特征点
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# knn筛选结果
matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
good = [m for (m, n) in matches if m.distance < 0.8 * n.distance]

# RANSAC
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=(0, 0, 255),
                   matchesMask=matchesMask,
                   flags=2)  # draw only inliers
img1_kp=img1
cv2.drawKeypoints(img1,kp1,img1_kp,[255,0,0])
plt.imshow(img1_kp), plt.show()
img2_kp=img1
cv2.drawKeypoints(img2,kp2,img2_kp,[255,0,0])
plt.imshow(img2_kp), plt.show()



vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(vis), plt.show()

# Draw all points
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
plt.imshow(img3), plt.show()