# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("./material/3_1.png")
img2 = cv2.imread("./material/3_2.png")
width = img1.shape[1]
sift = cv2.xfeatures2d.SIFT_create(512)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
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
plt.imsave("SIFT3.png",outImg)
# # Draw all points
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
# plt.imshow(img3), plt.show()