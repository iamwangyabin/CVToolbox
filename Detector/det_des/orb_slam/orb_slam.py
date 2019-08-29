# coding=utf-8
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

material_path = "/home/wang/workspace/CVToolbox/Detector/det_des/material"
img_path = os.listdir(material_path)

def get_kp_des(path):
    img1 = cv2.imread(path)
    # Numpy [[v,h]]
    s_img1=[]
    vtemp = np.vsplit(img1, 5)
    for i in vtemp:
        htemp=np.hsplit(i,5)
        s_img1.append(htemp)
    # 图像坐标是左上角
    # [[0,0],[0,128],[0,128*2],[0,128*2]
    #  [96,0],[0,128],[0,128*2],[0,128*2]]                                 ]
    origin=[]
    for i in range(5):
        # v = []
        for j in range(5):
            temp = [i*96,j*128]
            # v.append(temp)
            origin.append(temp)
    # 最大特征点数,需要修改，5000太大。
    orb = cv2.ORB_create(1000)
    kps=[]
    dess=[]
    for i in s_img1:
        for j in i:
            kp, des = orb.detectAndCompute(j, None)
            kps.append(kp)
            dess.append(des)
    merged_kps=[]
    merged_dess=[]
    num=0
    for i in kps:
        if i :
            for j in i:
                x,y=j.pt
                x=x+origin[num][1]
                y=y+origin[num][0]
                j.pt=(x,y)
                merged_kps.append(j)
            for k in dess[num]:
                merged_dess.append(k)
        num += 1
    return np.array(merged_kps),np.array(merged_dess)

img1_path = os.path.join(material_path, img_path[0])
img1 = cv2.imread(img1_path)
kp1,des1 = get_kp_des(img1_path)

img2_path = os.path.join(material_path, img_path[2])
img2 = cv2.imread(img2_path)
kp2,des2 = get_kp_des(img2_path)

# 提取并计算特征点
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# knn筛选结果
matches = bf.knnMatch(queryDescriptors=des1, trainDescriptors=des2, k=2)
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


vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(vis), plt.show()

# Draw all points
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
plt.imshow(img3), plt.show()

img1_kp=img1
cv2.drawKeypoints(img1,kp1,img1_kp,[255,0,0])
plt.imshow(img1_kp), plt.show()


img2_kp=img1
cv2.drawKeypoints(img2,kp2,img2_kp,[255,0,0])
plt.imshow(img2_kp), plt.show()


# coding=utf-8
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

material_path = "/home/wang/workspace/CVToolbox/Detector/det_des/material"
img_path = os.listdir(material_path)
img1_path = os.path.join(material_path, img_path[0])

class ExtractorNode:
    def __init__(self):
        self.vKeys=[]


class ORBextractor():
    def __init__(self):
        pass
    def DivideNode(self):
        cv2.boxPoints()



















img1 = cv2.imread(img1_path)

x,y = img1.shape[0:2]
# Numpy [[v,h]]
s_img1 = []
vtemp = np.vsplit(img1, 5)
for i in vtemp:
    htemp = np.hsplit(i, 5)
    s_img1.append(htemp)

orb = cv2.ORB_create(100)
img2 = cv2.resize(s_img1[0][1],(y,x))

kp, des = orb.detectAndCompute(img2, None)

img1_kp=img2
cv2.drawKeypoints(img2,kp,img1_kp,[255,0,0])
plt.imshow(img1_kp), plt.show()