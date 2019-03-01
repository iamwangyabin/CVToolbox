import numpy as np
from PIL import Image

img_IN = Image.open('sample.jpg')
img_IN = img_IN.convert('L')
img = np.array(img_IN)
row = img.shape[0]
col = img.shape[1]

# 一些参数
Ksize = 3
K_row = Ksize*2+1
segma = 0.2

center = Ksize
conv = np.zeros((K_row, K_row))
for i in range(K_row):
    for j in range(K_row):
        temp = np.exp(-((i-center)**2+(j-center)**2)/(2*segma**2))
        conv[i, j] = temp/(2*np.pi*segma*segma)

conv = conv/np.sum(conv)   # 归一化

img_gaus = np.zeros((row, col))
for i in range(row):
    for j in range(col):
        if i < Ksize or j < Ksize or i >= row-Ksize or j >= col-Ksize:
            img_gaus[i, j] = img[i, j]  # 边缘不做处理
        else:
            miniMatrix = img[i-Ksize:i+Ksize+1, j-Ksize:j+Ksize+1]
            # print(miniMatrix.shape)
            img_gaus[i, j] = np.sum(miniMatrix.T.dot(conv))

k = Image.fromarray(np.uint8(img_gaus))
k.save('3.jpg')

# 与OPENCV进行比较并无差异
# import cv2
# blur = cv2.GaussianBlur(img, (3, 3), 0.01)
# cv2.imwrite("sdaf.jpg", blur)

# def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
#     kx = cv2.getGaussianKernel(kernel_size, sigma)
#     ky = cv2.getGaussianKernel(kernel_size, sigma)
#     return np.multiply(kx, np.transpose(ky))

# sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8

# m = gaussian_kernel_2d_opencv(kernel_size=7, sigma=0.2)
