import cv2
import numpy as np

# cap = cv2.VideoCapture(0)

# #获取第一帧
# ret, frame1 = cap.read()
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)

# #遍历每一行的第1列
# hsv[...,1] = 255


# while(1):
#     ret, frame2 = cap.read()
#     next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

#     #返回一个两通道的光流向量，实际上是每个点的像素位移值
#     flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # print(flow.shape)
#     # print(flow)

#     #笛卡尔坐标转换为极坐标，获得极轴和极角
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

#     cv2.imshow('frame2',rgb)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite('opticalfb.png',frame2)
#         cv2.imwrite('opticalhsv.png',rgb)
#     prvs = next

# cap.release()
# cv2.destroyAllWindows()


# import numpy as np
# import cv2


# cap = cv2.VideoCapture(0)


# # ShiTomasi 角点检测参数
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

# # lucas kanade光流法参数
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# # 创建随机颜色
# color = np.random.randint(0,255,(100,3))

# # 获取第一帧，找到角点
# ret, old_frame = cap.read()
# #找到原始灰度图
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# #获取图像中的角点，返回到p0中
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# # 创建一个蒙版用来画轨迹
# mask = np.zeros_like(old_frame)

# while(1):
#     ret,frame = cap.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 计算光流
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # 选取好的跟踪点
#     good_new = p1[st==1]
#     good_old = p0[st==1]

#     # 画出轨迹
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv2.add(frame,mask)

#     cv2.imshow('frame',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

#     # 更新上一帧的图像和追踪点
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)

# cv2.destroyAllWindows()
# cap.release()

def calGradient(image):
    '''
    传入单通道的图像
    '''
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_f = np.copy(image)
    image_f = image_f.astype("float")
    row = image.shape[0]
    column = image.shape[1]
    gradientX = np.zeros((row, column))
    gradientY = np.zeros((row, column))
    for x in range(row - 1):
        for y in range(column - 1):
            gx = abs(image_f[x + 1, y] - image_f[x, y])
            gy = abs(image_f[x, y + 1] - image_f[x, y])
            gradientX[x, y] = gx
            gradientY[x, y] = gy
    gradient = gradientX + gradientY
    gradient = gradient.astype("uint8")
    return gradientX, gradientY, gradient


image1 = cv2.imread('WIN_20181205_15_24_44_Pro.jpg')
image2 = cv2.imread('WIN_20181205_15_24_46_Pro.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image_f1 = np.copy(image1)
image_f1 = image_f1.astype("float")
image_f2 = np.copy(image2)
image_f2 = image_f2.astype("float")

dx1, dy1, _d1 = calGradient(image1)
dx2, dy2, _d2 = calGradient(image2)

dx = 0.5*(dx1+dx2)
dy = 0.5*(dy1+dy2)
dt = image_f1-image_f2
u = np.zeros(image1.shape)
v = np.zeros(image1.shape)
window = 9
half = int(np.floor(window/2))

for i in range((half+1), (dx.shape[0]-half)):
    for j in range((half+1), (dx.shape[1]-half)):
        tempdx = dx[i-half:i+half, j-half:j+half]
        tempdy = dy[i-half:i+half, j-half:j+half]
        tempdt = dt[i-half:i+half, j-half:j+half]
        tempdx = tempdx.flatten(order='C')
        tempdy = tempdy.flatten(order='C')
        tempdt = tempdt.flatten(order='C')
        q = tempdx.reshape((len(tempdx), 1))
        w = tempdy.reshape((len(tempdy), 1))
        r = tempdt.reshape((len(tempdt), 1))
        A = np.concatenate((q, w), axis=1)
        U = -np.linalg.pinv((A.T).dot(A)).dot(A.T).dot(tempdt)
        u[i, j] = U[0]
        v[i, j] = U[1]

result = np.zeros((u.shape[0], u.shape[1]))
umax = 0
vmax = 0
umin = 10000
vmin = 10000

for i in range(1, u.shape[0]):
    for j in range(1, u.shape[1]):
        if u[i, j] > umax:
            umax = u[i, j]
        if u[i, j] < umin:
            umin = u[i, j]
        if v[i, j] > vmax:
            vmax = v[i, j]
        if v[i, j] < vmin:
            vmin = v[i, j]

result1 = np.zeros((u.shape[0], u.shape[1]))
result2 = np.zeros((u.shape[0], u.shape[1]))
for i in range(1, u.shape[0]):
    for j in range(1, u.shape[1]):
        # result[i, j, 2] = 0.4
        result1[i, j] = (u[i, j]-umin)/(umax-umin)
        result2[i, j] = (v[i, j]-vmin)/(vmax-vmin)


result3 = np.floor(result1*100)
k = result3.astype("uint8")
cv2.imshow("gradient", k)
cv2.waitKey(0)
cv2.destroyAllWindows()
