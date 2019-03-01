import glob
import cv2
import numpy as np
import PIL.Image as Image

def calibrate_opencv():
    w = 8
    h = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    images = glob.glob('Calibration/Imgs/*')

    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            # cv2.cornerSubPix(gray1, corners*4, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners*4)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx



# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
#     mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
# # 反投影误差
# total_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(
#         objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     total_error += error
# print("total error: ", total_error/len(objpoints))

def getHomoCV(imgpoints,objpoints):
    homo=[]
    for i in range(len(objpoints)):
        h = cv2.findHomography(imgpoints[i], objpoints[i])
        homo.append(h)
    return homo


w = 8
h = 6
images = glob.glob('Calibration/Imgs/*')
j=1
for i in images:
    img = cv2.imread(i)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    img2 = cv2.imread(i)
    # cv2.drawChessboardCorners(img2, (w, h), corners*4, ret)
    corners4=corners*4
    for i in corners4:
        cv2.circle(img2, tuple(i[0]), 30, (0,0,255), 60)
        cv2.circle(img2, tuple(i[0]), 4, (0, 0, 255),8)
    cv2.imwrite("corners"+str(j)+".jpg",img2)
    j=j+1

# cv2.namedWindow("findCorners", 2)
# cv2.imshow('findCorners', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


w = 8
h = 6
images = glob.glob('Calibration/Imgs/*')


font = cv2.FONT_HERSHEY_SIMPLEX
j=1
for i in images:
    img = cv2.imread(i)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    img2 = cv2.imread(i)
    # cv2.drawChessboardCorners(img2, (w, h), corners*4, ret)
    corners4=corners*4
    num=1
    for i in corners4:
        cv2.putText(img2, str(num),tuple(i[0]), font, 1.6, (255, 255, 0), 2)
        num=num+1
    cv2.imwrite("num"+str(j)+".jpg",img2)
    j=j+1



def calibrate_opencv():
    w = 9
    h = 7
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    images = glob.glob('Calibration/cal/*')

    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            cv2.cornerSubPix(gray, corners*4, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners*4)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx
