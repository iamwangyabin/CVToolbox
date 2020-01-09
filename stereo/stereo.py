import cv2
import numpy as np
import time
import stereoconfig
# import pcl
# import pcl.pcl_visualization


# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)

    return im1, im2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    for k in range(15):
        cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)  # 直线间隔：100

    return output


# 视差计算
def disparity_SGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    param = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 400,
             'speckleRange': 2,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    # 构建SGBM对象
    sgbm = cv2.StereoSGBM_create(**param)
    # 计算视差图
    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        disparity_left = sgbm.compute(left_image_down, right_image_down)
        disparity_right = sgbm.compute(right_image_down, left_image_down)

    return disparity_left


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d).astype(np.int16)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1

def select(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y


# 点云显示
# def view_cloud(pointcloud):
#     cloud = pcl.PointCloud_PointXYZRGBA()
#     cloud.from_array(pointcloud)
#
#     try:
#         visual = pcl.pcl_visualization.CloudViewing()
#         visual.ShowColorACloud(cloud)
#         v = True
#         while v:
#             v = not (visual.WasStopped())
#     except:
#         pass


if __name__ == '__main__':
    ix, iy = -1, -1
    cv2.namedWindow("camera")
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth",select)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1600)
    cap.set(4, 600)  # 打开并设置摄像头

    height = 600
    width = 800
    # 读取相机内参和外参
    config = stereoconfig.stereoCamera()
    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)

    while True:
        ret, frame = cap.read()
        frame1 = frame[0:600, 0:800]
        frame2 = frame[0:600, 800:1600]  # 割开双目图像
        iml_rectified, imr_rectified = rectifyImage(frame1, frame2, map1x, map1y, map2x, map2y)
        line = draw_line(iml_rectified, imr_rectified)
        cv2.imshow("camera", line)
        imgL = cv2.cvtColor(iml_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imr_rectified, cv2.COLOR_BGR2GRAY)
        # 立体匹配
        disp = disparity_SGBM(iml_rectified, imr_rectified)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
        dispnorm = disp
        disp[disp < 0] = 0
        dispnorm = cv2.normalize(disp, dispnorm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp = np.divide(disp.astype(np.float32), 16.)  # 除以16得到真实视差（因为SGBM算法得到的视差是×16的）

        threeD = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)
        selectdis = threeD[iy:iy+100, ix:ix+100, 2]
        selectdis[selectdis>99999] = 0
        # print(np.sum(selectdis))
        if np.sum(selectdis==0) != 10000:
            meandist = np.sum(selectdis)/(10000 - np.sum(selectdis==0))
            print(str(meandist)[:8]+"mm")

        cv2.rectangle(dispnorm, (ix, iy), (ix + 100, iy + 100), (255, 0, 0), 2)

        cv2.imshow("depth", dispnorm)  # 显示出修正畸变前后以及深度图的双目画面

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("./snapshot/BM_left.jpg", imgL)
            cv2.imwrite("./snapshot/BM_right.jpg", imgR)
            cv2.imwrite("./snapshot/BM_depth.jpg", dispnorm)

    cap.release()
    cv2.destroyAllWindows()
