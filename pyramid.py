import cv2


# 高斯金字塔
def pyramid_image(image):
    level = 3  # 金字塔的层数
    temp = image.copy()  # 拷贝图像
    pyramid_images = []
    for i in range(level):
        dst = cv2.pyrDown(temp)
        pyramid_images.append(dst)
        # cv2.imshow("高斯金字塔"+str(i), dst)
        temp = dst.copy()
    return pyramid_images


src = cv2.imread("IMG_2784.JPG")
cv2.imshow("原来", src)
pyramid_image(src)
cv2.waitKey(0)
cv2.destroyAllWindows()
