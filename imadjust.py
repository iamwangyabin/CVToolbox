import numpy as np
import PIL.Image as Image
import bisect
from numba import jit

# 这个版本至少在变负片时候有问题
def imadjust2(src, tol=1, vin=[0, 255], vout=(0, 255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img
    assert len(src.shape) == 2, "Input image should be 2-dims"
    tol = max(0, min(100, tol))
    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]
        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)
    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src < vin[0]] = 0
    vd = vs*scale+0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    dst = vd
    return dst

# 转换成负片
def inverte(img):
    img = (255-img)
    return img

# 这个可以用
def imadjust(x,a,b,c,d,gamma=1):
    # 是MATLAB里imadjust简化版本，将图像范围从[a,b]转换到[c,d].
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

if __name__ == '__main__':
    img = Image.open('sample.jpg').convert('L')
    img_np = np.array(img)
    image_f = img_np.astype("float")
    # 明暗反转
    u = imadjust(img_np, 0, 255, 255, 0, 1)
    img1 = Image.fromarray(np.uint8(u))
    img1.show()
