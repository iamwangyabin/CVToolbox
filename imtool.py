from PIL import Image
import numpy as np

# resize PIL image object
# input : numpy
# output  numpy
def imresize(im,sz):
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))

# 直方图均衡
def histeq(im,nbr_bins=256):
    imhist,bins=np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf=imhist.cumsum()
    cdf=255*cdf/cdf[-1]

    im2=np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

