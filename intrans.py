import numpy as np
import PIL.Image as Image
import imadjust
#######################
# 这个函数目前弃了，感觉没啥用，实现起来还很麻烦
#######################
img = Image.open('sample.jpg').convert('L')
img_np = np.array(img)
image_f = img_np.astype("float")

def gammaTransform(f, gamma):
    pass


def stretchTransform(f, varargin):
    pass


def spcfiedTransform(f, txfun):
    pass


def logTransform(f, varargin):
    pass



def intrans(f, method, varargin):
    if method == 'log':
        g = logTransform(f, varargin)
        return g
    '''
    这段检查一下是不是在01中间
    '''
    if method == 'neg':
        g = imcomplement(f)
    elif method == 'gamma':
        g = gammaTransform(f, varargin)
    elif method == 'stretch':
        g = stretchTransform(f, varargin)
    elif method == 'specified':
        g = spcfiedTransform(f, varargin)
    else:
        pass
