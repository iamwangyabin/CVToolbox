import PIL.Image as Image
import numpy as np

##################图像大多数操作可以用卷积来实现
##################Ix=I*Dx
img=Image.open('sample.jpg')
img=img.convert('L')
img_np = np.array(img)

filter=np.array([-1,0,1])

# 这个是竖着的滤波
def imfilterY(img,filter):
    s=img.shape
    r=np.zeros(s)
    for i in range(1,s[0]-1):
        for j in range(1,s[1]-1):
            temp=img[i-1:i+2,j].dot(filter)
            # temp=img[i-1:i+1,j-1:j+1].T.dot(filter)
            r[i,j]=temp
            print(img[i-1:i+2,j])
    return r

# 这个是横着的滤波
def imfilter(img,filter):
    s=img.shape
    r=np.zeros(s)
    for i in range(1,s[0]-1):
        for j in range(1,s[1]-1):
            temp=img[i,j-1:j+2].dot(filter)
            # temp=img[i-1:i+1,j-1:j+1].T.dot(filter)
            r[i,j]=temp
            print(img[i-1:i+2,j])
    return r

u=imfilter(img_np,filter)