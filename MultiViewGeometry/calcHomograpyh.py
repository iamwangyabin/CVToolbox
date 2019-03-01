import random
import numpy as np
########## matlab ; 是竖向拼接 对应numpy是vstack()
# 生成棋盘模板
w = 8
h = 6
pout=[]
for i in range(h):
    for j in range(w):
        pout.append(np.array((float(j*23),float(i*23))))
pout=np.array(pout)

# 使用RAN计算单应矩阵，单位应该是毫米
def calcHomography(w,h,pout,pin):
    pin=np.squeeze(pin)
    finalN=None
    minScore=99999999
    for i in range(1000):
        # 首先是从点集里面随机选出来四个来求，因为是RANSAC嘛。
        subset=random.sample(range(w*h),4)
        origin=[]
        dest=[]
        for k in subset:
            origin.append(pin[k])
            dest.append(pout[k])
        origin=np.array(origin)
        dest=np.array(dest)
        # x = dest[:,0]
        # y = dest[:,1]
        # X = origin[:,0]
        # Y = origin[:,1]
        # 之这样做的结果错了，最大的问题在于求出来的矩阵位置是错的，开始没有排好顺序，后面结果处理很麻烦，但是方法是对的。
        # rows0 = np.zeros((3, 4))
        # rowsXY=-np.vstack((X, Y, np.ones((1, 4))))
        # hx = np.hstack((rowsXY.T, rows0.T, np.transpose([x*X]), np.transpose([x*Y]), np.transpose([x])))
        # hy = np.hstack((rows0.T, rowsXY.T, np.transpose([y * X]), np.transpose([y * Y]), np.transpose([y])))
        # hx = np.vstack((rowsXY, rows0, x * X, x * Y, x))
        # hy = np.vstack((rows0, rowsXY, y * X, y * Y, y))
        # h=np.hstack((hx,hy))
        # U,_,VT=np.linalg.svd(-h.T)
        ################################  错了 H{t} = reshape(V(:,8),3,3) ;%把V的第9列转置，再重新组合成3*3的矩阵
        # u=VT
        # v = np.reshape(u[:,8], (3, 3))
        # origin = A
        # dest = B
        nbr_correspondences = 4
        # a是构造出来的Ax=0里面的A矩阵，之后往里面填充数据。
        a = np.zeros((2 * nbr_correspondences, 9))
        for i in range(nbr_correspondences):
            a[2 * i] = [-origin[i][0], -origin[i][1], -1, 0, 0, 0, dest[i][0] * origin[i][0], dest[i][0] * origin[i][1],
                        dest[i][0]]
            a[2 * i + 1] = [0, 0, 0, -origin[i][0], -origin[i][1], -1, dest[i][1] * origin[i][0],
                            dest[i][1] * origin[i][1], dest[i][1]]
        # 用SVD去解方程
        u, s, v = np.linalg.svd(a)
        # 下面可以用Opencv里的去验证
        # h = cv2.findHomography(origin, dest)
        homography_matrix = v[8].reshape((3, 3))
        # 归一化之后就是最后的单应矩阵了
        homography_matrix = homography_matrix / homography_matrix[2, 2]
        # 得到其次左边的形式
        b=np.ones(pin.shape[0])
        homopin=np.c_[pin,b]
        homopout = np.c_[pout, b]
        scores=0
        for i in range(len(homopin)):
            tem=homography_matrix.dot(homopin[i])
            # print(np.linalg.norm(homopout[i] - tem))
            scores = scores+np.linalg.norm(homopout[i] - tem)
        if scores<minScore:
            minScore=scores
            finalN=homography_matrix
    return finalN

# 打印一下单应矩阵，没啥用
def printH(h):
    for i in range(8):
        temp=''
        for j in range(9):
            temp=temp+str(h.T[i][j])+' '
        print(temp)


#
