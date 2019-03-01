import numpy as np

# DLT方法求单应给定四个或者更多对应点
# input : four or more homogeneous coordinates which is a 3*n numpy array -- fp
# output : homography matrix
def H_from_points(fp,tp):
    if fp.shape != tp.shape:
        raise RuntimeError("number not match")

    # normalize
    m=np.mean(fp[:2],axis=1)
    maxstd=np.max(np.std(fp[:2],axis=1))+1e-9
    C1=np.diag([1/maxstd,1/maxstd,1])
    C1[0][2]=-m[0]/maxstd
    C1[1][2]=-m[1]/maxstd
    fp=np.dot(C1,fp)

    m=np.mean(tp[:2],axis=1)
    maxstd=np.max(np.std(tp[:2],axis=1))+1e-9
    C2=np.diag([1/maxstd,1/maxstd,1])
    C2[0][2]=-m[0]/maxstd
    C2[1][2]=-m[1]/maxstd
    tp=np.dot(C2,tp)

    nbr_correspondences = fp.shape[1]
    # a是构造出来的Ax=0里面的A矩阵，之后往里面填充数据。
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i], tp[0][i] * fp[1][i],
                    tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i],
                        tp[1][i] * fp[1][i], tp[1][i]]
    # 用SVD去解方程
    U,S,V = np.linalg.svd(A)

    H=V[np.argmin(S)].reshape((3,3))

    # anti normalize
    H = np.dot(np.linalg.inv(C2),np.dot(H,C1))
    return H/H[2,2]


# 放着变换
# input : four or more homogeneous coordinates which is a 3*n numpy array -- fp
# output : homography matrix
def Haffine_from_points(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError("number not match")

    # normalize
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = np.dot(C1, fp)

    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy()
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = np.dot(C2, tp)

    A = np.concatenate((fp_cond[:2],tp[:2]),axis=0)
    # 用SVD去解方程
    U, S, V = np.linalg.svd(A.T)

    tmp=V[:2].T
    B=tmp[:2]
    C=tmp[2:4]

    tmp2=np.concatenate((np.dot(C,np.linalg.pinv(B)),np.zeros((2,1))),axis=1)
    H=np.vstack((tmp2,[0,0,1]))
    # anti normalize
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2, 2]



objpoints = []  # 在世界坐标系中的三维点
imgpoints = []
h_all=[]
for corners in imgpoints:
    b = np.ones(corners.shape[0])
    y = np.expand_dims(b, axis=0)
    d = np.c_[corners.squeeze(),y.T]
    h_all.append(H_from_points(objpoints[0].T),d.T )
    # h_all.append(H_from_points(d.T, objpoints[0].T))