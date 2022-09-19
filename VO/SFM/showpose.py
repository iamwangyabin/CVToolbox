import json
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def getReconData(filename):
    with open(filename,"r", encoding="utf-8") as f:
        cc = json.loads(f.read())
    data={}
    for c in range(len(cc)):
        for i in cc[c]["shots"]:
            data[i]={'rotation':cc[c]["shots"][i]["rotation"], 'translation': cc[c]["shots"][i]["translation"]}
    data_name=sorted(data.keys())
    return data,data_name

def optical_center(shot):
    R = cv2.Rodrigues(np.array(shot['rotation'], dtype=float))[0]
    t = shot['translation']
    return -R.T.dot(t)

def makeRotationAxis(axis,angle):
    c = np.cos(angle)
    s = np.cos(angle)
    t = 1 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    tx = t*x
    ty = t*y
    th = np.array([
        [tx * x + c, tx * y - s * z, tx * z + s * y, 0],
        [tx * y + s * z, ty * y + c, ty * z - s * x, 0],
        [tx * z - s * y, ty * z + s * x, t * z * z + c, 0],
        [0, 0, 0, 1]
    ])
    return th

def applyMatrix4(v,matrix):
    d = 1/(matrix[3][0] * v[0]+  matrix[3][1] *v[1]+ matrix[3][2]*v[2] + matrix[3][3])
    x = (matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2] +matrix[0][3])*d
    y = (matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2] +matrix[1][3])*d
    z = (matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2] +matrix[2][3])*d
    return np.array([x,y,z])

def rotate(vector, angleaxis):
    v = np.array([vector[0], vector[1], vector[2]])
    axis = np.array([angleaxis[0], angleaxis[1], angleaxis[2]])
    angle = np.linalg.norm(axis)
    axis = axis / np.linalg.norm(axis)
    # print(angle)
    matrix = makeRotationAxis(axis,angle)
    n_v = applyMatrix4(v,matrix)
    return n_v

def opticalCenter(shot):
    R = -np.array(shot['rotation'], dtype=float)
    t = np.array(shot['translation'], dtype=float)
    Rt = rotate(t, R)
    Rt = -Rt
    return Rt

def getWorldPosion(data,name):
    real_data={}
    for i in name:
        temp = opticalCenter(data[i])
        real_data[i] = temp
    return real_data

rawdataLF,nameLF=getReconData("./pano/reconstruction.json")


realdataLF = getWorldPosion(rawdataLF,nameLF)

def getWorldPosion(data,name):
    real_data={}
    for i in name:
        temp = optical_center(data[i])
        real_data[i] = temp
    return real_data

realdataLF = getWorldPosion(rawdataLF,nameLF)

# def getTSVData(filename):
#     f = open(filename,'r')
#     worldcoord={}
#     i=0
#     for line in f.readlines():
#         if i==0:
#             i+=1
#             continue
#         data=line.split()
#         worldcoord[data[0]] = [float(data[1]), float(data[2]), float(data[3])]
#         i+=1
#     return worldcoord

# worldcoord = getTSVData("./fr1/image_geocoords.tsv")


def showSFMresult(data1,name):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    x = []
    y = []
    z = []
    for key in name:
        value=data1[key]
        x.append( value[0] )
        y.append( value[1] )
        z.append( value[2] )
    ax.plot(x,y,z)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


showSFMresult(realdataLF,nameLF)


def show(aa,name):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in name:
        value = aa[i]
        ax.scatter(value[0], value[1], value[2], c='y')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

show(realdataLF,nameLF)

def showGroundtruth():
    f = open("./groundtruth.txt")
    x = []
    y = []
    z = []
    for line in f:
        if line[0] == '#':
            continue
        data = line.split()
        x.append( float(data[1] ) )
        y.append( float(data[2] ) )
        z.append( float(data[3] ) )
    ax = plt.subplot( 111, projection='3d')
    ax.plot(x,y,z)
    plt.show()

showGroundtruth()

def show(data1,name):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    x = []
    y = []
    z = []
    for key in name:
        value=data1[key]['translation']
        x.append( value[0] )
        y.append( value[1] )
        z.append( value[2] )
    ax.plot(x,y,z)
    # f = open("./groundtruth.txt")
    # x = []
    # y = []
    # z = []
    # for line in f:
    #     if line[0] == '#':
    #         continue
    #     data = line.split()
    #     x.append( float(data[1] ) )
    #     y.append( float(data[2] ) )
    #     z.append( float(data[3] ) )
    ax.plot(x,y,z)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


show(realdataLF,nameLF)

## name:list
##  从真实值找匹配值，并在找不到的地方放弃
def getGroundAlian(name):
    groundtruth={}
    f = open("./groundtruth.txt")
    for line in f:
        if line[0] == '#':
            continue
        data_ = line.split()
        groundtruth[data_[0][:13]]=[float(data_[1]), float(data_[2]), float(data_[3])]
    valueG={}
    new_name=[]
    for key in name:
        try:
            valueG[key]=groundtruth[key[:13]]
            new_name.append(key)
        except:
            print(key)
    return valueG,new_name

ground,new_name=getGroundAlian(nameLF)

def Change2Point(data,name):
    points=[]
    for i in name:
        point=data[i]
        points.append(point)
    points=np.array(points)
    return points

LFPoints=Change2Point(realdataLF,new_name)
GroundPoints=Change2Point(ground,new_name)

import transformations as tf

def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

s, A, b=align_reconstruction_naive_similarity(LFPoints.T, GroundPoints.T)


new_GroundPoints=s*A.dot(GroundPoints.T).T+b
new_LFPoints=s*A.dot(LFPoints.T).T+b

# just show points get from above
def show(data1,data2):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    x=data1[:,0]
    y=data1[:,1]
    z=data1[:,2]
    ax.plot(x,y,z,c='b')
    x=data2[:,0]
    y=data2[:,1]
    z=data2[:,2]
    ax.plot(x,y,z,c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

show(new_GroundPoints,LFPoints)

show(new_LFPoints,GroundPoints)


