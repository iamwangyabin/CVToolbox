import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from thop import profile, clever_format

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MCNN(nn.Module):
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return x

net = MCNN()
flop, para = profile(net, (torch.randn((1, 1, 512, 512)),))








## Draw accuracy vs flops
# sudo apt install msttcorefonts -qq
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font',family='Times New Roman')

# MCNN      macs:5.550G params:127.953K
# Bayesian   macs:108G params:21.499M
# Cascaded-MTL  macs:31.174G params:2.454M
# Switching-CNN macs:xx params:15.11M
# CP-CNN
# ANF
# TEDNet   macs:287.688G params:8.078M
# CAN     macs: 114.831G params:18.103M
# our   # 3.901M


MAE = [50, 100, 150, 200, 250, 300]
MSE = [0.051,0.069,0.081,0.089,0.098,0.106]

plt.figure()





l1 = plt.plot(x, y1, color='black', linewidth=2.0, linestyle='-',   label = 'CF'    )
l2 = plt.plot(x, y2, color='red', linewidth=2.0, linestyle='-',     label = 'RCTR'  )
l3 = plt.plot(x, y3, color='yellow', linewidth=2.0, linestyle='-',  label = 'SLIM'  )
l4 = plt.plot(x, y4, color='blue', linewidth=2.0, linestyle='-',    label = 'SRMP'  )
l5 = plt.plot(x, y5, color='gray', linewidth=2.0, linestyle='-',    label = 'LSMC'  )
l6 = plt.plot(x, y6, color='green', linewidth=2.0, linestyle='-',   label = 'TPMF-CF'   )
l7 = plt.plot(x, y7, color='purple', linewidth=2.0, linestyle='-',  label = 'CRABSN'    )
l8 = plt.plot(x, y8, color='brown', linewidth=2.0, linestyle='-',   label = 'LSMF-PR-b' )
l9 = plt.plot(x, y9, color='pink', linewidth=2.0, linestyle='-',    label = 'LSMF-PR'   )

for i, j in zip(x, y1):
    plt.scatter(i, j, marker='>',color="black")
for i, j in zip(x, y2):
    plt.scatter(i, j, marker='o',color="red")
for i, j in zip(x, y3):
    plt.scatter(i, j, marker='s',color="yellow")
for i, j in zip(x, y4):
    plt.scatter(i, j, marker='*',color="blue")
for i, j in zip(x, y5):
    plt.scatter(i, j, marker='D',color="gray")
for i, j in zip(x, y6):
    plt.scatter(i, j, marker='+',color="green")
for i, j in zip(x, y7):
    plt.scatter(i, j, marker='x',color="purple")
for i, j in zip(x, y8):
    plt.scatter(i, j, marker='1',color="brown")
for i, j in zip(x, y9):
    plt.scatter(i, j, marker='^',color="pink")

plt.xlabel('N')
plt.ylabel('Recall@N')

plt.legend(labels=['CF','RCTR','SLIM','SRMP','LSMC','TPMF-CF','CRABSN','LSMF-PR-b','LSMF-PR'],  loc='best', fontsize=8, framealpha=1, fancybox=False)

plt.grid()
plt.title("Recall@N on Flixster")
plt.savefig("./4.pdf")