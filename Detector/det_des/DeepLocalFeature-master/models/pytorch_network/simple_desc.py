import torch
from torch import nn
from torch.nn import functional as f

class Descriptor(nn.Module):

    def __init__(self,
            out_dim=128,init_num_channels=64,
            num_conv_layers=3,use_bias=False,
            conv_ksize=3):
        super(Descriptor, self).__init__()
        in_channel=2
        channels_list = [init_num_channels * 2 ** i for i in range(num_conv_layers)]

        self.conv1 = nn.Conv2d(in_channel, channels_list[0], kernel_size=conv_ksize, stride=2,padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(channels_list[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels_list[0], channels_list[1], kernel_size=conv_ksize, stride=2,padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(channels_list[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels_list[1], channels_list[2], kernel_size=conv_ksize, stride=2, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(channels_list[2])
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512 * 4, 512)
        self.fc2 = nn.Linear(512, out_dim)

        # ori_maps = f.normalize(ori_maps, dim=-1)

    def forward(self, x):
        residual = x

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out