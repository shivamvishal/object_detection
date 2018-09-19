import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


# utility
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# 1*1
#  3*3
# 5*5

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1,stride=2)

        self.branch3x3_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(48, 64, kernel_size=5, padding=2,stride=2)

        self.branch5x5dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch5x5dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1,stride=2)

        self.branch_pool_1 = BasicConv2d(in_channels, 32, kernel_size=1, stride=2)
        self.branch_pool_2 = BasicConv2d(32, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5dbl_1(x)
        branch5x5 = self.branch5x5dbl_2(branch5x5)
        branch5x5 = self.branch5x5dbl_3(branch5x5)

        branch3x3dbl = self.branch3x3_1(x)
        branch3x3dbl = self.branch3x3_2(branch3x3dbl)

        branch_pool = self.branch_pool_1(x)
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=3, stride=1, padding=1)  # TODO avg_pool2d
        branch_pool = self.branch_pool_2(branch_pool)

        # print(branch1x1.size())
        # print(branch5x5.size())
        # print(branch3x3dbl.size())
        # print(branch_pool.size())

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        # print(outputs.size())
        return torch.cat(outputs, 1)
