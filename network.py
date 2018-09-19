import torch

import torch.nn as nn
import torch.nn.functional as F

from inception import InceptionA


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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # convolution
        self.Conv2d_1a_7x7 = BasicConv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.Conv2d_3a_3x3 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        # inception
        self.Mixed_4a = InceptionA(128, 32)
        self.Mixed_5a = InceptionA(256, 64)

        # hypernet
        self.deconv = nn.ConvTranspose2d(in_channels=288, out_channels=288, kernel_size=4, stride=2, padding=1)
        self.conv_avgpool = BasicConv2d(672, 512, kernel_size=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # TODO verify 128 + 144 + 256,

        # fully connected
        self.fc1 = nn.Linear(512 * 33 * 20, 240)  # TODO dimension ??
        self.fc2 = nn.Linear(240, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        # convolution

        # 1056*640*3
        x = self.pool1(F.relu(self.Conv2d_1a_7x7(x)))

        # 528*320*32
        x = self.pool1(F.relu(self.Conv2d_2a_3x3(x)))

        # 264*160*64
        x = self.pool1(F.relu(self.Conv2d_3a_3x3(x)))

        # 132*80*128
        output_conv = x

        # inception
        x = self.Mixed_4a(x)

        # 66*40*256
        output_inceptionA = x
        x = self.Mixed_5a(x)

        # 33*20*288
        output_inceptionB = x

        # hypernet
        hyperin_conv = self.pool2(output_conv)  # maxpooling from conv layer
        # 66*40*128
        hyperin_deconv = self.deconv(output_inceptionB)

        # 66*40*288
        hyperin = output_inceptionA  # inception layer1 output
        # 66*40*256
        hyper_output = [hyperin_conv, hyperin_deconv, hyperin]


        hyper_output = torch.cat(hyper_output, 1)
        # 66*40*672
        hyper_output = self.conv_avgpool(hyper_output)

        hyper_avg_pool = self.avgpool(hyper_output)

        # 66*40*512

        # fully connected layer
        print(hyper_avg_pool.size())
        z = hyper_avg_pool.view(-1, 512 * 33 * 20)
        print(z.size())
        z = F.relu(self.fc1(z))
        print(z.size())
        z = self.fc2(z)
        print(z.size())
        z = self.fc3(z)
        print(z)
        return z

# net = Net()
