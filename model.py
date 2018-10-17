import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Unet(nn.Module):
    """
    Unet architecture. Reference: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels=1):
        super(Unet, self).__init__()
        self.in_channels = in_channels

        self.conv1_0 = nn.Conv2d(1, 64, 3)
        self.conv1_1 = nn.Conv2d(64, 64, 3)
        self.down_sample = nn.MaxPool2d(2, stride=2)

        self.conv2_0 = nn.Conv2d(64, 128, 3)
        self.conv2_1 = nn.Conv2d(128, 128, 3)

        self.conv3_0 = nn.Conv2d(128, 256, 3)
        self.conv3_1 = nn.Conv2d(256, 256, 3)

        self.conv4_0 = nn.Conv2d(256, 512, 3)
        self.conv4_1 = nn.Conv2d(512, 512, 3)

        self.conv5_0 = nn.Conv2d(512, 1024，3)
        self.conv5_1 = nn.Conv2d(1024, 1024, 3)

        self.upconv_6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_0 = nn.Conv2d(1024, 512, 3)
        self.conv6_1 = nn.Conv2d(512, 512, 3)

        self.upconv_7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_0 = nn.Conv2d(512, 256, 3)
        self.conv7_1 = nn.Conv2d(256, 256, 3)

        self.upconv_8 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_0 = nn.Conv2d(256, 128, 3)
        self.conv8_1 = nn.Conv2d(128, 128, 3)

        self.upconv_9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_0 = nn.Conv2d(128, 64, 3)
        self.conv9_1 = nn.Conv2d(64, 64, 3)
        self.conv9_2 = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        pass