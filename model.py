import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Unet(nn.Module):
    """
    Unet architecture. Reference: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels=1, out_channel=1):
        super(Unet, self).__init__()

        self.down_sample = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1_0 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv1_1 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_0 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4_0 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv5_0 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)

        # self.upconv_6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upconv_6 = nn.Upsample(scale_factor=2)
        self.conv6_0 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv6_1 = nn.Conv2d(256, 256, 3, padding=1)

        # self.upconv_7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv_7 = nn.Upsample(scale_factor=2)
        self.conv7_0 = nn.Conv2d(384, 128, 3, padding=1)
        self.conv7_1 = nn.Conv2d(128, 128, 3, padding=1)

        # self.upconv_8 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv_8 = nn.Upsample(scale_factor=2)
        self.conv8_0 = nn.Conv2d(192, 64, 3, padding=1)
        self.conv8_1 = nn.Conv2d(64, 64, 3, padding=1)

        # self.upconv_9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.upconv_9 = nn.Upsample(scale_factor=2)
        self.conv9_0 = nn.Conv2d(96, 32, 3, padding=1)
        self.conv9_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv9_2 = nn.Conv2d(32, out_channel, 1)

    def forward(self, x):

    	###################### Encoder ######################
        activation_1 = self.relu(self.conv1_0(x))
        activation_1 = self.relu(self.conv1_1(activation_1))

        activation_2 = self.down_sample(activation_1)
        activation_2 = self.relu(self.conv2_0(activation_2))
        activation_2 = self.relu(self.conv2_1(activation_2))

        activation_3 = self.down_sample(activation_2)
        activation_3 = self.relu(self.conv3_0(activation_3))
        activation_3 = self.relu(self.conv3_1(activation_3))

        activation_4 = self.down_sample(activation_3)
        activation_4 = self.relu(self.conv4_0(activation_4))
        activation_4 = self.relu(self.conv4_1(activation_4))

        activation_5 = self.down_sample(activation_4)
        activation_5 = self.relu(self.conv5_0(activation_5))
        activation_5 = self.relu(self.conv5_1(activation_5))

        ###################### Decoder ######################
        activation_6 = self.upconv_6(activation_5)
        # offset = (activation_4.shape[3] - activation_6.shape[3]) / 2
        concat_6 = torch.cat((activation_4, activation_6), dim=1)
        activation_6 = self.relu(self.conv6_0(concat_6))
        activation_6 = self.relu(self.conv6_1(activation_6))

        activation_7 = self.upconv_7(activation_6)
        # offset = (activation_3.shape[3] - activation_7.shape[3]) / 2
        concat_7 = torch.cat((activation_3, activation_7), dim=1)
        activation_7 = self.relu(self.conv7_0(concat_7))
        activation_7 = self.relu(self.conv7_1(activation_7))

        activation_8 = self.upconv_8(activation_7)
        # offset = (activation_2.shape[3] - activation_8.shape[3]) / 2
        concat_8 = torch.cat((activation_2, activation_8), dim=1)
        activation_8 = self.relu(self.conv8_0(concat_8))
        activation_8 = self.relu(self.conv8_1(activation_8))

        activation_9 = self.upconv_9(activation_8)
        # offset = (activation_1.shape[3] - activation_9.shape[3]) / 2
        concat_9 = torch.cat((activation_1, activation_9), dim=1)
        activation_9 = self.relu(self.conv9_0(concat_9))
        activation_9 = self.relu(self.conv9_1(activation_9))
        out = self.sigmoid(self.conv9_2(activation_9))

        return out


class Unet_3D(nn.Module):
    """
    Unet architecture. Reference: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels=3, out_channel=3):
        super(Unet_3D, self).__init__()

        self.down_sample = nn.MaxPool3d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1_0 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.bn1_0 = nn.BatchNorm3d(32)
        self.conv1_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm3d(64)


        self.conv2_0 = nn.Conv3d(64, 64, 3, padding=1)
        self.bn2_0 = nn.BatchNorm3d(64)
        self.conv2_1 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm3d(128)

        self.conv3_0 = nn.Conv3d(128, 128, 3, padding=1)
        self.bn3_0 = nn.BatchNorm3d(128)
        self.conv3_1 = nn.Conv3d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm3d(256)

        self.conv4_0 = nn.Conv3d(256, 256, 3, padding=1)
        self.bn4_0 = nn.BatchNorm3d(256)
        self.conv4_1 = nn.Conv3d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm3d(512)

        self.upconv_5 = nn.Upsample(scale_factor=2)
        self.conv5_0 = nn.Conv3d(768, 256, 3, padding=1)
        self.bn5_0 = nn.BatchNorm3d(256)
        self.conv5_1 = nn.Conv3d(256, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm3d(256)

        self.upconv_6 = nn.Upsample(scale_factor=2)
        self.conv6_0 = nn.Conv3d(384, 128, 3, padding=1)
        self.bn6_0 = nn.BatchNorm3d(128)
        self.conv6_1 = nn.Conv3d(128, 128, 3, padding=1)
        self.bn6_1 = nn.BatchNorm3d(128)

        self.upconv_7 = nn.Upsample(scale_factor=2)
        self.conv7_0 = nn.Conv3d(192, 64, 3, padding=1)
        self.bn7_0 = nn.BatchNorm3d(64)
        self.conv7_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.bn7_1 = nn.BatchNorm3d(64)

        self.conv8 = nn.Conv3d(64, out_channel, 1, padding=1)

    def forward(self, x):

        ###################### Encoder ######################
        activation_1 = self.relu(self.bn1_0(self.conv1_0(x)))
        activation_1 = self.relu(self.bn1_1(self.conv1_1(activation_1)))

        activation_2 = self.down_sample(activation_1)
        activation_2 = self.relu(self.bn2_0(self.conv2_0(activation_2)))
        activation_2 = self.relu(self.bn2_1(self.conv2_1(activation_2)))

        activation_3 = self.down_sample(activation_2)
        activation_3 = self.relu(self.bn3_0(self.conv3_0(activation_3)))
        activation_3 = self.relu(self.bn3_1(self.conv3_1(activation_3)))

        activation_4 = self.down_sample(activation_3)
        activation_4 = self.relu(self.bn4_0(self.conv4_0(activation_4)))
        activation_4 = self.relu(self.bn4_1(self.conv4_1(activation_4)))

        ###################### Decoder ######################
        activation_5 = self.upconv_5(activation_4)
        concat_5 = torch.cat((activation_3, activation_5), dim=1)
        activation_5 = self.relu(self.bn5_0(self.conv5_0(concat_5)))
        activation_5 = self.relu(self.bn5_1(self.conv5_1(activation_5)))

        activation_6 = self.upconv_6(activation_5)
        concat_6 = torch.cat((activation_2, activation_6), dim=1)
        activation_6 = self.relu(self.bn6_0(self.conv6_0(concat_6)))
        activation_6 = self.relu(self.bn6_1(self.conv6_1(activation_6)))

        activation_7 = self.upconv_7(activation_6)
        concat_7 = torch.cat((activation_1, activation_7), dim=1)
        activation_7 = self.relu(self.bn7_0(self.conv7_0(concat_7)))
        activation_7 = self.relu(self.bn7_1(self.conv7_1(activation_7)))

        out = self.conv8(activation_7)

        return out