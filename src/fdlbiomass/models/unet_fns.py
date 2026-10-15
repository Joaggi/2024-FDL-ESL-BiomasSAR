import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        layers = []
        for _ in range(2):  # Two convolution operations
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'))
            if use_batchnorm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout3d(p=dropout_rate))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
    