import torch
import torch.nn as nn

class SimpleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(SimpleConv3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=(1, 1, 36))
        self.final_activation = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        output = self.final_conv(x)
        output = self.final_activation(output)
        
        output = torch.squeeze(output, dim=4)
        return output

    def __str__(self):
        return f"{self.__class__.__name__}"