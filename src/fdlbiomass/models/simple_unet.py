import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    
class simple_UNet3D2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_batchnorm=True, dropout_rate=0.0, z_collapse_method='conv', final_z_dim=36):
        super(simple_UNet3D2, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 32, use_batchnorm, dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv(32, 64, use_batchnorm, dropout_rate)
        
        self.bottleneck = DoubleConv(64, 128, use_batchnorm, dropout_rate)
        
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64, use_batchnorm, dropout_rate)
        
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(64, 32, use_batchnorm, dropout_rate)

        self.z_collapse_method = z_collapse_method

        self.final_z_dim = final_z_dim


        if z_collapse_method == 'conv':
            self.final_conv = nn.Conv3d(32, out_channels, kernel_size=(1,1,final_z_dim))
        elif z_collapse_method == 'global_avg_pool':
            self.global_avg_pool = nn.AdaptiveAvgPool3d((None, None, 1))
            self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        elif z_collapse_method == 'progressive_z':
            self.final_conv1 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.final_conv2 = nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.final_conv3 = nn.Conv3d(8, out_channels, kernel_size=(1, 1, final_z_dim))

        self.final_activation = nn.ReLU()

    def forward(self, x):
        
        # Handle 3D input by adding a channel dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        
        # Adjust the slice extraction for both 3D and 4D inputs
        input_slice = x[:, :, :, x.shape[3]//2, :].detach() if x.dim() == 5 else x[:, :, x.shape[2]//2, :].detach()

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool1(enc2))

        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.decoder1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)
        
        # Adjust the slice extraction for both 3D and 4D inputs
        reconstructed_slice = dec2[:, :, :, dec2.shape[3]//2, :].detach() if dec2.dim() == 5 else dec2[:, :, dec2.shape[2]//2, :].detach()

        if self.z_collapse_method == 'conv':
            output = self.final_conv(dec2)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)
        elif self.z_collapse_method == 'global_avg_pool':
            output = self.global_avg_pool(dec2)
            output = self.final_conv(output)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)
        elif self.z_collapse_method == 'progressive_z':
            output = self.final_conv1(dec2)
            output = self.final_conv2(output)
            output = self.final_conv3(output)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)

        return output, input_slice, reconstructed_slice

    def __str__(self):
        return f"{self.__class__.__name__}_{self.z_collapse_method}_z{self.final_z_dim}"