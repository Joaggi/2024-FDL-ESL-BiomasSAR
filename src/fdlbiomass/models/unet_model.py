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
    

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels=1, use_batchnorm=True, dropout_rate=0.0, z_collapse_method='conv'):
        super(UNet3D, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 64, use_batchnorm, dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder2 = DoubleConv(64, 128, use_batchnorm, dropout_rate)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.encoder3 = DoubleConv(128, 256, use_batchnorm, dropout_rate)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.bottleneck = DoubleConv(256, 512, use_batchnorm, dropout_rate)
        
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.decoder3 = DoubleConv(512, 256, use_batchnorm, dropout_rate)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder2 = DoubleConv(256, 128, use_batchnorm, dropout_rate)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.decoder1 = DoubleConv(128, 64, use_batchnorm, dropout_rate)
        
        self.z_collapse_method = z_collapse_method

        if z_collapse_method == 'conv':
            self.final_conv = nn.Conv3d(64, out_channels, kernel_size=(1, 1, 36))
        elif z_collapse_method == 'global_avg_pool':
            self.global_avg_pool = nn.AdaptiveAvgPool3d((None, None, 1))
            self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        elif z_collapse_method == 'progressive_z':
            self.final_conv1 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.final_conv2 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.final_conv3 = nn.Conv3d(16, 8, kernel_size=(3, 3, 3),  padding=(1, 1, 1))
            self.final_conv4 = nn.Conv3d(8, out_channels, kernel_size=(1, 1, 36))
        self.final_activation = nn.ReLU()

    def forward(self, x):
        # print('x ', x.shape)
        input_slice = x[:, :, :, x.shape[3]//2 , : ].detach()
        # print('Input x slice shape ', x.shape)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        reconstructed_slice = dec1[:, :, :, dec1.shape[3] //2 , :].detach()
        
        # output = self.final_conv(dec1)
        # output = self.final_activation(output)
        
        # output = torch.squeeze(output, dim=4)
        # output = torch.squeeze(output, dim=1)

        if self.z_collapse_method == 'conv':
            output = self.final_conv(dec1)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)
        elif self.z_collapse_method == 'global_avg_pool':
            output = self.global_avg_pool(dec1)
            output = self.final_conv(output)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)
        elif self.z_collapse_method == 'progressive_z':
            output = self.final_conv1(dec1)
            output = self.final_conv2(output)
            output = self.final_conv3(output)
            output = self.final_conv4(output)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)

        return output, input_slice, reconstructed_slice

    def __str__(self):
        return f"{self.__class__.__name__}_{self.z_collapse_method}"
    



class UNet3D_MP(nn.Module):
    def __init__(self, in_channels, out_channels=1, use_batchnorm=True, dropout_rate=0.0, z_collapse_method='conv'):
        super(UNet3D_MP, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 64, use_batchnorm, dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(64, 128, use_batchnorm, dropout_rate)
        # self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(128, 256, use_batchnorm, dropout_rate)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.bottleneck = DoubleConv(256, 512, use_batchnorm, dropout_rate)
        
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2,  )
        self.decoder3 = DoubleConv(512, 256, use_batchnorm, dropout_rate)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=(2,2,1), stride=(2,2,1)   )
        self.decoder2 = DoubleConv(256, 128, use_batchnorm, dropout_rate)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2,   )
        self.decoder1 = DoubleConv(128, 64, use_batchnorm, dropout_rate)
        
        self.z_collapse_method = z_collapse_method

        if z_collapse_method == 'conv':
            self.final_conv = nn.Conv3d(64, out_channels, kernel_size=(1, 1, 36))
        elif z_collapse_method == 'global_avg_pool':
            self.global_avg_pool = nn.AdaptiveAvgPool3d((None, None, 1))
            self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        elif z_collapse_method == 'progressive_z':
            self.final_conv1 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.final_conv2 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.final_conv3 = nn.Conv3d(16, 8, kernel_size=(3, 3, 3),  padding=(1, 1, 1))
            self.final_conv4 = nn.Conv3d(8, out_channels, kernel_size=(1, 1, 36))
        self.final_activation = nn.ReLU()

    def forward(self, x):

        # print(x.shape)

        input_slice = x[:, :, :, x.shape[3]//2, :].detach()
        
        enc1 = self.encoder1(x)
        # print('enc 1 ', enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        # print('enc 2 ', enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        # print('dec3 shape ', dec3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        # print('dec2 ', dec2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        reconstructed_slice = dec1[:, :, :, dec1.shape[3]//2, :].detach()

        if self.z_collapse_method == 'conv':
            output = self.final_conv(dec1)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)
        elif self.z_collapse_method == 'global_avg_pool':
            output = self.global_avg_pool(dec1)
            output = self.final_conv(output)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)
        elif self.z_collapse_method == 'progressive_z':
            output = self.final_conv1(dec1)
            output = self.final_conv2(output)
            output = self.final_conv3(output)
            output = self.final_conv4(output)
            output = self.final_activation(output)
            output = torch.squeeze(output, dim=4)
            output = torch.squeeze(output, dim=1)

        return output, input_slice, reconstructed_slice

    def __str__(self):
        return f"{self.__class__.__name__}_{self.z_collapse_method}"