# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.parallel

# conditional DCGAN model, fully convolutional architecture
class _CnetG_1(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf, num_classes):
        super(_CnetG_1, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        
        self.inital_1 = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.inital_2 = nn.Sequential(
            nn.ConvTranspose2d(num_classes, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.conv_transpose1 = self.conv_transpose_bn_lrelu(self.ngf * 8, self.ngf * 4)
        self.conv_transpose2 = self.conv_transpose_bn_lrelu(self.ngf * 4, self.ngf * 2)
        self.conv_transpose3 = self.conv_transpose_bn_lrelu(self.ngf * 2, self.ngf)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            )

    def conv_transpose_bn_lrelu(self, in_channels, out_channels):
        layers = []
        
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
        
    def forward(self, input, label):
        x = self.inital_1(input)
        y = self.inital_2(label)
        x = torch.cat([x, y], 1)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv_transpose3(x)
        x = self.out(x)
        return x

class _CnetD_1(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf, num_classes):
        super(_CnetD_1, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.nc, int(self.ndf / 2), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
            
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_classes, int(self.ndf / 2), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv2 = self.conv_lrelu(self.ndf, self.ndf * 2) # (ndf*2) x 16 x 16
        self.conv3 = self.conv_lrelu(self.ndf * 2, self.ndf * 4) # (ndf*4) x 8 x 8
        self.conv4 = self.conv_lrelu(self.ndf * 4, self.ndf * 8) # (ndf*8) x 4 x 4
        self.out = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_lrelu(self, in_channels, out_channels):
        layers = []
        
        layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
        # layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(0.4))
        
        return nn.Sequential(*layers)
        
    def forward(self, input, label):
        x = self.conv1_1(input)
        y = self.conv1_2(label)
        x = torch.cat([x, y], 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)        
        x = self.out(x)
        x = self.sigmoid(x)
        return x.view(-1)