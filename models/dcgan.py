import torch.nn as nn

# DCGAN model, fully convolutional architecture
class _netG_0(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf):
        super(_netG_0, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        
        self.inital = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False), # (ngf*8) x 4 x 4
            nn.BatchNorm2d(self.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.conv_transpose1 = self.conv_transpose_bn_lrelu(self.ngf * 8, self.ngf * 4) # (ngf*4) x 8 x 8
        self.conv_transpose2 = self.conv_transpose_bn_lrelu(self.ngf * 4, self.ngf * 2) # (ngf*2) x 16 x 16
        self.conv_transpose3 = self.conv_transpose_bn_lrelu(self.ngf * 2, self.ngf) # (ngf) x 32 x 32
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False), # (nc) x 64 x 64
            nn.Tanh()
            )

    def conv_transpose_bn_lrelu(self, in_channels, out_channels):
        layers = []
        
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
        
    def forward(self, input):
        x = self.inital(input)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv_transpose3(x)
        x = self.out(x)
        return x

class _netD_0(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf):
        super(_netD_0, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False), # (ndf) x 32 x 32
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
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)        
        x = self.out(x)
        x = self.sigmoid(x)
        return x.view(-1)
