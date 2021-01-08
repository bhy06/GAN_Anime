import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# GAN Discriminator
class _netD_0(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf):
        super(_netD_0, self).__init__()
        self.ndf = ndf
        self.fc = nn.Sequential(
            nn.Linear(ndf * 64 * 3, ndf * 16 * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(ndf * 16 * 4, ndf * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(ndf * 16, ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(ndf * 4, 1), 
            nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x.view(-1, self.ndf * 64 * 3))
        return x.view(-1, 1)


# GAN Generator
class _netG_0(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf):
        super(_netG_0, self).__init__()
        self.nz = nz
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(ngf * 4, ngf * 16), 
            nn.ReLU(True), 
            nn.Dropout(0.5),
            nn.Linear(ngf * 16, ngf * 16 * 4), 
            nn.ReLU(True), 
            nn.Dropout(0.5),
            nn.Linear(ngf * 16 * 4, ngf * 64 * 3), 
            nn.Tanh())

    def forward(self, x):
        x = self.fc(x.view(-1,self.nz))
        return x.view(-1,3,64,64)