from __future__ import print_function
import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from torch.autograd import Variable

### load project files
import models.gan as gan
import models.dcgan as dcgan
import models.wdcgan as wdcgan
import models.wdcgan_gp as wdcgan_gp
import models.wresgan_gp as wresgan_gp
import models.cdcgan as cdcgan
import models.acgan_resnet as acgan_resnet

from models.gan import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--outDir', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--model', type=int, default=1, help='0 for gan, 1 for dcgan, 2 for wdcgan, 3 for wdcgan_gp, 4 for wresgan_gp, 5 for cdcgan, 6 for acgan_resnet')
parser.add_argument('--d_labelSmooth', type=float, default=0, help='for D, use soft label "1-labelSmooth" for real samples')

# =============================================================================
# # simply prefer this way
# arg_list = [
#     '--dataRoot', 'faces',
#     '--workers', '0',
#     '--batchSize', '64',
#     '--imageSize', '64',
#     '--nc', '3',
#     '--nz', '100',
#     '--ngf', '64',
#     '--ndf', '64',
#     '--niter', '100',
#     '--lr', '0.0001',
#     '--beta1', '0.5', 
#     '--cuda', 
#     '--ngpu', '1',
#     '--netG', '',
#     '--netD', '',
#     '--clamp_lower', '-0.01',
#     '--clamp_upper', '0.01',
#     '--Diters', '1',
#     '--outDir', './results',
#     '--model', '1',
#     '--d_labelSmooth', '0.1', 
# ]
# =============================================================================

# cgan class number
num_classes = 20

opt = parser.parse_args()
# opt = parser.parse_args(arg_list)
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass

opt.manualSeed = random.randint(1,10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

nc = int(opt.nc)
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

LAMBDA = 10 # Gradient penalty lambda hyperparameter

dataset = dset.ImageFolder(
    root=opt.dataRoot,
    transform=transforms.Compose([
            transforms.Resize(opt.imageSize),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=opt.workers)

# load models 
if opt.model == 0:
    netG = gan._netG_0(ngpu, nz, nc, ngf)
    netD = gan._netD_0(ngpu, nz, nc, ndf)
elif opt.model == 1:
    netG = dcgan._netG_0(ngpu, nz, nc, ngf)
    netD = dcgan._netD_0(ngpu, nz, nc, ndf)
elif opt.model == 2:
    netG = wdcgan._netG_0(ngpu, nz, nc, ngf)
    netD = wdcgan._netD_0(ngpu, nz, nc, ndf)
elif opt.model == 3:
    netG = wdcgan_gp._netG_0(ngpu, nz, nc, ngf)
    netD = wdcgan_gp._netD_0(ngpu, nz, nc, ndf)
elif opt.model == 4:
    netG = wresgan_gp.GoodGenerator(64, 64*64*3)
    netD = wresgan_gp.GoodDiscriminator(64)
elif opt.model == 5:
    netG = cdcgan._CnetG_1(ngpu, nz, nc, ngf, num_classes)
    netD = cdcgan._CnetD_1(ngpu, nz, nc, ndf, num_classes)
elif opt.model == 6:
    netG = acgan_resnet.GoodGenerator(64, 64*64*3)
    netD = acgan_resnet.GoodDiscriminator(64, num_classes)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()
criterion_CE = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

# initialize fixed noise with different shapes for different models
if opt.model == 4:
    fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_(0, 1)
else:
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)

fixed_label = torch.randint(0, num_classes - 1, (opt.batchSize,))
fixed_label_onehot = torch.eye(num_classes)[fixed_label]
fixed_label_onehot = fixed_label_onehot.view(opt.batchSize, num_classes, 1, 1)

onehot = torch.eye(num_classes).view(num_classes, num_classes, 1, 1)
fill = torch.zeros([num_classes, num_classes, opt.imageSize, opt.imageSize])
for i in range(num_classes):
    fill[i, i, :, :] = 1

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterion_MSE.cuda()
    criterion_CE.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    fixed_label_onehot = fixed_label_onehot.cuda()
    
input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_label_onehot = Variable(fixed_label_onehot)

# metrics output
metric = []

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

# use an exponentially decaying learning rate
scheduler_D = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
scheduler_G = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

def calc_gradient_penalty(netD, real_data, fake_data):
    # print("real_data: ", real_data.size(), fake_data.size())
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), int(real_data.nelement()/real_data.size(0))).contiguous().view(real_data.size(0), 3, 64, 64)
    alpha = alpha.cuda() 

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    
    if opt.model == 6:
        disc_interpolates, _ = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def gen_rand_noise_with_label(label, batchSize):
    if label is None:
        label = np.random.randint(0, num_classes, batchSize)
    #attach label into noise
    noise = np.random.normal(0, 1, (batchSize, 128))   
    prefix = np.zeros((batchSize, num_classes))
    prefix[np.arange(batchSize), label] = 1
    noise[np.arange(batchSize), :num_classes] = prefix[np.arange(batchSize)]

    noise = torch.from_numpy(noise).float()
    noise = noise.cuda()

    return noise

# label for acgan input
fixed_label2 = []
for c in range(opt.batchSize):
    fixed_label2.append(c%num_classes)
fixed_noise2 = gen_rand_noise_with_label(fixed_label2, opt.batchSize)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        
        #GAN/DCGAN:
        if opt.model == 0 or opt.model == 1:
            start_iter = time.time()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_data, _ = data
            batch_size = real_data.size(0)
            input.resize_(real_data.size()).copy_(real_data)
            label.resize_(batch_size).fill_(real_label - opt.d_labelSmooth) # use smooth label for discriminator

            output = netD(input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()
            # train with fake
            noise.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.fill_(fake_label)
            output = netD(fake.detach()) # add ".detach()" to avoid backprop through G
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step() 

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label) 
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward() 
            D_G_z2 = output.data.mean()
            optimizerG.step()
        
            end_iter = time.time()
            
        #WGAN
        if opt.model == 2:
            start_iter = time.time()
            ############################
            # (1) Update D network: maximize -D(x) + D(G(z))
            ###########################
            for _ in range(opt.Diters):
                
                netD.zero_grad()
                real_data, _ = data
                batch_size = real_data.size(0)
                input.resize_(real_data.size()).copy_(real_data)
               
                output_real = netD(input)
                D_x = output_real.data.mean()

                noise.resize_(batch_size, nz, 1, 1)
                noise.data.normal_(0, 1)
                fake = netG(noise)
                output_fake = netD(fake.detach()) # add ".detach()" to avoid backprop through G
                D_G_z1 = output_fake.data.mean()
                
                errD = -torch.mean(output_real) + torch.mean(output_fake)
                
                wasserstein_D = torch.mean(output_real) - torch.mean(output_fake)
                
                errD.backward()
                optimizerD.step()

                # Clip weights of D network
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                    
            ############################
            # (2) Update G network: maximize -D(G(z))
            ###########################
            netG.zero_grad()
            noise.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output_fake = netD(fake) 
            errG = -torch.mean(output_fake)
            errG.backward() 
            D_G_z2 = output_fake.data.mean()
            optimizerG.step()
        
            end_iter = time.time()
        
        # WGAN-GP
        elif opt.model == 3:
            start_iter = time.time()
            ############################
            # (1) Update D network: maximize -D(x) + D(G(z))
            ###########################
            for _ in range(opt.Diters):
                
                netD.zero_grad()
                real_data, _ = data
                batch_size = real_data.size(0)
                input.resize_(real_data.size()).copy_(real_data)
               
                output_real = netD(input)
                D_x = output_real.data.mean()

                noise.resize_(batch_size, nz, 1, 1)
                noise.data.normal_(0, 1)               
                fake = netG(noise)
                output_fake = netD(fake.detach()) # add ".detach()" to avoid backprop through G
                D_G_z1 = output_fake.data.mean()
                
                gradient_penalty = calc_gradient_penalty(netD, input.data, fake.data)
                
                errD = -torch.mean(output_real) + torch.mean(output_fake) + gradient_penalty
                
                wasserstein_D = torch.mean(output_real) - torch.mean(output_fake)
                
                errD.backward()
                optimizerD.step() 
                    
            ############################
            # (2) Update G network: maximize -D(G(z))
            ###########################
            netG.zero_grad()
            noise.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output_fake = netD(fake) 
            errG = -torch.mean(output_fake)
            errG.backward() 
            D_G_z2 = output_fake.data.mean()
            optimizerG.step()
        
            end_iter = time.time()
          
        # WResGAN-GP:
        elif opt.model == 4: 
            start_iter = time.time()
            ############################
            # (1) Update D network: maximize -D(x) + D(G(z))
            ###########################
            for _ in range(opt.Diters):
                    
                netD.zero_grad()
                real_data, _ = data
                batch_size = real_data.size(0)
                input.resize_(real_data.size()).copy_(real_data)
                   
                output_real = netD(input)
                D_x = output_real.data.mean()
    
                noise.resize_(batch_size, nz)
                noise.data.normal_(0, 1)               
                fake = netG(noise)
                output_fake = netD(fake.detach()) # add ".detach()" to avoid backprop through G
                D_G_z1 = output_fake.data.mean()
                    
                gradient_penalty = calc_gradient_penalty(netD, input.data, fake.data)
                    
                errD = -torch.mean(output_real) + torch.mean(output_fake) + gradient_penalty
                    
                wasserstein_D = torch.mean(output_real) - torch.mean(output_fake)
                    
                errD.backward()
                optimizerD.step()
                        
            ############################
            # (2) Update G network: maximize -D(G(z))
            ###########################
            netG.zero_grad()
            noise.resize_(batch_size, nz)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output_fake = netD(fake) 
            errG = -torch.mean(output_fake)
            errG.backward()
            D_G_z2 = output_fake.data.mean()
            optimizerG.step()

            end_iter = time.time()
            
        # cDCGAN
        elif opt.model == 5:
            start_iter = time.time()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_data, real_labels = data
            batch_size = real_data.size(0)
            
            input.resize_(real_data.size()).copy_(real_data)
            label.resize_(batch_size).fill_(real_label - opt.d_labelSmooth) # use smooth label for discriminator
            
            real_labels_input = Variable(fill[real_labels]).cuda()        
            output = netD(input, real_labels_input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()
            
            # train with fake
            noise.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            label_random = (torch.rand(batch_size, 1) * num_classes).type(torch.LongTensor).squeeze()
            labelG_input = Variable(onehot[label_random]).cuda()
            labelD_input = Variable(fill[label_random]).cuda()
                       
            fake = netG(noise, labelG_input)
            label.data.fill_(fake_label)        
            output = netD(fake.detach(), labelD_input)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            noise.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            label_random = (torch.rand(batch_size, 1) * num_classes).type(torch.LongTensor).squeeze()
            labelG_input = Variable(onehot[label_random]).cuda()
            labelD_input = Variable(fill[label_random]).cuda()
            
            fake = netG(noise, labelG_input)
            label.data.fill_(real_label)
            output = netD(fake, labelD_input)
            errG = criterion(output, label)
            errG.backward() 
            D_G_z2 = output.data.mean()
            optimizerG.step()
            
            end_iter = time.time()
        
        # ACGAN
        elif opt.model == 6:
            start_iter = time.time()
            
            real_data, real_labels = data
            real_data = real_data.cuda()
            real_labels = real_labels.cuda()
            batch_size = real_data.size(0)
            
            ############################
            # (1) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False
                
            netG.zero_grad()           
            
            f_label = np.random.randint(0, num_classes, batch_size)
            noise_AC = gen_rand_noise_with_label(f_label, batch_size)
            noise_AC.requires_grad_(True)        

            fake = netG(noise_AC)            
            output_fake, output_fake_label = netD(fake)    
            
            aux_label = torch.from_numpy(f_label).long()
            aux_label = aux_label.cuda()        
            
            aux_errG = criterion_CE(output_fake_label, aux_label).mean()
    
            errG = -torch.mean(output_fake) + aux_errG
            errG.backward() 
            
            D_G_z2 = output_fake.data.mean()        
            optimizerG.step()
                    
            ############################
            # (2) Update D network
            ###########################
            for p in netD.parameters():
                p.requires_grad = True
  
            for _ in range(opt.Diters):                  
                netD.zero_grad()
         
                input.resize_(real_data.size()).copy_(real_data)            
                
                output_real, output_real_label = netD(input) 
                aux_errD = criterion_CE(output_real_label, real_labels).mean()
                
                D_x = -output_real.data.mean()

                f_label = np.random.randint(0, num_classes, batch_size)
                noise_AC = gen_rand_noise_with_label(f_label, batch_size)
                noise_AC.requires_grad_(False)            
                
                # calculate label loss for fake
                fake = netG(noise_AC)                
                output_fake, output_fake_label = netD(fake.detach()) # add ".detach()" to avoid backprop through G                
                D_G_z1 = output_fake.data.mean()             

                gradient_penalty = calc_gradient_penalty(netD, input.data, fake.data)  
                             
                errD = -torch.mean(output_real) + torch.mean(output_fake) + gradient_penalty + aux_errD            
                
                wasserstein_D = torch.mean(output_real) - torch.mean(output_fake)
                
                errD.backward()
                optimizerD.step() # .step() can be called once the gradients are computed
                    
            end_iter = time.time()                   
        
        if opt.model == 0 or opt.model == 1 or opt.model == 5:
            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
                      % (epoch, opt.niter, i, len(dataloader), errD.data, errG.data, D_x, D_G_z1, D_G_z2, end_iter-start_iter))
                metric.append('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
                      % (epoch, opt.niter, i, len(dataloader), errD.data, errG.data, D_x, D_G_z1, D_G_z2, end_iter-start_iter))
        else:
            if i % 100 == 0:
                print('[%d/%d][%d/%d] W_distance: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
                      % (epoch, opt.niter, i, len(dataloader), wasserstein_D.data, errD.data, errG.data, D_x, D_G_z1, D_G_z2, end_iter-start_iter))
                metric.append('[%d/%d][%d/%d] W_distance: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
                      % (epoch, opt.niter, i, len(dataloader), wasserstein_D.data, errD.data, errG.data, D_x, D_G_z1, D_G_z2, end_iter-start_iter))
            
        if i % 100 == 0:
            # the first 64 samples from the mini-batch are saved.
            vutils.save_image(real_data[0:64,:,:,:],
                    '%s/real_samples.png' % opt.outDir, nrow=8)
            
            if opt.model == 5:
                fake = netG(fixed_noise, fixed_label_onehot)
            elif opt.model == 6:
                fake = netG(fixed_noise2)
            else:
                fake = netG(fixed_noise)
                
            vutils.save_image(fake.data[0:64,:,:,:],
                    '%s/fake_samples_epoch_%03d.png' % (opt.outDir, epoch), nrow=8)
            
    if (epoch+1) % 20 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outDir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outDir, epoch))
    
    # update learning rate    
    scheduler_D.step()
    scheduler_G.step()

# write metrics
with open(os.path.join(opt.outDir, 'output_epoch.txt'), 'w') as f:
    for item in metric:
        f.write("%s\n" % item)  
