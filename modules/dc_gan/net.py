import torch
import torch.nn as nn
import torch.nn.functional as F

"""
GENERATOR
"""
class Generator(nn.Module):
    def __init__(self, z_dim, img_size, ngf = 64):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.img_size = img_size
        self.img_channel = 3
        self.ngf = ngf # number generator init feature

        self.init_size = self.img_size // 4

        self.conv1 = nn.ConvTranspose2d(self.z_dim, self.ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf * 8)

        self.conv2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf * 8)

        self.conv3 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ngf * 4)

        self.conv4 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ngf * 4)

        self.conv5 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.ngf * 2)

        self.conv6 = nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False)
        #self.bn6 = nn.BatchNorm2d(self.ngf) # Dont use batch norm in the last layer of generator !!
        self.bn6 = lambda x: x

        self.conv7 = nn.ConvTranspose2d(self.ngf, self.img_channel, 3, 1, 1, bias=False)

    def forward(self, x, negative_slope=0.1):
        """
        :param x: of size (b, z)
        :return:
        """
        b,z = x.size()

        # (b,z) -> (b,z,1,1)
        x = x.view(b, z, 1, 1)

        # (_,_,4,4)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=negative_slope)

        # (_,_,8,8)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=negative_slope)

        # (_,_,16,16)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=negative_slope)

        # (_,_,32,32)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=negative_slope)

        # (_,_,64,64)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=negative_slope)

        # (_,_,128,128)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=negative_slope)

        # (_,_,128,128), [-1., 1.]
        x = torch.tanh(self.conv7(x))

        return x

"""
DISCRIMINATOR
"""
class Discriminator(nn.Module):
    def __init__(self, ndf = 64):
        super(Discriminator, self).__init__()

        self.ndf = ndf # number discriminator feature

        self.conv1 = nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(self.ndf * 2)
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)

        self.bn3 = nn.BatchNorm2d(self.ndf * 4)
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)

        self.bn4 = nn.BatchNorm2d(self.ndf * 8)
        self.conv5 = nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False)

        self.bn5 = nn.BatchNorm2d(self.ndf * 16)
        self.conv6 = nn.Conv2d(self.ndf * 16, 1, 4, 1, 0, bias=False)

    def forward(self, x, negative_slop=0.1):
        """

        :param x: of size (b, 3, h, w)
        :return:
        """
        b,c,h,w = x.shape

        # (b,3,64,64)
        x = F.leaky_relu(self.conv1(x), negative_slope=negative_slop)

        # (b,_,32,32)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=negative_slop)

        # (b,_,16,16)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=negative_slop)

        # (b,_,8,8)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=negative_slop)

        # (b,_,4,4)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=negative_slop)

        # (b,_,1,1)
        x = torch.sigmoid(self.conv6(x))
        x = x.view(-1, 1)

        return x

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, path):
        torch.save(self.state_dict(), path)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

