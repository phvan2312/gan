import numpy as np
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable

from gan.loader import MotorbikeDataset
from gan.modules.dc_gan.net import Discriminator, Generator, weights_init
import gan.constants as constants

import gan.utils as utils

exp_prefix = "dc_gan_2"
os.makedirs("%s_debug_images" % exp_prefix, exist_ok=True)
os.makedirs("%s_saved_models" % exp_prefix, exist_ok=True)

def main(image_path):
    train_transforms = [
        transforms.Resize(size=(constants.img_size, constants.img_size)), # can be replaced with other transformations

        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomApply([transforms.ColorJitter(brightness=(1,1.2), contrast=(1,1.2), saturation=0, hue=0)],
        #                        p=0.5),
        transforms.ToTensor()
    ]

    train_dataset = MotorbikeDataset(path=image_path, size=constants.img_size, margin=constants.img_margin,
                                     count=constants.debug_n_sample,
                                     transforms=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=constants.batch_size, shuffle=True)

    real_label = .9
    fake_label = 0.

    D_net = Discriminator(ndf=64)
    G_net = Generator(z_dim=100,img_size=constants.img_size,ngf=64)
    criterion = nn.BCELoss()

    D_net.apply(weights_init)
    G_net.apply(weights_init)

    # # loading from pre-trained
    # weight_path = "./dc_gan_saved_models/model_1000.pt"
    # weight_ckpt = torch.load(weight_path, map_location='cpu')
    #
    # D_net.load_state_dict(weight_ckpt['d_model_state_dict'])
    # G_net.load_state_dict(weight_ckpt['g_model_state_dict'])
    # ####

    # From CPU to GPU
    D_net = D_net.to(constants.DEVICE)
    G_net = G_net.to(constants.DEVICE)
    criterion = criterion.to(constants.DEVICE)

    # Setup Adam optimizers for both G and D
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=constants.d_lr, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(G_net.parameters(), lr=constants.g_lr, betas=(0.5, 0.999))

    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lambda epoch: 0.9 ** epoch)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lambda epoch: 0.9 ** epoch)

    n_samples = len(train_dataset)
    img_list = []

    dataset_mean = torch.Tensor(train_dataset.mean).to(constants.DEVICE).view(1,3,1,1)
    dataset_std  = torch.Tensor(train_dataset.std).to(constants.DEVICE).view(1,3,1,1)

    for epoch_id in range(constants.epoch):
        if epoch_id > 1 and epoch_id % constants.lr_schedule_after_epoch == 0:
            scheduler_D.step()
            scheduler_G.step()

        for batch_id, (real_imgs, fns) in enumerate(train_loader):

            ###
            # debug, pls comment if not use
            # print (fns)
            # utils.tensor_show(real_imgs)
            ###
            G_net.train()
            D_net.train()

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(real_label), requires_grad=False)
            fake = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(fake_label), requires_grad=False)

            # Configure input
            real_imgs = real_imgs.to(constants.DEVICE)
            valid = valid.to(constants.DEVICE)
            fake  = fake.to(constants.DEVICE)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], 100)))).to(constants.DEVICE)

            # Generate a batch of images
            gen_imgs = G_net(z)

            # Denormalize
            gen_imgs = (gen_imgs + 1.) / 2
            gen_imgs = (gen_imgs - dataset_mean) / dataset_std #transforms.Normalize(mean=list(train_dataset.mean), std=list(train_dataset.std))(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            G_loss = criterion(D_net(gen_imgs), valid)

            G_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(D_net(real_imgs), valid)
            fake_loss = criterion(D_net(gen_imgs.detach()), fake)
            D_loss = (real_loss + fake_loss) / 2.

            D_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch_id, constants.epoch, batch_id, len(train_loader), D_loss.item(), G_loss.item())
            )

            batches_done = epoch_id * len(train_loader) + batch_id

            if batches_done % constants.sample_interval == 0:
                G_net.eval()

                # Generate a batch of images
                gen_imgs = G_net(z)

                # de-normalize
                gen_imgs = (gen_imgs + 1.) / 2

                # save-image for debugging
                save_image(gen_imgs.data[:25],
                           "%s_debug_images/%d.png" % (exp_prefix, batches_done),
                           nrow=5, normalize=True)

                # save model
                utils.save_models(g_model=G_net, d_model=D_net,
                                  other_params={
                                      "mean": train_dataset.mean,
                                      "std": train_dataset.std,
                                      "epoch": epoch_id,
                                      "d_loss": D_loss.item(),
                                      "g_loss": G_loss.item()
                                  },
                                  out_path="%s_saved_models/model_%d.pt" % (exp_prefix, batches_done)
                                  )

if __name__ == "__main__":
    main("./data/motobike")