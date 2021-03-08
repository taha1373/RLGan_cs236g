import torch
import os
import numpy as np
import time
import datetime
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image
from gan import Generator, Discriminator
from collections import OrderedDict
from utils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Trainer(object):
    """Generator trainer"""
    def __init__(self, args, latent_loader, model_decoder, visualizer):
        """
        initialize Generator trainer

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        latent_loader : iterator
            latent space data loader
        model_decoder : torch.nn.Module
            decoder model used in auto-decoder
        visualizer : any
            method for visualizing results
        """

        # decoder settings
        self.model_decoder = model_decoder
        self.vis = visualizer

        # Data loader
        self.latent_loader = latent_loader

        # exact model and loss
        self.args = args
        self.adv_loss = args.adv_loss

        # Model hyper-parameters
        self.l_size = args.l_size
        self.g_num = args.g_num
        self.z_dim = args.z_dim
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.parallel = args.parallel

        self.lambda_gp = args.lambda_gp
        self.total_step = args.total_step
        self.d_iters = args.d_iters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.lr_decay = args.lr_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.pretrained_num = args.pretrained_num

        # self.use_tensorboard = args.use_tensorboard
        self.save_dir = args.save_dir
        self.log_path = os.path.join(self.save_dir, args.log_path)
        self.model_save_path = os.path.join(self.save_dir, args.model_save_path)
        self.sample_path = os.path.join(self.save_dir, args.sample_path)

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        self.writer = SummaryWriter(self.log_path)

        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step

        self.build_model()

        # if self.use_tensorboard:
        #     self.build_tensorboard()

        # Start with trained model
        if self.pretrained_num:
            self.load_pretrained_model()

    def train(self):
        """
        trains models
        """
        # Data iterator
        train_iter = iter(self.latent_loader)

        # step_per_epoch = len(self.data_loader)
        train_step_per_epoch = len(self.latent_loader)

        model_save_step = int(self.model_save_step * train_step_per_epoch)

        # # Fixed input for debugging
        fixed_z_np = np.arange(-self.args.max_action, self.args.max_action,
                               (self.args.max_action * 2) / self.batch_size)
        fixed_z_n = tensor2var(torch.FloatTensor(fixed_z_np,))
        fixed_z = fixed_z_n.unsqueeze(1)

        # Start with trained model
        if self.pretrained_num:
            start = self.pretrained_num + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_latent = next(train_iter)[0]
            except:
                print('except')
                train_iter = iter(self.latent_loader)
                real_latent = next(train_iter)[0]

            real_latent = real_latent.unsqueeze(1)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_latent = tensor2var(real_latent)
            d_out_real, dr1 = self.D(real_latent)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var((torch.randn(real_latent.size(0), self.z_dim)))
            fake_latent, gf1 = self.G(z)
            d_out_fake, df1 = self.D(fake_latent)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_latent.size(0), 1, 1, 1).cuda().expand_as(real_latent)
                interpolated = Variable(alpha * real_latent.data + (1 - alpha) * fake_latent.data, requires_grad=True)
                out, _ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_latent.size(0), self.z_dim))
            fake_latent, _ = self.G(z)

            # Compute loss with fake images
            g_out_fake, _ = self.D(fake_latent)  # batch x n  TODO "_"
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                d_real = d_loss_real.data
                if d_real.dim():
                    d_real_first = d_real[0]
                else:
                    d_real_first = d_real

                d_fake = d_loss_fake.data
                if d_fake.dim():
                    d_fake_first = d_fake[0]
                else:
                    d_fake_first = d_fake

                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, ".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step, d_real_first, d_fake_first))
                self.writer.add_scalar("d_out_real", d_real_first, step + 1)
                self.writer.add_scalar("d_out_fake", d_fake_first, step + 1)

            # Sample images
            if (step + 1) % self.sample_step == 0:
                # fake_latent, _ = self.G(fixed_z)

                encoded = fake_latent.contiguous().view(self.batch_size, self.l_size)

                pc_1 = self.model_decoder(encoded)

                self.vis(pc_1, self.batch_size)
                plt.title("GAN result")
                plt.savefig(os.path.join(self.sample_path, "step_{}".format(step + 1)))
                plt.show()

            if (step + 1) % model_save_step == 0:
                self.writer.flush()
                # print(os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):
        """
        builds generator model for training
        """
        self.G = Generator(self.l_size, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.l_size, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr,
                                            [self.beta1, self.beta2])

        print(self.G)
        print(self.D)

    def load_pretrained_model(self):
        """
        load pretrained model to continue training based on number of steps (pretrained_num)

        Raises
        ------
        FileNotFoundError
            if no model dict is present ({pretrained_num}_G.th and {pretrained_num}_D.th)
        """
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.pretrained_num))
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path))
        else:
            raise FileNotFoundError('generator does not exist')

        D_path = os.path.join(self.model_save_path, '{}_D.pth'.format(self.pretrained_num))
        if os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path))
        else:
            raise FileNotFoundError('discriminator does not exist')
        print('loaded trained models (step: {})..!'.format(self.pretrained_num))

    def reset_grad(self):
        """
        reset gradients
        """
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()