import os
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch
from networks_4_human import P_net, Q_net, D_net, weights_init

import torchvision.utils as vutils

class AAE_basic:
    @staticmethod
    def name():
        return 'aae_anomaly'

    def __init__(self, opt, dataloader=None):
        super(AAE_basic, self).__init__()

        self.opt = opt
        self.model_saved = False
        self.d_loss = None
        self.g_loss = None
        self.recon_loss = None
        self.z_loss = None
        self.dataloader = dataloader
        self.epoch = 0
        self.cuda = torch.cuda.is_available()
        self.N = 1000
        self.fake = None
        self.P = None
        self.Q = None
        self.D = None

        self.loss_log = []

        self.Q = Q_net(self.opt.iheight, self.opt.iwidth, self.opt.zdim, self.opt.nc, self.opt.ngf, self.opt.ngpu, self.opt.ctinit)
        self.P = P_net(self.opt.iheight, self.opt.iwidth, self.opt.zdim, self.opt.nc, self.opt.ngf, self.opt.ngpu, self.opt.ctinit)
        self.D = D_net(self.opt.zdim, self.N)
        self.P.apply(weights_init)
        self.Q.apply(weights_init)
        # self.D.apply(weights_init)

        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'Q.pth'))['epoch']
            self.Q.load_state_dict(torch.load(os.path.join(self.opt.resume, 'Q.pth'))['state_dict'])
            self.P.load_state_dict(torch.load(os.path.join(self.opt.resume, 'P.pth'))['state_dict'])
            self.D.load_state_dict(torch.load(os.path.join(self.opt.resume, 'D.pth'))['state_dict'])
            print("\tDone.\n")

        if self.cuda:
            self.Q = self.Q.cuda()
            self.P = self.P.cuda()
            self.D = self.D.cuda()

        self.TINY = 1e-15
        self.X = torch.zeros(self.opt.batchsize, 3, self.opt.iheight, self.opt.iwidth)
        self.gt = torch.LongTensor(self.opt.batchsize)
        self.X = Variable(self.X, requires_grad=False)
        self.gt = Variable(self.gt, requires_grad=False)

    def train(self):
        if self.opt.manualseed is not None:
            torch.manual_seed(self.opt.manualseed)

        # Set optimizators
        P_decoder = optim.Adam(self.P.parameters(), lr=self.opt.gen_lr, betas=(self.opt.beta1, 0.999))
        Q_encoder = optim.Adam(self.Q.parameters(), lr=self.opt.gen_lr, betas=(self.opt.beta1, 0.999))

        Q_generator = optim.Adam(self.Q.parameters(), lr=self.opt.reg_lr, betas=(self.opt.beta1, 0.999))
        D_gauss_solver = optim.Adam(self.D.parameters(), lr=self.opt.reg_lr, betas=(self.opt.beta1, 0.999))

        scheduler_1 = MultiStepLR(P_decoder, milestones=[50, 1000], gamma=0.1)
        scheduler_2 = MultiStepLR(Q_encoder, milestones=[50, 1000], gamma=0.1)
        scheduler_3 = MultiStepLR(Q_generator, milestones=[50, 1000], gamma=0.1)
        scheduler_4 = MultiStepLR(D_gauss_solver, milestones=[50, 1000], gamma=0.1)

        for self.epoch in range(self.opt.epochs):

            scheduler_1.step()
            scheduler_2.step()
            scheduler_3.step()
            scheduler_4.step()

            self.train_epoch(self.P, self.Q, self.D, P_decoder, Q_encoder, Q_generator, D_gauss_solver)
            print("epoch {0} done.".format(self.epoch))
            self.save_weights(self.epoch)



    def train_epoch(self, P, Q, D_gauss, P_decoder_solver, Q_encoder_solver, Q_generator_solver, D_gauss_solver):
        # Set the networks in train mode (apply dropout when needed)
        Q.train()
        P.train()
        D_gauss.train()

        count = 0
        for data in self.dataloader:
            self.set_input(data)

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Reconstruction phase
            #######################
            z_sample = Q(self.X)
            self.fake = P(z_sample)

            if count % 100 == 0:  # save image
                vutils.save_image(self.X[0:64, :, :, :],
                                  './results/{0}/{1}_{2}_real.png'.format(self.opt.time, self.epoch, count),
                                  nrow=10, normalize=True)
                vutils.save_image(self.fake[0:64, :, :, :],
                                  './results/{0}/{1}_{2}_fake.png'.format(self.opt.time, self.epoch, count),
                                  nrow=10, normalize=True)
            count+=1

            if self.opt.mask:  # masking selection for simualtion map
                x = torch.zeros(self.X.shape).cuda()
                y = torch.ones(self.X.shape).cuda()
                mask = torch.where(self.X == 0, x, y).cuda()
                self.fake = torch.where(mask == 1, self.fake, self.X)
                self.recon_loss = F.mse_loss(self.fake, self.X)
            else:
                self.recon_loss = F.mse_loss(self.fake, self.X)

            print("Reconstruction loss: {0}".format(self.recon_loss.item()))
            self.loss_log.append(self.recon_loss.item())


            self.recon_loss.backward(retain_graph=self.opt.z_loss)

            P_decoder_solver.step()
            Q_encoder_solver.step()

            if self.opt.z_loss:
                z_fake_sample = Q(self.fake)
                self.z_loss = F.mse_loss(z_fake_sample, z_sample, reduce=True, size_average=self.opt.size_average)
                self.z_loss.backward()
                Q_encoder_solver.step()
                Q.zero_grad()

            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Regularization phase
            #######################
            # Discriminator
            Q.eval()
            # sample from N(0,5). Note torch.randn samples from N(0,1)
            z_real_gauss = Variable(torch.randn(self.opt.batchsize, self.opt.zdim) * self.opt.z_multiplier)
            if self.cuda:
                z_real_gauss = z_real_gauss.cuda()

            z_fake_gauss = Q(self.X)

            D_real_gauss = D_gauss(z_real_gauss)
            D_fake_gauss = D_gauss(z_fake_gauss)

            self.d_loss = -torch.mean(torch.log(D_real_gauss + self.TINY) + torch.log(1 - D_fake_gauss + self.TINY))

            self.d_loss.backward()
            D_gauss_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            # Generator
            Q.train()
            z_fake_gauss = Q(self.X)

            D_fake_gauss = D_gauss(z_fake_gauss)
            self.g_loss = -torch.mean(torch.log(D_fake_gauss + self.TINY))

            self.g_loss.backward()
            Q_generator_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

        return

    def test(self):
        if self.opt.load_weights:
            P_net_path = "{}/P.pth".format(self.opt.outf)
            Q_net_path = "{}/Q.pth".format(self.opt.outf)
            P_pretrained_dict = torch.load(P_net_path)['state_dict']
            Q_pretrained_dict = torch.load(Q_net_path)['state_dict']

            try:
                self.P.load_state_dict(P_pretrained_dict)
                self.Q.load_state_dict(Q_pretrained_dict)
            except IOError:
                raise IOError("weights not found")
            print('   Loaded weights.')

        # Create big error tensor for the test set.
        self.an_scores = torch.FloatTensor(len(self.dataloader.dataset), 1).zero_()
        self.gt_labels = torch.LongTensor(len(self.dataloader.dataset), 1).zero_()
        self.filenames = [None]*len(self.dataloader.dataset)

        if self.opt.gpu_ids:
            self.an_scores = self.an_scores.cuda()

        self.Q.eval()
        self.P.eval()

        for i, data in enumerate(self.dataloader):
            self.set_input(data)

            z_sample = self.Q(self.X)
            self.fake = self.P(z_sample)

            if self.opt.z_test:
                fake_z_sample = self.Q(self.fake)
                error = torch.pow((fake_z_sample - z_sample), 2).view(z_sample.size(0), -1).sum(1)
            else:
                error = torch.pow((self.fake - self.X), 2).view(self.X.size(0), -1).sum(1)

            self.filenames[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.dataloader.dataset.imgs[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)]
            if self.opt.gpu_ids:
                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.data.unsqueeze(1)
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.data.unsqueeze(1)
            else:
                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0),
                1] = error.cpu().data.view(error.size(0), 1)
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.cpu().data

        self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
        return self.an_scores.cpu().numpy()

    def set_input(self, input):
        self.X.data.resize_(input[0].size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])
        if self.cuda:
            self.X, self.gt = self.X.cuda(), self.gt.cuda()

    def save_weights(self, epoch):
        weight_dir = os.path.join(self.opt.outf, 'output/{0}'.format(self.opt.time))
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        torch.save({'epoch': epoch + 1, 'state_dict': self.Q.state_dict()}, '%s/Q.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.P.state_dict()}, '%s/P.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.D.state_dict()}, '%s/D.pth' % (weight_dir))


