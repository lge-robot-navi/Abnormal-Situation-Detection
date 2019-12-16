import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
N = 1000

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        mod.weight.data.normal_(0.0, 0.01)

# Encoder
class Q_net(nn.Module):
    def __init__(self, iheight, iwidth, nz, nc, ndf, ngpu, ctinit, add_final_conv=True):
        super(Q_net, self).__init__()
        self.ngpu = ngpu
        assert iwidth % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        main.add_module('initial_conv_{0}-{1}'.format(nc, ndf),  nn.Conv2d(nc, ndf, 4, 2, 1, bias=True))
        main.add_module('initial_relu_{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = iwidth / 2, ndf

        self.nz = nz

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        if add_final_conv:
            main.add_module('final_{0}-{1}_conv'.format(cndf, 1), nn.Conv2d(cndf, self.nz, ctinit, 1, 0, bias=True))

        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.view(-1, self.nz).squeeze()
        return output


# Decoder
class P_net(nn.Module):
    def __init__(self, iheight, iwidth, nz, nc, ngf, ngpu, ctinit):
        super(P_net, self).__init__()
        self.ngpu = ngpu
        assert iwidth % 16 == 0, "isize has to be a multiple of 16"
        self.nz = nz
        cngf, tisize = ngf // 2, 4

        while tisize >= iwidth:  # != --> >=
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()

        main.add_module('initial_{0}-{1}_convt'.format(self.nz, cngf), nn.ConvTranspose2d(self.nz, cngf, ctinit, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf), nn.ReLU(inplace=True))

        csize, _ = 4, cngf
        while csize < (iwidth // 2):
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2), nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2), nn.ReLU(inplace=True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=True))
        main.add_module('final_{0}_tanh'.format(nc), nn.Tanh())
        self.main = main

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        if isinstance(z.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
        else:
            output = self.main(z)
        return output


class D_net(nn.Module):
    def __init__(self, nz, N):
        super(D_net, self).__init__()
        self.lin1 = nn.Linear(nz, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.5, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.5, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
