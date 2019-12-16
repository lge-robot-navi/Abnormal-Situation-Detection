import random
import glob
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch
from options import Options
from networks_4_human import P_net, Q_net, D_net, weights_init
import numpy as np
import os
import sys

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.preprocessing import image
sys.stderr = stderr

def data_read_one(file_name):
    img = image.load_img(file_name, target_size=(map_y_size, map_x_size), color_mode='rgb')
    img = image.img_to_array(img)
    img_out = img.reshape(1, map_layer_num, map_y_size, map_x_size)
    return img_out


def load_data(data_path):
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize((64, 64)),  # 64x64 input
                                   transforms.ToTensor(),  # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                                   transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                        (0.5, 0.5, 0.5)),  # (c - m)/s 니까...
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=8)

    return dataloader
map_x_size = 64
map_y_size = 64
map_layer_num = 3

opt = Options().parse()
opt.iwidth = map_x_size
opt.iheight = map_y_size
opt.batchsize = 1
opt.ngpu = 0
opt.gpu_ids = -1

# ---new---
ctinit = map_x_size
while ctinit > 4:
    ctinit = ctinit / 2
opt.ctinit = int(ctinit)
# ---new---

# opt.mask = 1

model_saved = False
d_loss = None
g_loss = None
recon_loss = None
z_loss = None

N = 1000
fake = None
P = None
Q = None
D = None

Q = Q_net(opt.iheight, opt.iwidth, opt.zdim, opt.nc, opt.ngf, opt.ngpu, opt.ctinit)
P = P_net(opt.iheight, opt.iwidth, opt.zdim, opt.nc, opt.ngf, opt.ngpu, opt.ctinit)
D = D_net(opt.zdim, N)
P.apply(weights_init)
Q.apply(weights_init)
# self.D.apply(weights_init)

TINY = 1e-15
X = torch.zeros(opt.batchsize, 3, opt.iheight, opt.iwidth)
X = Variable(X, requires_grad=False)

P_net_path = "output/2019-11-17 17:14:56.747041/P.pth".format(opt.outf)
Q_net_path = "output/2019-11-17 17:14:56.747041/Q.pth".format(opt.outf)
P_pretrained_dict = torch.load(P_net_path)['state_dict']
Q_pretrained_dict = torch.load(Q_net_path)['state_dict']
P.load_state_dict(P_pretrained_dict)
Q.load_state_dict(Q_pretrained_dict)

#
# # Create big error tensor for the test set.
an_scores = torch.FloatTensor(1, 1).zero_()

Q.eval()
P.eval()

test_loader = load_data('./data/unsupervised/test_un/')

# image_files = random.sample(glob.glob('data/0.normal/*.bmp'), 10)
# data_length = np.shape(image_files)[0]

# for idx in range(data_length):
for data, idx in test_loader:

    input = data

    z_sample = Q(input)
    fake = P(z_sample)

    # # Masking part from simulation training -
    if opt.mask:
        x = torch.zeros(X.shape)
        y = torch.ones(X.shape)
        mask = torch.where(X == 0, x, y)
        fake = torch.where(mask == 1, fake, X)
    # -----------------------------------------

    if opt.z_test:
        fake_z_sample = Q(fake)
        error = torch.pow((fake_z_sample - z_sample), 2).view(z_sample.size(0), -1).sum(1)
    else:
        error = torch.pow((fake - X), 2).view(X.size(0), -1).sum(1)
    an_scores = error.data.numpy()
    # an_scores = (an_scores - torch.min(an_scores)) / (torch.max(an_scores) - torch.min(an_scores))  # score normalization
    print(an_scores)

