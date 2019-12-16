""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse

class Options():
    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--ndata', default=False, help='True when data is given as ndarray')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=100, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--iwidth', type=int, default=128, help='input image width network after transformation.')
        self.parser.add_argument('--iheight', type=int, default=128, help='input image height for network after transformation.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--manualseed', type=int, help='manual seed')
        self.parser.add_argument('--mask', type=int, default='0', help='masking for simulation map')
        self.parser.add_argument('--ctinit', type=int, default='1', help='final convolution size parameter')
        self.parser.add_argument('--time', type=str, default='', help='current time')


        # Train
        self.parser.add_argument('--outf', type=str, default='', help='visdom server of the web display')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
        self.parser.add_argument('--gen_lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--reg_lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--alpha', type=float, default=75, help='alpha to weight l1 loss. default=500')
        self.parser.add_argument('--threshold', type=float, default=0.00, help='threshold for binary classification')
        self.parser.add_argument('--size_average', action='store_true', default=False, help='use average error over the mini-batch')
        self.parser.add_argument('--z_test', action='store_true', default=False, help='Use z value for test instead image.')
        self.parser.add_argument('--z_loss', action='store_true', default=False, help='Add z loss for training.')
        self.parser.add_argument('--z_multiplier', type=float, default=5.0, help='initial learning rate for adam')
        self.parser.add_argument('--zdim', type=int, default=150, help='size of the latent z vector')
        self.opt = None

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
