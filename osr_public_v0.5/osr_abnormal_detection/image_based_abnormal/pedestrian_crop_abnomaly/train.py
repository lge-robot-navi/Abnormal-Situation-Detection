"""
--------------------------------------------------------------------------
    pedestrian crop image anomaly
    H.C. Shin, creatrix@etri.re.kr, 2019.10.24
--------------------------------------------------------------------------
    Copyright (C) <2019>  <H.C. Shin, creatrix@etri.re.kr>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
--------------------------------------------------------------------------
"""
# --phase 'train' --dataroot "./data" --epochs 1 --batchsize 512
from __future__ import print_function
from data import load_data
from model_4_human import AAE_basic
from options import Options
import glob
import random
import random
import glob
import numpy as np
import logging
import os
import sys
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from datetime import datetime

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.preprocessing import image
sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

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

def data_read(image_files):
    x_train = np.zeros((len(image_files), map_layer_num, map_y_size, map_x_size))
    for idx, file_name in enumerate(image_files):
        img = image.load_img(file_name, target_size=(map_y_size, map_x_size), color_mode='rgb')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x_train[idx, :, :, :] = img.reshape(1, map_layer_num, map_y_size, map_x_size)
    return x_train


def data_read_one(file_name):
    print(file_name)
    img = image.load_img(file_name, target_size=(map_y_size, map_x_size), color_mode='rgb')
    img = image.img_to_array(img)
    img_out = img.reshape(1, map_layer_num, map_y_size, map_x_size)
    return img_out


def main():
    # ------------------------------------------------ Training Phase ------------------------------------------------
    # image_files = random.sample(glob.glob('E:\\work\\pedestrian_crop_python_process\\Pedestrain_cropDB\\train\\0\\*.bmp'), 10)
    # image_files = random.sample(glob.glob('data/0.normal/*.bmp'), 10)
    # data_in = data_read(image_files)

    opt = Options().parse()
    opt.iwidth = map_x_size
    opt.iheight = map_y_size


    #---new--- depth for size
    ctinit = map_x_size
    while ctinit>4:
        ctinit = ctinit/2
    opt.ctinit = int(ctinit)
    #---new---

    opt.batchsize = 64
    opt.epochs = 1000
    opt.mask = 0  # 1: masking for simulation map
    opt.time = datetime.now()

    train_dataloader = load_data('./data/unsupervised/train/')  # path to trainset
    result_path = './results/{0}/'.format(opt.time)  # reconstructions durnig the training
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # dataloader = load_data(opt, data_in)
    model = AAE_basic(opt, train_dataloader)
    model.train()
    # print(np.sum(model.test()))

    # ------------------------------------------------ Test Phase ------------------------------------------------
    # image_files = random.sample(glob.glob('data/0.normal/*.bmp'), 10)
    # data_length = np.shape(image_files)[0]
    #
    # # dataloader = load_data(opt, data_in)
    # # model = AAE_basic(opt, dataloader)
    # # # model.train()
    # # print(np.sum(model.test()))
    #
    # opt.batchsize = 1
    # for idx in range(data_length):
    #     data_in = data_read_one(image_files[idx])
    #     dataloader = load_data(opt, data_in)
    #     model = AAE_basic(opt, dataloader)
    #     print(model.test())

global map_x_size, map_y_size, map_layer_num
map_x_size = 64
map_y_size = 64
map_layer_num = 3

if __name__ == '__main__':
    main()
