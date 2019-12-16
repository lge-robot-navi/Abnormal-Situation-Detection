"""
--------------------------------------------------------------------------
    fixed video data preprocessing
    2019.10.24
    H.C. Shin, creatrix@etri.re.kr
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

from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import PIL
from tqdm import tqdm
import glob
import argparse

def load_data(dataset, mode, path, save_dir, arg):
    strides = arg.strides
    patch_height = arg.patch_height
    patch_width = arg.patch_width

    if mode == 'train':
        print('Preprocessing Train..')
        f_list_tr = glob.glob(path+'Train/**/*.tif')
        f_list_tr.sort()
        X_train = []
        for idx, f_path in enumerate(f_list_tr):
            if (idx+1)%500==0:
                print(datetime.datetime.now(), idx+1)
            with Image.open(f_path) as img:
                img = np.asarray(img)[100:]
                img = np.expand_dims(img/255., axis=0)
                for h in range(img.shape[1]//strides):
                    for w in range(img.shape[2]//strides):
                        if patch_width + w*strides > img.shape[2]:
                            if patch_height + h*strides > img.shape[1]:
                                X_train.append(img[:,-patch_height:, -patch_width:])
                            else:
                                X_train.append(img[:,h*strides: patch_height + h*strides, -patch_width:])
                        else:
                            if patch_height + h*strides > img.shape[1]:
                                X_train.append(img[:,-patch_height:, w*strides: patch_height + w*strides])
                            else:
                                X_train.append(img[:,h*strides: patch_height + h*strides, w*strides: patch_width + w*strides])
        X_train = np.concatenate(X_train)
        print('X_train shape: ', X_train.shape)
        if not os.path.exists(save_dir+'/{}_dataset_patch'.format(dataset)):
            os.mkdir(save_dir+'/{}_dataset_patch'.format(dataset))
        np.save(save_dir+'/{}_dataset_patch/X_train'.format(dataset), X_train)
        
    if mode =='test':
        print('Preprocessing Test..')
        f_list_te = glob.glob(path+'Test/**/*.tif')
        f_list_te.sort()
        X_test = []
        L_test = []
        X_test_label = []

        for idx, f_path in enumerate(f_list_te):
            if (idx+1)%500==0:
                print(datetime.datetime.now(), idx+1)
            with Image.open(f_path) as img:
                img = np.asarray(img)[100:]
                img = np.expand_dims(img/255., axis=0)
                
                for h in range(img.shape[1]//strides):
                    for w in range(img.shape[2]//strides):
                        if patch_width + w*strides > img.shape[2]:
                            if patch_height + h*strides > img.shape[1]:
                                X_test.append(img[:,-patch_height:, -patch_width:])
                                L_test.append(idx)
                            else:
                                X_test.append(img[:,h*strides: patch_height + h*strides, -patch_width:])
                                L_test.append(idx)
                        else:
                            if patch_height + h*strides > img.shape[1]:
                                X_test.append(img[:,-patch_height:, w*strides: patch_height + w*strides])
                                L_test.append(idx)
                            else:
                                X_test.append(img[:,h*strides: patch_height + h*strides, w*strides: patch_width + w*strides])
                                L_test.append(idx)
            # labeling
            f_path = f_path.replace('\\', '/')
            f_gt_list = f_path.split('/')
            f_gt_list[-2] += '_gt'
            f_gt_list[-1] = f_gt_list[-1][:-3] + 'bmp'
            if f_gt_list[-2] in ['Test006_gt', 'Test009_gt', 'Test012_gt']:
                f_gt_list[-1] = 'frame' + f_gt_list[-1]
            f_gt_path = '/'.join(f_gt_list)
            
            with Image.open(f_gt_path) as img:
                img = np.asarray(img)[100:]
                img = np.expand_dims(img/255., axis=0)
                
                for h in range(img.shape[1]//strides):
                    for w in range(img.shape[2]//strides):
                        if patch_width + w*strides > img.shape[2]:
                            if patch_height + h*strides > img.shape[1]:
                                X_test_label.append(img[:,-patch_height:, -patch_width:])
                            else:
                                X_test_label.append(img[:,h*strides: patch_height + h*strides, -patch_width:])
                        else:
                            if patch_height + h*strides > img.shape[1]:
                                X_test_label.append(img[:,-patch_height:, w*strides: patch_height + w*strides])
                            else:
                                X_test_label.append(img[:,h*strides: patch_height + h*strides, w*strides: patch_width + w*strides])
                                
        X_test = np.concatenate(X_test)
        X_test_label = np.concatenate(X_test_label)
        L_test = np.array(L_test)
        print('X_test shape: {}, X_test label shape: {}, L_test shape: {}'.format(X_test.shape, X_test_label.shape, L_test.shape))
        
        if not os.path.exists(save_dir+'/{}_dataset_patch'.format(dataset)):
            os.mkdir(save_dir+'/{}_dataset_patch'.format(dataset))
        
        np.save(save_dir+'/{}_dataset_patch/X_test'.format(dataset), X_test)
        np.save(save_dir+'/{}_dataset_patch/Y_test'.format(dataset), X_test_label)
        np.save(save_dir+'/{}_dataset_patch/L_test'.format(dataset), L_test)
    
    if mode == 'smooth_test':
        # Test data
        f_list_te = glob.glob(path+'Test/**/*.tif')
        f_list_te.sort()
        X_test = []
        L_test = []
        Y_test = []
        thresholds = 200
        for idx, f_path in enumerate(f_list_te):
            if (idx+1)%500==0:
                print(datetime.datetime.now(), idx+1)
            with Image.open(f_path) as img:
                img = np.expand_dims(np.asarray(img)[100:]/255., axis=0)
                for h in range(img.shape[1]//strides):
                    for w in range(img.shape[2]//strides):
                        if patch_width + w*strides > img.shape[2]:
                            if patch_height + h*strides > img.shape[1]:
                                X_test.append(img[:,-patch_height:, -patch_width:])
                                L_test.append(idx)
                            else:
                                X_test.append(img[:,h*strides: patch_height + h*strides, -patch_width:])
                                L_test.append(idx)
                        else:
                            if patch_height + h*strides > img.shape[1]:
                                X_test.append(img[:,-patch_height:, w*strides: patch_height + w*strides])
                                L_test.append(idx)
                            else:
                                X_test.append(img[:,h*strides: patch_height + h*strides, w*strides: patch_width + w*strides])
                                L_test.append(idx)
            # labeling
            f_path = f_path.replace('\\', '/')
            f_gt_list = f_path.split('/')
            f_gt_list[-2] += '_gt'
            f_gt_list[-1] = f_gt_list[-1][:-3] + 'bmp'
            if f_gt_list[-2] in ['Test006_gt', 'Test009_gt', 'Test012_gt']:
                f_gt_list[-1] = 'frame' + f_gt_list[-1]
            f_gt_path = '/'.join(f_gt_list)
            
            with Image.open(f_gt_path) as img:
                img = np.expand_dims(np.asarray(img)[100:], axis=0)
                for h in range(img.shape[1]//strides):
                    for w in range(img.shape[2]//strides):
                        if patch_width + w*strides > img.shape[2]:
                            if patch_height + h*strides > img.shape[1]:
                                if np.sum(img[:,-patch_height:, -patch_width:]) > 255*thresholds:
                                    Y_test.append(1)
                                    #Y_test_tmp.append(np.sum(img[:,-patch_height:, -patch_width:]))
                                else:
                                    Y_test.append(0)
                            else:
                                if np.sum(img[:,h*strides: patch_height + h*strides, -patch_width:]) > 255*thresholds:
                                    Y_test.append(1)
                                    #Y_test_tmp.append(np.sum(img[:,h*strides: patch_height + h*strides, -patch_width:]))
                                else:
                                    Y_test.append(0)
                        else:
                            if patch_height + h*strides > img.shape[1]:
                                if np.sum(img[:,-patch_height:, w*strides: patch_height + w*strides]) > 255*thresholds:
                                    Y_test.append(1)
                                    #Y_test_tmp.append(np.sum(img[:,-patch_height:, w*strides: patch_height + w*strides]))
                                else:
                                    Y_test.append(0)
                            else:
                                if np.sum(img[:,h*strides: patch_height + h*strides, w*strides: patch_width + w*strides]) > 255*thresholds:
                                    Y_test.append(1)
                                    #Y_test_tmp.append(np.sum(img[:,h*strides: patch_height + h*strides, w*strides: patch_width + w*strides]))
                                else:
                                    Y_test.append(0)
        X_test = np.concatenate(X_test)
        Y_test = np.array(Y_test)
        L_test = np.array(L_test)
        print('X_test shape: {}, X_test label shape: {}, L_test shape: {}'.format(X_test.shape, Y_test.shape, L_test.shape))
        print("# of abnormal: %06d ; # of normal: % 06d"%(np.sum(Y_test), Y_test.shape[0] - np.sum(Y_test)))
        
        if not os.path.exists(save_dir+'/{}_dataset_patch'.format(dataset)):
            os.mkdir(save_dir+'/{}_dataset_patch'.format(dataset))
        
        np.save(save_dir+'/{}_dataset_patch/X_test_smooth'.format(dataset), X_test)
        np.save(save_dir+'/{}_dataset_patch/Y_test_smooth'.format(dataset), Y_test)
        np.save(save_dir+'/{}_dataset_patch/L_test_smooth'.format(dataset), L_test)

path = './dataset/'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='UCSDped2')
parser.add_argument('--save_dir', type=str, default='save')
parser.add_argument('--model', type=str, default='AAE')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patch_height', type=int, default=45)
parser.add_argument('--patch_width', type=int, default=45)
parser.add_argument('--strides', type=int, default=25)

arg = parser.parse_args()
    
if __name__ == '__main__':
    os.makedirs(arg.save_dir, exist_ok=True)

    if arg.dataset == 'UCSDped2':
        data_path = path + 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        print('Dataset path: ', data_path)
        # load train data
        load_data(arg.dataset, 'train', data_path, arg.save_dir, arg)
        # load test data
        load_data(arg.dataset, 'test', data_path, arg.save_dir, arg)
        # load smooth test data
        load_data(arg.dataset, 'smooth_test', data_path, arg.save_dir, arg)
    else:
        print('No Dataset')
            