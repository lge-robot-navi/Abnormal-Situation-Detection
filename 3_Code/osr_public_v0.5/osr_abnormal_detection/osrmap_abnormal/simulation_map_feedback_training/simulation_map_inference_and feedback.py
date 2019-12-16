"""
--------------------------------------------------------------------------
    simulation map  abnormal detection and feedback code
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
import os
import threading
import glob
import pickle
import torch
import numpy as np
from keras import optimizers
from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.python.client import device_lib
from numpy import unravel_index
from scipy import stats
from time import sleep
from torch.autograd import Variable
from options_AAE import Options
from networks_AAE import P_net, Q_net, D_net, weights_init


def data_read_one_prob_concatnate(file_name):
    '''
    map data read
    2019.10.24
    :input {simulation map}
    :return {probability map}
    '''
    img = image.load_img(file_name, color_mode='rgb')
    img = image.img_to_array(img)
    img_out = img[0:map_y_size, 0:map_x_size, :]
    return img_out


def prob_map_generation(map_in):
    '''
    probability map calculation
    2019.10.24
    :input {object map}
    :return {probability map}
    '''
    global prob_hist
    x_size, y_size, h_size = np.shape(map_in)
    map_prob = 100 * np.ones((y_size, x_size, h_size))
    for y in range(y_size):
        for x in range(x_size):
            if map_in[y, x, 0] > 0:
                den = np.clip(map_in[y, x, 0], 0, 10)
                spd = np.clip(map_in[y, x, 1], 0, 10)
                dir = np.clip(map_in[y, x, 2], 1, 9)
                ext_prob = 200*(np.sum(prob_hist[y][x][0][1:10])/(np.sum(prob_hist[y][x][0][:])+1))**(1/4)+1
                den_prob = 100*prob_hist[y][x][0][den]/np.sum(prob_hist[y][x][0][1:10]+1)+1
                ext_den_prob = stats.hmean([ext_prob, den_prob])
                spd_prob = 100*prob_hist[y][x][1][spd]/np.sum(prob_hist[y][x][1][1:10]+1)
                dir_prob = 100*prob_hist[y][x][2][dir]/np.sum(prob_hist[y][x][2][1:9]+1)
                map_prob[y, x, 0] = np.round_(np.clip(ext_den_prob, 1, 100), 1)
                map_prob[y, x, 1] = np.round_(np.clip(spd_prob, 1, 100), 1)
                map_prob[y, x, 2] = np.round_(np.clip(dir_prob, 1, 100), 1)
    return map_prob


def cnn_fc2_maha_dist(x_input):
    '''
    cnn model based Mahalanobis distance
    2019.10.24
    :input {map_data_in}
    :return {Mahalanobis distance}
    '''
    global model_fc2
    global n_mu, n_inv_cov, a_mu, a_inv_cov
    global datagen

    for x_batch in datagen.flow(x_input, batch_size=1):
        fc2_in = model_fc2.predict(x_batch)
        break
    n_maha_dist = np.sqrt(np.dot(np.dot(fc2_in-n_mu, n_inv_cov), (fc2_in-n_mu).T))
    a_maha_dist = np.sqrt(np.dot(np.dot(fc2_in-a_mu, a_inv_cov), (fc2_in-a_mu).T))
    maha_dist_return = [n_maha_dist, a_maha_dist]
    return maha_dist_return


def prob_hist_initialize():
    '''
    loading probability weight for probability map
    2019.10.24
    :input {weight file}
    :return {}
    '''
    global prob_hist
    global prob_time_prev

    prob_name = 'output\\probability_histogram_DB_python_smth'
    with open(prob_name, 'rb') as fp:
        prob_hist = pickle.load(fp)
    print("Probability histogram ready")
    prob_time_prev = 0


def model_cnn_initialize():
    '''
    cnn model and weight load
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    global model, model_fc2, datagen
    global n_mu, n_inv_cov, a_mu, a_inv_cov
    global cnn_time_prev

    json_name = "output\\model_cnn_map_datagen_fc2.json"
    json_file = open(json_name, "r")
    model = model_from_json(json_file.read())
    json_file.close()
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=1. / 255)
    model_weight_file_name = 'output\\weight_cnn_map_datagen_fc2.h5'
    model.load_weights(model_weight_file_name)
    model_fc2 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    print("CNN & FC2 model ready")

    maha_name = 'output\\mahalanobis_pre_calculation_map.npz'
    maha_data_load = np.load(maha_name)
    n_mu = maha_data_load['n_mu']
    n_inv_cov = maha_data_load['n_inv_cov']
    a_mu = maha_data_load['a_mu']
    a_inv_cov = maha_data_load['a_inv_cov']
    maha_data_load.close()
    print("Mahalanobis calculation ready")

    cnn_time_prev = 0


def check_prob_weight():
    '''
    probability weight update check & load
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    timer_prob = threading.Timer(100, check_prob_weight)
    timer_prob.start()
    global prob_hist
    global prob_time_prev

    prob_name = 'output\\probability_histogram_DB_python_smth'
    prob_time = os.path.getmtime(prob_name)
    if prob_time != prob_time_prev:
        sleep(1)
        with open(prob_name, 'rb') as fp:
            prob_hist = pickle.load(fp)
        print("============================================Probability histogram updated")
        prob_time_prev = prob_time


def check_cnn_weight():
    '''
    cnn weight update check & load
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    timer_cnn = threading.Timer(100, check_cnn_weight)
    timer_cnn.start()
    global cnn_time_prev
    global model, model_fc2
    global n_mu, n_inv_cov, a_mu, a_inv_cov

    cnn_name = 'output\\weight_cnn_map_datagen_fc2.h5'
    cnn_time = os.path.getmtime(cnn_name)
    if cnn_time != cnn_time_prev:
        sleep(1)
        model.load_weights(cnn_name)
        model_fc2 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
        maha_name = 'output\\mahalanobis_pre_calculation_map.npz'
        maha_data_load = np.load(maha_name)
        n_mu = maha_data_load['n_mu']
        n_inv_cov = maha_data_load['n_inv_cov']
        a_mu = maha_data_load['a_mu']
        a_inv_cov = maha_data_load['a_inv_cov']
        maha_data_load.close()
        print("============================================CNN model weight & Mahalanobis parameter updated")
        cnn_time_prev = cnn_time


def model_aae_initialize():
    '''
    aae weight update check & load
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    global opt
    global P, Q, D, X

    opt = Options().parse()
    opt.iwidth = map_x_size
    opt.iheight = map_y_size
    opt.batchsize = 1
    opt.ngpu = 0
    opt.gpu_ids = -1
    opt.mask = 1
    N = 1000

    Q = Q_net(opt.iheight, opt.iwidth, opt.zdim, opt.nc, opt.ngf, opt.ngpu)
    P = P_net(opt.iheight, opt.iwidth, opt.zdim, opt.nc, opt.ngf, opt.ngpu)
    D = D_net(opt.zdim, N)
    P.apply(weights_init)
    Q.apply(weights_init)

    X = torch.zeros(opt.batchsize, 3, opt.iheight, opt.iwidth)
    X = Variable(X, requires_grad=False)

    P_net_path = "output/P.pth".format(opt.outf)
    Q_net_path = "output/Q.pth".format(opt.outf)
    P_pretrained_dict = torch.load(P_net_path, map_location='cpu')['state_dict']
    Q_pretrained_dict = torch.load(Q_net_path, map_location='cpu')['state_dict']
    P.load_state_dict(P_pretrained_dict)
    Q.load_state_dict(Q_pretrained_dict)
    P.eval()
    Q.eval()


def aae_recon_error(data_in):
    '''
    aae weight reconstruction error calculation
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    data_in = data_in.transpose(2, 0, 1)
    data_in = np.expand_dims(data_in, axis=0)
    t_data_in = torch.from_numpy(data_in)
    X.data.resize_(t_data_in.size()).copy_(t_data_in)
    z_sample = Q(X)
    fake = P(z_sample)
    if opt.mask:
        x = torch.zeros(X.shape)
        y = torch.ones(X.shape)
        mask = torch.where(X == 0, x, y)
        fake = torch.where(mask == 1, fake, X)
    if opt.z_test:
        fake_z_sample = Q(fake)
        error = torch.pow((fake_z_sample - z_sample), 2).view(z_sample.size(0), -1).sum(1)
    else:
        error = torch.pow((fake - X), 2).view(X.size(0), -1).sum(1)
    an_scores = error.data.numpy()
    aae_result = np.round_(np.log(an_scores[0]), 2)
    return aae_result


def position_maha_prob_calculation(map_in, map_prob):
    '''
    cnn Mahalanobis distance and calculation
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    map_cont = np.concatenate((25 * map_in, 2.5 * map_prob), axis=1)
    x_in = np.reshape(map_cont, (1, np.shape(map_cont)[0], np.shape(map_cont)[1], np.shape(map_cont)[2]))
    maha_result = cnn_fc2_maha_dist(x_in)
    maha_nor = np.round_(maha_result[0][0][0], 2)
    maha_abn = np.round_(maha_result[1][0][0], 2)
    maha_dist_ratio = np.round_(np.clip((maha_result[0] / (maha_result[1] + 0.0001))[0, 0], 0, 100), 2)
    map_prob_temp = np.multiply(np.multiply(map_prob[:, :, 0], map_prob[:, :, 1]), map_prob[:, :, 2])
    prob_value = np.round_(np.clip(np.min(map_prob_temp), 1, 100), 2)
    position = unravel_index(map_prob_temp.argmin(), map_prob_temp.shape)
    normalized_maha_score = np.round_(np.log(maha_dist_ratio+0.0001), 2)

    return maha_nor, maha_abn, normalized_maha_score, prob_value, position


# ------- main()------------------------------------------------------------
global map_x_size, map_y_size, map_layer_num
map_y_size = 128
map_x_size = 128
map_layer_num = 3

os.system("start python map_CNN_training_command.py")
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
print(device_lib.list_local_devices())
os.system("start python Probability_training_command.py")
os.system("start python map_AAE_training_command.py")

prob_hist_initialize()
model_aae_initialize()
model_cnn_initialize()
check_prob_weight()
check_cnn_weight()

while 1:
    image_files = glob.glob('mapDB\\test\\0\\*.bmp')
    data_length = np.shape(image_files)[0]
    inference_normal_result = np.zeros((data_length, 5))
    false_alarm_cnt = 0
    for idx in range(data_length):
        map_data = data_read_one_prob_concatnate(image_files[idx])
        aae_score = aae_recon_error(map_data)
        map_in = (map_data/25).astype(int)
        map_prob = prob_map_generation(map_in)
        maha_nor, maha_abn, maha_score, prob_score, position = position_maha_prob_calculation(map_in, map_prob)
        aae_result = np.round_(aae_score/prob_score, 2)
        print(idx, '\t', maha_nor, '\t', maha_abn,  '\t', maha_score, '\t', prob_score, '\t', aae_result)

        if aae_result < 2 and maha_abn > 2 and maha_score > 0:
            print('===================== false count ===============================', false_alarm_cnt)
            false_alarm_cnt += 1
            map_save = np.concatenate((25*map_in, 2.5 * map_prob), axis=1)
            file_name = 'mapDB\\train\\cnn\\0\\feecback_normal_' + str(repeat) + '_' + str(idx) + '_' + str(aae_result) + '_' + str(maha_score) + '.bmp'
            image.save_img(file_name, map_save)
