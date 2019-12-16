"""
--------------------------------------------------------------------------
    map data receiving and abnormal detection code
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

import cv2
import rospy
from grid_map_msgs.msg import GridMap
import numpy as np
import datetime
import copy
import pickle
from scipy import stats
from scipy import signal
from keras import optimizers
from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

layered_map = None


def sub_osr_map_cb(msg):
    '''
    map data receiving from ROS
    2019.10.24
    :input {ROS msg}
    :return {formated map}
    '''
    global layered_map
    layered_map = None

    if len(msg.layers) > 0:
        layered_map = np.zeros((msg.data[0].layout.dim[1].size, msg.data[0].layout.dim[0].size, len(msg.layers)), 'float32')
        map_size_0 = msg.data[1].layout.dim[0].size
        map_size_1 = msg.data[1].layout.dim[1].size
        # Human
        layered_map[:, :, 0] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('person_number')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 1] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('person_flow_x')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 2] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('person_flow_y')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 3] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('person_posture')].data, (map_size_0, map_size_1))))
        # Car
        layered_map[:, :, 4] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('car_number')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 5] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('car_flow_x')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 6] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('car_flow_y')].data, (map_size_0, map_size_1))))
        # Elevation
        layered_map[:, :, 7] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('elevation')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 8] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('elevation_update')].data, (map_size_0, map_size_1))))
        # Thermal
        layered_map[:, :, 9] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('thermal')].data, (map_size_0, map_size_1))))
        layered_map[:, :, 10] = np.flipud(np.rot90(np.reshape(msg.data[msg.layers.index('thermal_update')].data, (map_size_0, map_size_1))))


def map_reducer(map_in):
    '''
    map data resolution reducer
    2019.10.24
    :input {numpy array}
    :return {numpy array}
    '''
    size = np.shape(map_in)
    map_reduced = np.zeros((int(size[0]/2), int(size[1]/2), int(size[2])))
    for idx_y in range(0, size[0], 2):
        for idx_x in range(0, size[1], 2):
            # Human
            map_in_temp = copy.deepcopy(map_in[idx_y:idx_y+2, idx_x:idx_x+2, 0])
            map_in_temp[np.isnan(map_in_temp)] = -99
            position = np.unravel_index(map_in_temp.argmax(), map_in_temp.shape)
            map_reduced[int(idx_y/2), int(idx_x/2), 0:4] = copy.deepcopy(map_in[int(idx_y+position[0]), int(idx_x+position[1]), 0:4])
            # Car
            map_in_temp = copy.deepcopy(map_in[idx_y:idx_y+2, idx_x:idx_x+2, 4])
            map_in_temp[np.isnan(map_in_temp)] = -99
            position = np.unravel_index(map_in_temp.argmax(), map_in_temp.shape)
            map_reduced[int(idx_y/2), int(idx_x/2), 4:7] = copy.deepcopy(map_in[int(idx_y+position[0]), int(idx_x+position[1]), 4:7])
            # Elevation
            map_in_temp_data = copy.deepcopy(map_in[idx_y:idx_y+2, idx_x:idx_x+2, 7])
            map_in_temp_data[np.isnan(map_in_temp_data)] = -99
            map_in_temp_update = copy.deepcopy(map_in[idx_y:idx_y+2, idx_x:idx_x+2, 8])
            map_in_temp_update[np.isnan(map_in_temp_update)] = -99
            map_in_temp = map_in_temp_data * map_in_temp_update
            position = np.unravel_index(map_in_temp.argmax(), map_in_temp.shape)
            map_reduced[int(idx_y/2), int(idx_x/2), 7:9] = copy.deepcopy(map_in[int(idx_y+position[0]), int(idx_x+position[1]), 7:9])
            # Thermal
            map_in_temp_data = copy.deepcopy(map_in[idx_y:idx_y+2, idx_x:idx_x+2, 9])
            map_in_temp_data[np.isnan(map_in_temp_data)] = -99
            map_in_temp_update = copy.deepcopy(map_in[idx_y:idx_y+2, idx_x:idx_x+2, 10])
            map_in_temp_update[np.isnan(map_in_temp_update)] = -99
            map_in_temp = map_in_temp_data * map_in_temp_update
            position = np.unravel_index(map_in_temp.argmax(), map_in_temp.shape)
            map_reduced[int(idx_y/2), int(idx_x/2), 9:11] = copy.deepcopy(map_in[int(idx_y+position[0]), int(idx_x+position[1]), 9:11])

    return map_reduced


def prob_hist_initialize():
    '''
    loading probability weight for probability map
    2019.10.24
    :input {weight file}
    :return {}
    '''
    global prob_hist
    global prob_time_prev

    prob_name = 'output/probability_histogram_DB_pickle2'
    with open(prob_name, 'rb') as fp:
        prob_hist = pickle.load(fp)
    print("-------------------------------------------------Probability histogram ready", np.shape(prob_hist))
    prob_time_prev = 0


def prob_map_generation(hum_den, hum_spd, hum_dir, car_den, car_spd, car_dir):
    '''
    dynamic object probability map calculation
    2019.10.24
    :input {object map}
    :return {probability map}
    '''
    global prob_hist
    y_size, x_size = np.shape(hum_den)
    map_prob = 100 * np.ones((y_size, x_size, 3))
    for y in range(y_size):
        for x in range(x_size):
            if hum_den[y, x] > 0:
                den = int(np.clip(hum_den[y, x], 0, 10))
                spd = int(np.clip(hum_spd[y, x], 0, 10))
                dir = int(np.clip(hum_dir[y, x], 0, 8))
                ext_prob = 100*((np.sum(prob_hist[y][x][0][1:10])/(np.sum(prob_hist[y][x][0][:])+0.01))**(1/2))+1
                den_prob = 100*prob_hist[y][x][0][den]/np.sum(prob_hist[y][x][0][1:10]+0.01)+1
                ext_den_prob = stats.hmean([ext_prob, den_prob])
                spd_prob = 100*prob_hist[y][x][1][spd]/np.sum(prob_hist[y][x][1][:]+0.01)
                dir_prob = 100*prob_hist[y][x][2][dir]/np.sum(prob_hist[y][x][2][1:]+0.01)
                map_prob[y, x, 0] = np.clip(int(ext_den_prob), 1, 100)
                map_prob[y, x, 1] = np.clip(int(spd_prob), 1, 100)
                map_prob[y, x, 2] = np.clip(int(dir_prob), 1, 100)

            if car_den[y, x] > 0:
                den = int(np.clip(car_den[y, x], 0, 10))
                spd = int(np.clip(car_spd[y, x], 0, 10))
                dir = int(np.clip(car_dir[y, x], 0, 8))
                ext_prob = 100*((np.sum(prob_hist[y][x][3][1:10])/(np.sum(prob_hist[y][x][3][:])+0.01))**(1/2))+1
                den_prob = 100*prob_hist[y][x][3][den]/np.sum(prob_hist[y][x][3][1:10]+0.01)+1
                ext_den_prob = stats.hmean([ext_prob, den_prob])
                spd_prob = 100*prob_hist[y][x][4][spd]/np.sum(prob_hist[y][x][4][:]+0.01)
                dir_prob = 100*prob_hist[y][x][5][dir]/np.sum(prob_hist[y][x][5][1:]+0.01)
                map_prob[y, x, 0] = np.clip(int(ext_den_prob), 1, 100)
                map_prob[y, x, 1] = np.clip(int(spd_prob), 1, 100)
                map_prob[y, x, 2] = np.clip(int(dir_prob), 1, 100)

    return map_prob


def obj_spd_dir(vy, vx, spd_th):
    '''
    object property conversion
    2019.10.24
    :input {object property}
    :return {object property}
    '''
    obj_spd = round(np.sqrt(np.power(vy, 2) + np.power(vx, 2)), 1)
    if obj_spd > spd_th:
        obj_ang = np.arctan2(vy, vx) * 180 / np.pi  # direction: up=1, right=3, down=5, left=7, stop=0
        if 67.5 < obj_ang <= 112.5:
            obj_dir = 1
        if 22.5 < obj_ang <= 67.5:
            obj_dir = 2
        if -22.5 < obj_ang <= 22.5:
            obj_dir = 3
        if -67.5 < obj_ang <= -22.5:
            obj_dir = 4
        if -112.5 < obj_ang <= -67.5:
            obj_dir = 5
        if -157.5 < obj_ang <= -112.5:
            obj_dir = 6
        if obj_ang > 157.5 or obj_ang <= -157.5:
            obj_dir = 7
        if 112.5 < obj_ang <= 157.5:
            obj_dir = 8
    else:
        obj_spd = 0
        obj_dir = 0

    return obj_spd, obj_dir


def map_den_spd_dir(map_data):
    '''
    object map generation
    2019.10.24
    :input {map_data}
    :return {object_maps}
    '''
    hum_ext = np.zeros((np.shape(map_data)[0], np.shape(map_data)[1]))
    hum_spd = np.zeros((np.shape(map_data)[0], np.shape(map_data)[1]))
    hum_dir = np.zeros((np.shape(map_data)[0], np.shape(map_data)[1]))
    car_ext = np.zeros((np.shape(map_data)[0], np.shape(map_data)[1]))
    car_spd = np.zeros((np.shape(map_data)[0], np.shape(map_data)[1]))
    car_dir = np.zeros((np.shape(map_data)[0], np.shape(map_data)[1]))

    hum_position = np.argwhere(map_data[:, :, 0] > 0)
    hum_num = np.shape(hum_position)[0]

    for hum_idx in range(hum_num):
        hum_vy = map_data[hum_position[hum_idx][0], hum_position[hum_idx][1], 1]
        hum_vx = map_data[hum_position[hum_idx][0], hum_position[hum_idx][1], 2]
        obj_spd, obj_dir = obj_spd_dir(hum_vy, hum_vx, 0.3)
        hum_ext[hum_position[hum_idx][0], hum_position[hum_idx][1]] = 1
        hum_spd[hum_position[hum_idx][0], hum_position[hum_idx][1]] = obj_spd
        hum_dir[hum_position[hum_idx][0], hum_position[hum_idx][1]] = obj_dir
    hum_den = hum_ext * signal.convolve2d(hum_ext, np.ones((10, 10)), boundary='fill', mode='same')

    car_position = np.argwhere(map_data[:, :, 4] > 0)
    car_num = np.shape(car_position)[0]
    for car_idx in range(car_num):
        car_vy = map_data[car_position[car_idx][0], car_position[car_idx][1], 5]
        car_vx = map_data[car_position[car_idx][0], car_position[car_idx][1], 6]
        obj_spd, obj_dir = obj_spd_dir(car_vy, car_vx, 1.0)
        car_ext[car_position[car_idx][0], car_position[car_idx][1]] = 1
        car_spd[car_position[car_idx][0], car_position[car_idx][1]] = obj_spd
        car_dir[car_position[car_idx][0], car_position[car_idx][1]] = obj_dir
    car_den = car_ext * signal.convolve2d(car_ext, np.ones((20, 20)), boundary='fill', mode='same')

    return hum_num, hum_den, hum_spd, hum_dir, car_num, car_den, car_spd, car_dir


def model_cnn_initialize():
    '''
    filtering cnn model and weight load object map filtering
    2019.10.24
    :input {model & weight file}
    :return {}
    '''
    global model, model_fc2, datagen
    global n_mu, n_inv_cov, a_mu, a_inv_cov
    global cnn_time_prev

    json_name = "output/model_object_cnn_map_fc2.json"
    json_file = open(json_name, "r")
    model = model_from_json(json_file.read())
    json_file.close()
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=1. / 255)
    model_weight_file_name = 'output/weight_cnn_map_datagen_fc2.h5'
    model.load_weights(model_weight_file_name)
    model_fc2 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    print("-------------------------------------Object CNN & FC2 model ready")

    maha_name = 'output/mahalanobis_pre_calculation_map.npz'
    maha_data_load = np.load(maha_name)
    n_mu = maha_data_load['n_mu']
    n_inv_cov = maha_data_load['n_inv_cov']
    a_mu = maha_data_load['a_mu']
    a_inv_cov = maha_data_load['a_inv_cov']
    maha_data_load.close()
    print("-------------------------------------Object Mahalanobis calculation ready")

    cnn_time_prev = 0


def cnn_fc2_maha_dist(x_input):
    '''
    filtering cnn model based Mahalanobis distance for object map
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


def cnn_maha_dist_cal(map_cont):
    '''
    cnn filtering model based Mahalanobis distance calculation for object map
    2019.10.24
    :input {map_data_in}
    :return {Mahalanobis distance}
    '''
    x_in = np.reshape(map_cont, (1, np.shape(map_cont)[0], np.shape(map_cont)[1], np.shape(map_cont)[2]))
    maha_result = cnn_fc2_maha_dist(x_in)
    maha_nor = np.round(maha_result[0][0][0], 1)
    maha_abn = np.round(maha_result[1][0][0], 1)

    return maha_nor, maha_abn


def model_cnn_initialize_elv():
    '''
    cnn model and weight load for elevation map
    2019.10.24
    :input {model and weight file}
    :return {}
    '''
    global model_elv, model_fc2_elv, datagen_elv
    global n_mu_elv, n_inv_cov_elv, a_mu_elv, a_inv_cov_elv
    global cnn_time_prev_elv
    global map_elv_normal
    img = cv2.imread('output/map_elv_normal.png', cv2.IMREAD_COLOR)
    map_elv_normal = img[:, :, 1]

    json_name_elv = "output/model_cnn_map_elevation_datagen_fc2.json"
    json_file_elv = open(json_name_elv, "r")
    model_elv = model_from_json(json_file_elv.read())
    json_file_elv.close()
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model_elv.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    datagen_elv = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=1. / 255)
    model_weight_file_name_elv = 'output/weight_cnn_map_elevation_datagen_fc2.h5'
    model_elv.load_weights(model_weight_file_name_elv)
    model_fc2_elv = Model(inputs=model_elv.input, outputs=model_elv.get_layer('fc2').output)
    print("-------------------------------------Elevation CNN & FC2 model ready")

    maha_name_elv = 'output/mahalanobis_pre_calculation_map_elevation.npz'
    maha_data_load_elv = np.load(maha_name_elv)
    n_mu_elv = maha_data_load_elv['n_mu_elv']
    n_inv_cov_elv = maha_data_load_elv['n_inv_cov_elv']
    a_mu_elv = maha_data_load_elv['a_mu_elv']
    a_inv_cov_elv = maha_data_load_elv['a_inv_cov_elv']
    maha_data_load_elv.close()
    print("-------------------------------------Elevation Mahalanobis calculation ready")

    cnn_time_prev_elv = 0


def cnn_fc2_maha_dist_elv(x_input):
    '''
    cnn model based Mahalanobis distance calculation for elevation map
    2019.10.24
    :input {map_data_in}
    :return {Mahalanobis distance}
    '''
    global model_fc2_elv
    global n_mu_elv, n_inv_cov_elv, a_mu_elv, a_inv_cov_elv
    global datagen_elv

    for x_batch in datagen_elv.flow(x_input, batch_size=1):
        fc2_in = model_fc2_elv.predict(x_batch)
        break
    n_maha_dist = np.sqrt(np.dot(np.dot(fc2_in-n_mu_elv, n_inv_cov_elv), (fc2_in-n_mu_elv).T))
    a_maha_dist = np.sqrt(np.dot(np.dot(fc2_in-a_mu_elv, a_inv_cov_elv), (fc2_in-a_mu_elv).T))
    maha_dist_elv_return = [n_maha_dist, a_maha_dist]

    return maha_dist_elv_return


def position_maha_prob_calculation_elv(map_elv_img, map_elv_prb):
    '''
    cnn model based Mahalanobis distance calculation for elevation map
    2019.10.24
    :input {map_data_in}
    :return {2nd filter result}
    '''
    map_cont = np.dstack((map_elv_img, map_elv_prb))
    x_in = np.reshape(map_cont, (1, np.shape(map_cont)[0], np.shape(map_cont)[1], np.shape(map_cont)[2]))
    maha_result = cnn_fc2_maha_dist_elv(x_in)
    maha_nor = np.round(maha_result[0][0][0], 1)
    maha_abn = np.round(maha_result[1][0][0], 1)
    maha_dist_ratio = np.round_(np.clip((maha_result[0] / (maha_result[1] + 0.0001))[0, 0], 0, 100), 2)
    prob_value = int(np.min(map_elv_prb))
    position = np.unravel_index(map_elv_prb.argmin(), map_elv_prb.shape)
    normalized_maha_score = np.round(np.log(maha_dist_ratio+0.0001), 1)

    return maha_nor, maha_abn, normalized_maha_score, prob_value, position


def map_process_obj(map_in):
    '''
    object map property processing
    2019.10.24
    :input {map_data}
    :return {object results}
    '''
    now = datetime.datetime.now()
    map_data = np.nan_to_num(map_in)

    hum_num, hum_den, hum_spd, hum_dir, car_num, car_den, car_spd, car_dir = map_den_spd_dir(map_data)
    prob_map = prob_map_generation(hum_den, hum_spd, hum_dir, car_den, car_spd, car_dir)
    prob_map_multi = prob_map[:, :, 0] * prob_map[:, :, 1] * prob_map[:, :, 2]
    obj_prb_min = np.min(prob_map_multi)
    obj_prb_min_pos = np.unravel_index(np.argmin(prob_map_multi, axis=None), prob_map_multi.shape)
    obj_prb_min_normalized = int(100 * np.clip(obj_prb_min, 0, 5000) / 5000)

    obj_type = 1
    if hum_den[obj_prb_min_pos[0], obj_prb_min_pos[1]] < car_den[obj_prb_min_pos[0], obj_prb_min_pos[1]]:
            obj_type = 2

    hum_v_max = np.max(hum_spd)
    car_v_max = np.max(car_spd)
    map_image = np.dstack((30*hum_den, 60*car_den, 20*((np.round_(hum_spd+car_spd, decimals=0)))+hum_dir+car_dir))
    map_image = (np.clip(map_image, 0, 255)).astype(np.uint8)
    map_prob_image = (2.5*prob_map).astype(np.uint8)
    maha_nor, maha_abn = cnn_maha_dist_cal(np.hstack((map_image, map_prob_image)))
    obj_ID = int(2)
    if obj_prb_min_pos[1] > 80:
        obj_ID = int(1)
    if obj_prb_min_pos[0] > 80:
        obj_ID = int(3)

    return hum_num, hum_v_max, car_num, car_v_max, obj_prb_min_normalized, obj_prb_min_pos, maha_nor, maha_abn, obj_ID, obj_type


def map_process_elv(map_in):
    '''
    elevation map property processing
    2019.10.24
    :input {map_data}
    :return {elevation results}
    '''
    now = datetime.datetime.now()
    map_data = np.nan_to_num(map_in)
    map_elv_up = map_data[:, :, 7] * np.clip(map_data[:, :, 8], 0, 1)
    map_elv_obv = 255*np.clip(map_elv_up / 5, 0, 1)
    map_elv_prb = 255*np.clip((1 - np.clip(map_elv_up / 5 - map_elv_normal / 255, 0, 1) / (map_elv_normal+0.4)), 0, 1)
    elv_maha_nor, elv_maha_abn, elv_maha_score, elv_prb_min, elv_prb_min_pos = position_maha_prob_calculation_elv(map_elv_obv, map_elv_prb)
    elv_min_value = np.round(map_data[elv_prb_min_pos[0], elv_prb_min_pos[1], 7], 1)
    elv_prb_min_normalized = int(np.clip(elv_prb_min/2.5, 0, 100))
    elv_ID = int(map_data[elv_prb_min_pos[0], elv_prb_min_pos[1], 8])

    return elv_min_value, elv_prb_min_normalized, elv_prb_min_pos, elv_maha_nor, elv_maha_abn, elv_ID


def map_process_thr(map_in):
    '''
      thermal map property processing
      2019.10.24
      :input {map_data}
      :return {thermal results}
      '''
    now = datetime.datetime.now()
    map_data = np.nan_to_num(map_in)
    map_thr_up = signal.medfilt2d(map_data[:, :, 9] * np.clip(map_data[:, :, 10], 0, 1), kernel_size=3)
    thr_max = np.round(np.max(map_thr_up), 1)
    thr_max_pos = np.unravel_index(np.argmax(map_thr_up, axis=None), map_thr_up.shape)
    thr_ID = int(map_data[thr_max_pos[0], thr_max_pos[1], 10])

    return thr_max, thr_max_pos, thr_ID


def main():
    '''
      map data recieving from ROS and processing
      2019.10.24
      :input {map_data}
      :return {detected results}
      '''
    global layered_map
    rospy.init_node('osr_map_receiver')
    prob_hist_initialize()
    model_cnn_initialize()
    model_cnn_initialize_elv()

    try:
        period = rospy.get_param('~period', 1)
        map_topic = rospy.get_param('~map_topic', '/osr_map')
    except KeyError:
        print("ros parameters are not set")
    rate = rospy.Rate(period)
    sub_osr_map = rospy.Subscriber(map_topic, GridMap, sub_osr_map_cb)

    while not rospy.is_shutdown():
        if layered_map is None:
            continue
        current_layered_map = map_reducer(layered_map)

        hum_num, hum_v_max, car_num, car_v_max, obj_prb_min, obj_prb_min_pos, maha_nor, maha_abn, obj_ID, obj_type = map_process_obj(current_layered_map)
        elv_min_value, elv_prb_min_normalized, elv_prb_min_pos, elv_maha_nor, elv_maha_abn, elv_ID  = map_process_elv(current_layered_map)
        thr_max, thr_max_pos, thr_ID = map_process_thr(current_layered_map)
        rate.sleep()


if __name__ == "__main__":
    main()
