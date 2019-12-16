"""
--------------------------------------------------------------------------
    map probability inference code
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

import glob
import pickle
import numpy as np
from scipy import stats
from keras.preprocessing import image


def data_read_one_prob_concatnate(file_name):
    '''
    Map data read from concatenated map
    2019.10.24
    :file_name {image file}
    :return {numpy array}
    '''
    img = image.load_img(file_name, color_mode='rgb')
    img = image.img_to_array(img)
    img_out = img[0:map_y_size, 0:map_x_size, :]
    return img_out


def prob_map_generation(map_in):
    '''
    Probability map generation
    2019.10.24
    :map_in {numpy array}
    :return {numpy array}
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


def prob_hist_initialize():
    '''
    Probability histogram initialize
    2019.10.24
    '''
    global prob_hist
    global prob_time_prev

    prob_name = 'output\\probability_histogram_DB_pickle2'
    with open(prob_name, 'rb') as fp:
        prob_hist = pickle.load(fp)
    print("Probability histogram ready")
    prob_time_prev = 0


global map_x_size, map_y_size, map_layer_num
map_y_size = 128
map_x_size = 128
map_layer_num = 3
prob_hist_initialize()
image_files = glob.glob('map_input_DB\\*.bmp')
data_length = np.shape(image_files)[0]

for idx in range(data_length):
    map_data = data_read_one_prob_concatnate(image_files[idx])
    map_in = (map_data/25).astype(int)
    map_prob = prob_map_generation(map_in)
    print('Probability map generation:', idx)
    map_save = np.concatenate((25*map_in, 2.5 * map_prob), axis=1)
    file_name = 'map_output_DB\\ProbMap_' + str(idx) + '.bmp'
    image.save_img(file_name, map_save)

