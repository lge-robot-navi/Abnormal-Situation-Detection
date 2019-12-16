"""
--------------------------------------------------------------------------
    map probability probability training code
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
import numpy as np
from keras.preprocessing import image
import pickle
import copy


def data_read(image_files):
    '''
    Map data read
    2019.10.24
    :image_files {image file}
    :return {numpy array}
    '''
    x_train = np.zeros((len(image_files),
        map_y_size, map_x_size, map_layer_num))
    for idx, file_name in enumerate(image_files):
        img = image.load_img(file_name,
             target_size=(map_y_size, map_x_size)
             , color_mode='rgb')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x_train[idx, :, :, :] = img.reshape(1,
                map_y_size, map_x_size, map_layer_num)
    return x_train


global map_x_size, map_y_size, map_layer_num
map_x_size = 128
map_y_size = 128
map_layer_num = 3

prob_hist = [[[0 for _ in range(map_layer_num)] for _ in range(map_y_size)]
        for _ in range(map_x_size)]
for x in range(map_x_size):
    for y in range(map_y_size):
        prob_hist[x][y][0] = np.zeros(11)
        prob_hist[x][y][1] = np.zeros(11)
        prob_hist[x][y][2] = np.zeros(10)

image_files = glob.glob('map_input_DB\\*.bmp')
unlabel_data = data_read(image_files)

for idx in range(np.shape(unlabel_data)[0]):
    print('Probability histogram calculation:', idx, '/ ', np.shape(unlabel_data)[0])
    map_in = (np.reshape(unlabel_data[idx, :, :, :],
         (map_y_size, map_x_size, map_layer_num))/25).astype(int)

    for x in range(map_x_size):
        for y in range(map_y_size):
            if map_in[x, y, 0] == 0:
                prob_hist[x][y][0][0] += 1
            else:
                den = np.clip(map_in[x, y, 0], 0, 10)
                prob_hist[x][y][0][den] += 1

                spd = np.clip(map_in[x, y, 1], 0, 10)
                prob_hist[x][y][1][spd] += 1

                dir = np.clip(map_in[x, y, 2], 1, 9)
                prob_hist[x][y][2][dir] += 1

Prob_hist = prob_hist_smoothing(prob_hist)

with open('output\\probability_histogram_DB_python_smth', 'wb') as fp:
    pickle.dump(prob_hist, fp)
print('prob_hist_saved...')