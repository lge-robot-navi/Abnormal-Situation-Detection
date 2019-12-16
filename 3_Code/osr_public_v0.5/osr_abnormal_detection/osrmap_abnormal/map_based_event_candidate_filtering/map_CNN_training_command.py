"""
--------------------------------------------------------------------------
    map cnn filter training code
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

import keras
import os
import glob
import math
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from keras.preprocessing import image
from keras import optimizers
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def data_read(image_files):
    '''
    cnn training data read
    2019.10.24
    :image_files {image file}
    :return {numpy array}
    '''
    x_train = np.zeros((len(image_files), map_y_size, map_x_size, map_layer_num))
    for idx, file_name in enumerate(image_files):
        img = image.load_img(file_name, color_mode='rgb')
        img = image.img_to_array(img)
        x_train[idx, :, :, :] = np.expand_dims(img, axis=0)
    return x_train


def cnn_prediction_result(x_train):
    '''
    cnn prediction calculation
    2019.10.24
    :image_files {numpy array}
    :return {prediction result}
    '''
    predict_result = 0
    for idx in range(np.shape(x_train)[0]):
        x_input = np.reshape(x_train[idx, :, :, :], (1, map_y_size, map_x_size, map_layer_num))
        for x_batch in datagen.flow(x_input, batch_size=1):
            predict_result = predict_result + model.predict_classes(x_batch)
            break
    return (predict_result/np.shape(x_train)[0])[0]


def mahalanobis_fc2_pre_calculation(datagen, model_fc2, x_train):
    '''
    cnn prediction calculation
    2019.10.24
    :image_files {numpy array}
    :return {prediction result}
    '''
    train_fc2 = np.zeros((np.shape(x_train)[0], 2))
    for idx in range(np.shape(x_train)[0]):
        x_input = np.reshape(x_train[idx, :, :, :], (1, map_y_size, map_x_size, map_layer_num))
        for x_batch in datagen.flow(x_input, batch_size=1):
            result = model_fc2.predict(x_batch)
            break
        train_fc2[idx, 0] = result[0, 0]
        train_fc2[idx, 1] = result[0, 1]
    return train_fc2


global map_x_size, map_y_size, map_layer_num
map_x_size = 128
map_y_size = 64
map_layer_num = 3

print(device_lib.list_local_devices())

# <editor-fold desc="[Network Define]">
# ------------------- Network Define and Reinitialize--------------------------------------
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(map_y_size, map_x_size, map_layer_num), name="input_128x256x3"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2, name="fc2"))
model.add(Activation('softmax'))

adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_json = model.to_json()
with open("output\\model_cnn_map_datagen_fc2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('output\\weight_cnn_map_datagen_fc2_initialized.h5')
print(model.summary())
# </editor-fold>

while 1:
    # --------------------- Weight Initialize -----------------------------------------------
    try:
        model.load_weights('output\\weight_cnn_map_datagen_fc2.h5')
        print('Pre-trained weight loaded')
    except:
        print('No pre-trained weight')

    # ------------------- Feedback Training Data Read --------------------------------------
    image_files = glob.glob('mapDB\\train\\cnn\\0_aae_sort\\*.bmp')
    x_normal_train = data_read(image_files)
    image_files = glob.glob('mapDB\\train\\cnn\\1_aae_sort\\*.bmp')
    x_abnormal_train = data_read(image_files)

    normal_length = np.shape(x_normal_train)[0]
    abnormal_length = np.shape(x_abnormal_train)[0]
    x_train = np.concatenate((x_normal_train, x_abnormal_train), axis=0)
    y_train = keras.utils.to_categorical(np.concatenate((np.zeros((normal_length, 1)), np.ones((abnormal_length, 1))), axis=0), num_classes=2)

    # ------------------- Network Train --------------------------------------
    datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=1./255)
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=1000), steps_per_epoch=math.ceil((normal_length + abnormal_length)/1000), epochs=1000, verbose='0')
    print('Normal length:', normal_length, '\t', 'Normal_train_result: ',  cnn_prediction_result(x_normal_train))
    print('Abnormal length:', abnormal_length, '\t', 'Abnormal_train_result:', cnn_prediction_result(x_abnormal_train))

    # ------------- Mahalanobis Parameter Save----------------------------------
    model_fc2 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    normal_train_fc2 = mahalanobis_fc2_pre_calculation(datagen, model_fc2, x_normal_train)
    n_mu = np.mean(normal_train_fc2, axis=0)
    n_inv_cov = sp.linalg.inv(np.cov(normal_train_fc2.T))
    abnormal_train_fc2 = mahalanobis_fc2_pre_calculation(datagen, model_fc2, x_abnormal_train)
    a_mu = np.mean(abnormal_train_fc2, axis=0)
    a_inv_cov = sp.linalg.inv(np.cov(abnormal_train_fc2.T))
    np.savez_compressed('output\\mahalanobis_pre_calculation_map.npz', n_mu=n_mu, n_inv_cov=n_inv_cov, a_mu=a_mu, a_inv_cov=a_inv_cov)

    # ------------------- Trained Weight Save --------------------------------------
    model.save_weights('output\\weight_cnn_map_datagen_fc2.h5')
    print('=========================================cnn_weigh_saved=====================================================')
