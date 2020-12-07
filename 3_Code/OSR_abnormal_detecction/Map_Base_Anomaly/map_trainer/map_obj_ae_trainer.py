import copy
import glob
import random
import math
import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Input, Activation, Flatten, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras.models import model_from_json
import osr_functions as ofn


global map_x_size, map_y_size, map_layer_num

map_x_size = 42
map_y_size = 35
map_z_ae_size = 3
ks = 2
fn = 8

# ------------------- Network Define --------------------------------------
model_ae = Sequential()
model_ae.add(Conv2D(fn, (ks, ks), padding='same', input_shape=(map_y_size, map_x_size, map_z_ae_size)))
model_ae.add(BatchNormalization())
model_ae.add(Activation('relu'))
model_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model_ae.add(Conv2D(fn * 2, (ks, ks), padding='same'))
model_ae.add(BatchNormalization())
model_ae.add(Activation('relu'))
model_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model_ae.add(Conv2D(fn * 4, (ks, ks), padding='same'))
model_ae.add(BatchNormalization())
model_ae.add(Activation('relu'))
model_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model_ae.add(Conv2D(fn * 8, (ks, ks), padding='same'))
model_ae.add(BatchNormalization())
model_ae.add(Activation('relu'))
model_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model_ae.add(Conv2D(fn * 16, (ks, ks), padding='same'))
model_ae.add(BatchNormalization())
model_ae.add(Activation('relu'))
model_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model_ae.add(Flatten())

model_ae.add(Reshape((1, 1, fn * 16)))
model_ae.add(Conv2DTranspose(fn * 8, (2, 2), strides=2, activation='relu', padding='same'))
model_ae.add(BatchNormalization())
model_ae.add(Conv2DTranspose(fn * 4, (2, 2), strides=2, activation='relu', output_padding=(0, 1)))
model_ae.add(BatchNormalization())
model_ae.add(Conv2DTranspose(fn * 2, (2, 2), strides=2, activation='relu', padding='same'))
model_ae.add(BatchNormalization())
model_ae.add(Conv2DTranspose(fn, (2, 2), strides=2, activation='relu', output_padding=(1, 1)))
model_ae.add(BatchNormalization())
model_ae.add(Conv2DTranspose(map_z_ae_size, (2, 2), strides=2, activation='relu', output_padding=(1, 0)))

adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(model_ae.summary())
model_json = model_ae.to_json()
with open("output/model_ae_obj_map.json", "w") as json_file:
    json_file.write(model_json)

try:
    model_ae.load_weights('output/weight_ae_obj_map.h5')
    print('Pre-trained weight loaded')
except:
    print('No pre-trained weight')

while 1:
    image_files_nor_org = sorted(glob.glob('mapDB_v2/0/obj/*.*'), reverse=True)
    image_files_nor_mov = sorted(glob.glob('mapDB_v2/moving_agent/obj/*.*'), reverse=False)
    image_files_nor_fdb = sorted(glob.glob('mapDB_v2/feedback/caution/obj/*.*'), reverse=False)
    print('Object AE train DB:', np.shape(image_files_nor_org), np.shape(image_files_nor_mov), np.shape(image_files_nor_fdb))
    image_files = image_files_nor_org + image_files_nor_mov + image_files_nor_fdb
    x_trn_obv = ofn.data_read_separate_obj(image_files)
    print('obv:', np.shape(x_trn_obv))

    trn_bat_size = 500
    history = model_ae.fit(x_trn_obv, x_trn_obv, batch_size=trn_bat_size, epochs=100, verbose=2)
    model_ae.save_weights('output/weight_ae_obj_map.h5')
    print('Object AE trained weight saved')
