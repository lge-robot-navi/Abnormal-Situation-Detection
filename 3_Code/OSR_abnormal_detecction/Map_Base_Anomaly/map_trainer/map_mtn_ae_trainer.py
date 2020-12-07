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
map_z_ae_size = 2
ks = 2
fn = 8

# ------------------- Network Define --------------------------------------
motion_ae = Sequential()
motion_ae.add(Conv2D(fn, (ks, ks), padding='same', input_shape=(map_y_size, map_x_size, map_z_ae_size)))
motion_ae.add(BatchNormalization())
motion_ae.add(Activation('relu'))
motion_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

motion_ae.add(Conv2D(fn * 2, (ks, ks), padding='same'))
motion_ae.add(BatchNormalization())
motion_ae.add(Activation('relu'))
motion_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

motion_ae.add(Conv2D(fn * 4, (ks, ks), padding='same'))
motion_ae.add(BatchNormalization())
motion_ae.add(Activation('relu'))
motion_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

motion_ae.add(Conv2D(fn * 8, (ks, ks), padding='same'))
motion_ae.add(BatchNormalization())
motion_ae.add(Activation('relu'))
motion_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

motion_ae.add(Conv2D(fn * 16, (ks, ks), padding='same'))
motion_ae.add(BatchNormalization())
motion_ae.add(Activation('relu'))
motion_ae.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

motion_ae.add(Flatten())

motion_ae.add(Reshape((1, 1, fn * 16)))
motion_ae.add(Conv2DTranspose(fn * 8, (2, 2), strides=2, activation='relu', padding='same'))
motion_ae.add(BatchNormalization())
motion_ae.add(Conv2DTranspose(fn * 4, (2, 2), strides=2, activation='relu', output_padding=(0, 1)))
motion_ae.add(BatchNormalization())
motion_ae.add(Conv2DTranspose(fn * 2, (2, 2), strides=2, activation='relu', padding='same'))
motion_ae.add(BatchNormalization())
motion_ae.add(Conv2DTranspose(fn, (2, 2), strides=2, activation='relu', output_padding=(1, 1)))
motion_ae.add(BatchNormalization())
motion_ae.add(Conv2DTranspose(map_z_ae_size, (2, 2), strides=2, activation='relu', output_padding=(1, 0)))

adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
motion_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(motion_ae.summary())

motion_ae_json = motion_ae.to_json()
with open("output/model_ae_mtn_map.json", "w") as json_file:
    json_file.write(motion_ae_json)

try:
    motion_ae.load_weights('output/weight_ae_mtn_map.h5')
    print('Pre-trained weight loaded')
except:
    print('No pre-trained weight')

while 1:
    image_files_v1_org = sorted(glob.glob('mapDB/0/map_obj_motion/*.*'), reverse=True)
    image_files_v1_fdb = sorted(glob.glob('mapDB/feedback/caution/mtn/*.*'), reverse=True)
    image_files_v2_org = sorted(glob.glob('mapDB_v2/0/mtn/*.*'), reverse=True)
    image_files_v2_mov = sorted(glob.glob('mapDB_v2/moving_agent/mtn/*.*'), reverse=False)
    image_files_v2_fdb = sorted(glob.glob('mapDB_v2/feedback/caution/mtn/*.*'), reverse=True)

    image_files_v1 = image_files_v1_org + image_files_v1_fdb
    image_files_v2 = image_files_v2_org + image_files_v2_mov + image_files_v2_fdb
    print('Motion DB:', 'v1:', np.shape(image_files_v1), 'v2:', np.shape(image_files_v2))
    image_files = image_files_v1 + image_files_v2
    x_in, x_out = ofn.data_read_for_motion_prediction(image_files)
    print('Motion Train DB shape:', np.shape(x_in), np.shape(x_out))

    history = motion_ae.fit(x_in, x_out, batch_size=500, epochs=100, verbose=2)
    motion_ae.save_weights('output/weight_ae_mtn_map.h5')
    print('Motion preiction AE trained weight saved')


