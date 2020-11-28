import os
import copy
import logging
import keras
import glob
import math
import random
import numpy as np
import scipy as sp
import osr_functions as ofn
from keras import optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

global map_x_size, map_y_size, map_layer_num
map_x_size = 42
map_y_size = 35
map_layer_num = 6

print(device_lib.list_local_devices())

# ------------------- Network Define --------------------------------------
ks = 2
fn = 4

model_cnn = Sequential()
model_cnn.add(Conv2D(fn, (ks, ks), padding='same', input_shape=(map_y_size, map_x_size, map_layer_num), name="input_35x42x6"))
model_cnn.add(BatchNormalization())
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn.add(Conv2D(fn*2, (ks, ks), padding='same'))
model_cnn.add(BatchNormalization())
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn.add(Conv2D(fn*2, (ks, ks), padding='same'))
model_cnn.add(BatchNormalization())
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn.add(Conv2D(fn*4, (ks, ks), padding='same'))
model_cnn.add(BatchNormalization())
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn.add(Conv2D(fn*4, (ks, ks), padding='same'))
model_cnn.add(BatchNormalization())
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn.add(Flatten())
model_cnn.add(Dense(fn*2, activation='relu'))
model_cnn.add(BatchNormalization())
model_cnn.add(Dense(fn, activation='relu'))
model_cnn.add(BatchNormalization())
model_cnn.add(Dense(2, name="fc2"))
model_cnn.add(Activation('softmax'))

adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=1./255)
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_json = model_cnn.to_json()
with open("output/model_cnn_obj_map.json", "w") as json_file:
    json_file.write(model_json)
print(model_cnn.summary())

# --------------------- Weight Initialize -----------------------------------------------
try:
    model_cnn.load_weights('output/weight_cnn_obj_map.h5')
    print('Pre-trained weight loaded')
except:
    print('No pre-trained weight')

while 1:
    # ------------------- Feedback Training Data Read --------------------------------------
    image_files_nor_org = glob.glob('mapDB_v2/0/obj/*.*')
    image_files_nor_mov = glob.glob('mapDB_v2/moving_agent/obj/*.*')
    image_files_nor_fdb = glob.glob('mapDB_v2/feedback/caution/obj/*.*')
    print('Object CNN DB:', np.shape(image_files_nor_org), np.shape(image_files_nor_mov), np.shape(image_files_nor_fdb))
    image_files_nor = image_files_nor_org + image_files_nor_mov + image_files_nor_fdb
    x_normal_train = ofn.data_read_dstack(image_files_nor)

    image_files_abn_syn = glob.glob('mapDB_v2/1/obj/*.*')
    # image_files_abn_fdb = glob.glob('mapDB/1_feedback/map_obj_single/*.*')
    # image_files_abn = image_files_abn_syn + image_files_abn_fdb
    image_files_abn = image_files_abn_syn 
    x_abnormal_train = ofn.data_read_dstack(image_files_abn)
    print('Object CNN TrainDB: Nor', np.shape(x_normal_train), 'Abn', np.shape(x_abnormal_train))
    x_train = np.concatenate((x_normal_train, x_abnormal_train), axis=0)
    y_train = keras.utils.to_categorical(np.concatenate((np.zeros((np.shape(x_normal_train)[0], 1)), np.ones((np.shape(x_abnormal_train)[0], 1))), axis=0), num_classes=2)

    # ------------------- Network Train --------------------------------------
    datagen.fit(x_train)
    trn_bat_size = 500
    history = model_cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=trn_bat_size), steps_per_epoch=int(np.shape(x_train)[0] / trn_bat_size), epochs=200, verbose=2, callbacks=[early_stopping])
    print('Object CNN Normal Train:', np.shape(x_normal_train)[0], 'Result: ',  ofn.cnn_prediction_result(model_cnn, x_normal_train))
    print('Object CNN Abnormal Train:', np.shape(x_abnormal_train)[0], 'Result:', ofn.cnn_prediction_result(model_cnn, x_abnormal_train))

    # ------------- Mahalanobis Parameter Save----------------------------------
    model_fc2 = Model(inputs=model_cnn.input, outputs=model_cnn.get_layer('fc2').output)
    normal_train_fc2 = ofn.mahalanobis_fc2_pre_calculation(datagen, model_fc2, x_normal_train)
    n_mu = np.mean(normal_train_fc2, axis=0)
    n_inv_cov = sp.linalg.inv(np.cov(normal_train_fc2.T))
    abnormal_train_fc2 = ofn.mahalanobis_fc2_pre_calculation(datagen, model_fc2, x_abnormal_train)
    a_mu = np.mean(abnormal_train_fc2, axis=0)
    a_inv_cov = sp.linalg.inv(np.cov(abnormal_train_fc2.T))
    np.savez_compressed('output/maha_weight_obj_map.npz', n_mu=n_mu, n_inv_cov=n_inv_cov, a_mu=a_mu, a_inv_cov=a_inv_cov)

    # ------------------- Trained Weight Save --------------------------------------
    model_cnn.save_weights('output/weight_cnn_obj_map.h5')
    print('Object CNN trained weight saved')
