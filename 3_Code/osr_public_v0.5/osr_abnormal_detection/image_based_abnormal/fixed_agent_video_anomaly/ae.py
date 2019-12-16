import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.datasets import *
from keras import backend as K
import datetime
from preprocess_data import *

class AE:
    def __init__(self):
        self.input_width = 45
        self.input_height = 45
        self.c_dim = 1
        self.filters = 32
        self.latent_dim = 32

    def build_encoder(self):
        inp = Input(shape=(self.input_height, self.input_width, self.c_dim))
        _ = Conv2D(filters=self.filters, kernel_size=5, strides=1, padding='same', activation='relu')(inp)
        _ = BatchNormalization()(_)
        _ = Conv2D(filters=self.filters*2, kernel_size=5, strides=3, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        _ = Conv2D(filters=self.filters*4, kernel_size=5, strides=1, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        _ = Conv2D(filters=self.filters*4, kernel_size=5, strides=3, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        _ = Flatten()(_)
        out = Dense(self.latent_dim)(_)
        
        return Model(inp, out)
        
    def build_decoder(self):
        inp = Input(shape=(self.latent_dim,))
        _ = Dense(self.filters*(self.input_height//9)*(self.input_width//9), activation='relu')(inp)
        _ = BatchNormalization()(_)
        _ = Reshape((self.input_height//9, self.input_width//9, self.filters))(_)
        _ = Conv2DTranspose(filters=self.filters*4, kernel_size=5, strides=3, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        _ = Conv2DTranspose(filters=self.filters*4, kernel_size=5, strides=1, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        _ = Conv2DTranspose(filters=self.filters*2, kernel_size=5, strides=3, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        _ = Conv2D(filters=self.filters, kernel_size=5, strides=1, padding='same', activation='relu')(_)
        _ = BatchNormalization()(_)
        out = Conv2D(filters=self.c_dim, kernel_size=5, strides=1, padding='same', activation='sigmoid')(_)
        
        return Model(inp, out)

    def build_autoencoder(self, encoder, decoder):
        inp = Input(shape=(self.input_height, self.input_width, self.c_dim,))
        _ = encoder(inp)
        out = decoder(_)
        
        return Model(inp, out)
	
ae = AE()
encoder = ae.build_encoder()
decoder = ae.build_decoder()
autoencoder = ae.build_autoencoder(encoder, decoder)

autoencoder.summary()