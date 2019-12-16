"""
--------------------------------------------------------------------------
    fixed video anomaly code
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

import keras.layers
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.datasets import *
from keras import backend as K
from spektral.layers import *
from geometry import ccm_normal, ccm_uniform, clip, get_distance
import datetime
from preprocess_data import *

def sampling(args):
    z_mean, z_log_var = args
    return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(shape=K.shape(z_mean))

def lap_pyramid(x, kernel, pool, max_levels = 3):
    current = x
    pyramid = []
    for level in range(max_levels):
        filtered = kernel(current)
        diff = current - filtered
        pyramid.append(diff)
        current = pool(current)
    pyramid.append(current)
    return pyramid

class AAE:
    def __init__(self):
        self.input_widths = 45
        self.input_heights = 45
        self.gf_dim = 64
        self.df_dim = 32
        self.latent_dim = 256
        self.c_dim = 1
        self.radius = 1.
        self.sigma = 5.

    def build_encoder(self):
        inp = Input((self.input_heights, self.input_widths, self.c_dim))
        _ = keras.layers.Conv2D(self.df_dim, (5, 5), strides=(1, 1), padding='same')(inp)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Conv2D(self.df_dim, (5, 5), strides=(3, 3), padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Conv2D(self.df_dim * 2, (5, 5), strides=(1, 1), padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Conv2D(self.df_dim * 2, (5, 5), strides=(3, 3), padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Flatten()(_)
        z_mean = keras.layers.Dense(self.latent_dim)(_)
        z_log_var = keras.layers.Dense(self.latent_dim)(_)
        z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
        return Model(inp, [z, z_mean, z_log_var])

    def build_decoder(self):
        inp = Input((self.latent_dim,))
        _ = keras.layers.Dense(self.gf_dim * 2 * (self.input_heights // 9) * (self.input_widths // 9))(inp)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.0)(_)
        _ = keras.layers.Reshape((self.input_heights // 9, self.input_widths // 9, self.gf_dim * 2))(_)
        _ = keras.layers.Conv2DTranspose(self.gf_dim * 2, (5, 5), strides=(3, 3), padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.0)(_)
        _ = keras.layers.Conv2D(self.gf_dim * 2, (5, 5), strides=(1, 1), padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.0)(_)
        _ = keras.layers.Conv2DTranspose(self.gf_dim, (5, 5), strides=(3, 3), padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.0)(_)
        _ = keras.layers.Conv2D(self.c_dim, (5, 5), strides=(1, 1), padding='same')(_)
        out = keras.layers.Activation('sigmoid')(_)
        return Model(inp, out)

    def build_generator(self, encoder, decoder):
        inp = Input((self.input_heights, self.input_widths, self.c_dim))
        z, _, _ = encoder(inp)
        out = decoder(z)
        return Model(inp, out)

    def build_discriminator(self):
        inp = Input((self.latent_dim,))
        _ = keras.layers.Dense(self.latent_dim)(inp)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Dense(self.latent_dim)(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Dense(self.latent_dim)(_)
        _ = keras.layers.BatchNormalization()(_)
        _ = keras.layers.LeakyReLU(0.2)(_)
        _ = keras.layers.Dense(1)(_)
        out = keras.layers.Activation('sigmoid')(_)
        return Model(inp, out)

    def build_checker(self, encoder):
        inp = Input((self.input_heights, self.input_widths, self.c_dim,))
        z,z_mean,z_log_vae = encoder(inp)
        out = CCMMembership(r = self.radius, sigma = self.sigma)(z)
        return Model(inp, out)
        
# build model
aae = AAE()
encoder = aae.build_encoder()
decoder = aae.build_decoder()
generator = aae.build_generator(encoder, decoder)
discriminator = aae.build_discriminator()

generator.summary()
discriminator.summary()

size = 5
s = 1.0
grid = np.float32(np.mgrid[0:5,0:5].T)
gaussian = lambda x: np.exp((x - size//2)**2/(-2*s**2))**2
kernel = np.sum(gaussian(grid), axis=2)
kernel /= np.sum(kernel)
kernel = np.tile(kernel, (1, 1, 1, 1))
kernel = np.rollaxis(kernel, 2)
kernel = np.rollaxis(kernel, 3)

gaussian_kernel = keras.layers.Conv2D(1, (5, 5), padding='same', kernel_initializer ='random_normal', use_bias = False)
gaussian_pooling = keras.layers.AveragePooling2D(2)
gaussian_kernel.build((None,1,1))
gaussian_kernel.trainable = False
gaussian_pooling.trainable = False

gaussian_kernel.set_weights([kernel])    
tmp = lambda x,y: K.mean(keras.layers.average([K.sum(K.abs(a - b), axis=(1, 2, 3)) for a, b in zip(x, y)]))

# set loss
gamma = 1.0

inp = Input((aae.input_heights, aae.input_widths, aae.c_dim))
inp_w_noise = Input((aae.input_heights, aae.input_widths, aae.c_dim))
inp_pyramid = lap_pyramid(inp, gaussian_kernel, gaussian_pooling)
inp_pyramid_w_noise = lap_pyramid(inp_w_noise, gaussian_kernel, gaussian_pooling)

recon = generator(inp)
recon_w_noise = generator(inp_w_noise)
out_pyramid = lap_pyramid(recon, gaussian_kernel, gaussian_pooling)
out_pyramid_w_noise = lap_pyramid(recon_w_noise, gaussian_kernel, gaussian_pooling)

z, z_mean_, z_log_var_ = encoder(inp)
kl_loss = 0
z_noise, z_mean_noise, z_log_var_noise = encoder(inp_w_noise)

inp_z = Input((aae.latent_dim,))

D_z = discriminator(inp_z)
M_z = CCMMembership(r=aae.radius, sigma=aae.sigma)(inp_z)
D_z = (D_z * 9 + M_z) / 10

D_real = discriminator(z)
M_real = CCMMembership(r=aae.radius, sigma=aae.sigma)(z)
D_real = (D_real * 9 + M_real) / 10

D_fake = discriminator(z_noise)
M_fake = CCMMembership(r=aae.radius, sigma=aae.sigma)(z_noise)
D_fake = (D_fake * 9 + M_fake) / 10

d_loss_z = K.binary_crossentropy(K.ones_like(D_real), D_z)
d_loss_real = K.binary_crossentropy(K.ones_like(D_real), D_real)
d_loss_fake = K.binary_crossentropy(K.zeros_like(D_fake), D_fake)
d_loss = K.mean(d_loss_real + d_loss_fake + d_loss_z) / 3

g_loss_ = K.binary_crossentropy(K.ones_like(D_fake), D_fake)
g_loss_r = tmp(inp_pyramid, out_pyramid) + tmp(inp_pyramid_w_noise, out_pyramid_w_noise)
g_loss_r2 = tmp(inp_pyramid, out_pyramid) + tmp(inp_pyramid, out_pyramid_w_noise)
g_loss = tmp(inp_pyramid, out_pyramid) / 10
g_loss1 = K.mean(g_loss_ / 2 + g_loss_r / 10 + kl_loss)
g_loss2 = K.mean(g_loss_r2) / 10

d_updates = Adam(lr = 1.0e-04, beta_1 = 0.5).get_updates(discriminator.trainable_weights, [], d_loss)
D_train = K.function([inp, inp_w_noise, inp_z], [d_loss], d_updates)

# generator
g_updates = Adam(lr = 1.0e-04, beta_1 = 0.5).get_updates(generator.trainable_weights, [], g_loss)
G_train = K.function([inp, inp_w_noise, inp_z], [g_loss], g_updates)

# encoder: adversarial & honest recon.
g_updates = Adam(lr = 1.0e-04, beta_1 = 0.5).get_updates(encoder.trainable_weights, [], g_loss1)
G_train_en = K.function([inp, inp_w_noise], [g_loss1], g_updates)

# decoder: denoising
g_updates = Adam(lr = 1.0e-04, beta_1 = 0.5).get_updates(decoder.trainable_weights, [], g_loss2)
G_train_de = K.function([inp, inp_w_noise], [g_loss2], g_updates)        