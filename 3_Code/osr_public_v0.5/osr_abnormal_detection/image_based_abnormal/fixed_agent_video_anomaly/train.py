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

from preprocess_data import *

if __name__ == '__main__':
    # load train datasets
    X_train = np.load('./save/UCSDped2_dataset_patch/X_train.npy')
    X_train = np.expand_dims(X_train, -1)
    print(X_train.shape)

    n_batch = X_train.shape[0] // arg.batch_size

    if not os.path.exists('./save_model'):
        os.mkdir('./save_model')

    print('Training Start ', datetime.datetime.now())
    if arg.model =='AAE':
        from aae import *
        for e in range(arg.epochs):
            d_loss_sum = 0
            g_loss_sum = 0
            enc_loss_sum = 0
            dec_loss_sum = 0
            for b in range(n_batch):
                imgs_idx = np.random.choice(np.arange(X_train.shape[0]), arg.batch_size)
                imgs = X_train[imgs_idx]
                if aae.radius > 0:
                    noise_z = ccm_uniform(arg.batch_size, dim=aae.latent_dim, r=aae.radius)
                else:
                    noise_z = ccm_normal(arg.batch_size, dim=aae.latent_dim, r=aae.radius)

                imgs_w_noise = np.clip(imgs + np.random.normal(0, np.sqrt(0.01), size = imgs.shape), 0, 1)
                d_loss_sub, = D_train([imgs, imgs_w_noise, noise_z])
                d_loss_sum += d_loss_sub
                imgs_w_noise = np.clip(imgs + np.random.normal(0, np.sqrt(0.01), size = imgs.shape), 0, 1)
                g_loss_sub, = G_train([imgs, imgs_w_noise, noise_z])
                g_loss_sum += g_loss_sub
                imgs_w_noise = np.clip(imgs + np.random.normal(0, np.sqrt(0.01), size = imgs.shape), 0, 1)
                enc_loss_sub, = G_train_en([imgs, imgs_w_noise])
                enc_loss_sum += enc_loss_sub
                imgs_w_noise = np.clip(imgs + np.random.normal(0, np.sqrt(0.01), size = imgs.shape), 0, 1)
                dec_loss_sub, = G_train_de([imgs, imgs_w_noise])
                dec_loss_sum += dec_loss_sub

            print(datetime.datetime.now(), "Epochs: %03d/%03d ; D_loss: %.06f ; G_loss: %0.6f ; enc_loss: %0.6f ; dec_loss: %0.6f"\
                  %(e+1, arg.epochs, d_loss_sum/n_batch, g_loss_sum/n_batch, enc_loss_sum/n_batch, dec_loss_sum/n_batch))
            
            # save model
            encoder.save('./save_model/ucsdped2_encoder.h5')
            decoder.save('./save_model/ucsdped2_decoder.h5')
            discriminator.save('./save_model/ucsdped2_discriminator.h5')
            
    elif arg.model == 'AE':
        from ae import *
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
        autoencoder.fit(X_train, X_train, epochs=arg.epochs, batch_size=arg.batch_size, shuffle=True)
        print(datetime.datetime.now())
        
        # save model
        autoencoder.save('./save_model/ucsdped2_autoencoder.h5')
    
    else:
        print('model not exist')
        