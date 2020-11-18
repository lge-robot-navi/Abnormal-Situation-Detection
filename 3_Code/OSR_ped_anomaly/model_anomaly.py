import numpy as np
import tensorflow.keras as ke
import random
import os, sys
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import argparse
from sklearn import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from tensorflow.python.client import device_lib
# from skimage.transform import resize

# sys.path.remove('/opt/ros/kinetic/lib.python2.7/dist-packages')
# import cv2
# sys.path.append('/opt/ros/kinetic/lib.python2.7/dist-packages')

N_channel = 64  # decide the model hidden channel

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='temp_test', type=str, help='result saving folder name')
    parser.add_argument('--path', default='./data/pohang/unsu/train/0.normal/', type=str, help='path to dataset')
    parser.add_argument('--count', default=0, type=int, help='number of data to grab. default:0')
    parser.add_argument('--latent_size', default=512, type=int, help='latent space size. default: 512')
    parser.add_argument('--resume', default=False, type=bool, help='load best model. default: False')
    parser.add_argument('--input_size', default=64, type=int, help='default 64x64')
    parser.add_argument('--input_channel', default=3, type=int, help='default RGB channel')
    parser.add_argument('--train_epoch', default=2000, type=int, help='# of training epoch. default: 3000')
    parser.add_argument('--batchsize', default=32, type=int, help='Batch size. default: 50')

    opt = parser.parse_args()
    opt.path = './data/test_pohang/crop/'  # path to data

    return opt
def main():
    """ Training """
    opt = opt_parser()
    # device_lib.list_local_devices()

    opt.exp_name = 'Test_512_old_pohangDB'
    opt.path = './data/trim/train/0.normal/'

    # print("number of thread {}".format(torch.get_num_thread))

    train_model(opt)
    # test_model(opt)

    # model = init_ped_anoamly(opt)
    # imgs = random.sample(glob.glob(opt.path + '*.jpg'), 2)
    # tpatch = data_read(imgs)
    # a = test_patches(model, opt, tpatch)
    # print(a)
def build_model_enc(input_size, input_channel):
    model = ke.models.Sequential()
    model.add(ke.layers.Conv2D(N_channel, 3, input_shape=(input_size, input_size, input_channel)))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(ke.layers.Conv2D(N_channel*2, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(ke.layers.Conv2D(N_channel*4, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(ke.layers.Conv2D(N_channel*8, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())

    return model
def build_model_dec(input_channel):
    model = ke.models.Sequential()

    model.add(ke.layers.Conv2DTranspose(N_channel*4, 3, input_shape=(5, 5, N_channel*8)))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.UpSampling2D())
    model.add(ke.layers.Conv2DTranspose(N_channel*2, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.UpSampling2D())
    model.add(ke.layers.Conv2DTranspose(N_channel, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.UpSampling2D())
    model.add(ke.layers.Conv2D(input_channel, 5, activation='sigmoid'))

    return model
def build_model_checker(input_size, input_channel):
    model = ke.models.Sequential()

    model.add(ke.layers.Conv2D(N_channel, 3, input_shape=(input_size, input_size, input_channel)))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(ke.layers.Conv2D(N_channel*2, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(ke.layers.Conv2D(N_channel*4, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())
    model.add(ke.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(ke.layers.Conv2D(N_channel*8, 3))
    model.add(ke.layers.BatchNormalization())
    model.add(ke.layers.ReLU())

    model.add(ke.layers.Flatten())
    model.add(ke.layers.Dense(1, activation="sigmoid"))

    return model
def build_model_disc(latent_size):
    model = ke.models.Sequential()

    model.add(ke.layers.Flatten())
    model.add(ke.layers.Dense(latent_size, activation="linear"))
    model.add(ke.layers.Dense(64, activation="relu"))
    model.add(ke.layers.Dense(64, activation="relu"))
    model.add(ke.layers.Dense(1, activation="sigmoid"))
    return model
def build_model_aae(opt):
    model_enc = build_model_enc(opt.input_size, opt.input_channel)
    model_dec = build_model_dec(opt.input_channel)
    model_disc = build_model_disc(opt.latent_size)

    model_ae = ke.models.Sequential()
    model_ae.add(model_enc)
    model_ae.add(model_dec)

    model_checker = build_model_checker(opt.input_size, opt.input_channel)

    model_enc_disc = ke.models.Sequential()
    model_enc_disc.add(model_enc)
    model_enc_disc.add(model_disc)


    return model_enc, model_dec, model_disc, model_ae, model_enc_disc, model_checker#, model_full
def imagegrid(dec, epochnumber, exp_name, latent_size):
    fig = plt.figure(figsize=[8, 8])

    for i in range(9):
        topred = 4*(np.random.rand(latent_size)-0.5)  # TODO: Noise? latent space search?
        topred = np.reshape(topred, (1, latent_size))
        img = dec.predict(topred)
        img = img.reshape((64, 64, 3))
        ax = fig.add_subplot(3, 3, i+1)
        ax.set_axis_off()
        ax.imshow(img)

    if not os.path.isdir("./results/{0}".format(exp_name)):
        os.mkdir("./results/{0}".format(exp_name))
    fig.savefig("./results/{0}/{1}.png".format(exp_name, str(epochnumber)))
    # plt.show()
    # plt.close(fig)
def recongrid(enc, dec, epochnumber, exp_name, batch):
    fig = plt.figure(figsize=[8, 8])

    img = dec.predict(enc.predict(batch))
    for i in range(9):
        # img = img.reshape((64, 64, 3))
        ax = fig.add_subplot(3, 3, i+1)
        ax.set_axis_off()
        ax.imshow(img[i, :, :, :])

    if not os.path.isdir("./results/{0}".format(exp_name)):
        os.mkdir("./results/{0}".format(exp_name))
    fig.savefig("./results/{0}/{1}_recon.png".format(exp_name, str(epochnumber)))
    # plt.show()
    # plt.close(fig)
def settrainable(model, toset):
    for layer in model.layers:
        layer.trainable = toset
    model.trainable = toset
def data_read(image_files):
    x_train = np.zeros((len(image_files), 64, 64, 3))
    for idx, file_name in enumerate(image_files):
        # img = image.load_img(file_name, target_size=(64, 64), color_mode="grayscale")
        img = image.load_img(file_name, target_size=(64, 64))
        img = np.expand_dims(image.img_to_array(img), axis=0)
        x_train[idx, :, :, :] = img.reshape(1, 64, 64, 3)
    return x_train
def init_model(opt):
    model_enc, model_dec, model_disc, model_ae, model_enc_disc, model_checker = build_model_aae(opt)
    # model_enc.summary()
    # model_dec.summary()
    # model_disc.summary()
    # model_ae.summary()
    # model_enc_disc.summary()
    # model_checker.summary()

    model_disc.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")
    model_enc_disc.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")
    model_ae.compile(optimizer=ke.optimizers.Adam(lr=1e-3), loss="binary_crossentropy")
    model_checker.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")

    model = {
        "enc": model_enc,
        "dec": model_dec,
        "disc": model_disc,
        "ae": model_ae,
        "enc_disc": model_enc_disc,
        "checker": model_checker
    }
    # model =[model_enc, model_dec, model_disc, model_ae, model_enc_disc]

    return model
def load_best_model(model, opt):
    model['enc'].load_weights('./weights/best_enc_' + str(opt.latent_size) + '.h5')
    model['dec'].load_weights('./weights/best_dec_' + str(opt.latent_size) + '.h5')
    model['disc'].load_weights('./weights/best_disc_' + str(opt.latent_size)  + '.h5')
    model['ae'].load_weights('./weights/best_ae_' + str(opt.latent_size) + '.h5')
    model['enc_disc'].load_weights('./weights/best_enc_disc_' + str(opt.latent_size) + '.h5')
    model['checker'].load_weights('./weights/best_checker_' + str(opt.latent_size) + '.h5')

    return model
def load_data(opt): #TODO: separate normal and abnormal loading
    if opt.count:
        grab = opt.count
    else:
        grab = len(os.listdir(opt.path))
    image_files = random.sample(glob.glob(opt.path + '*.jpg'), grab)

    x_train = data_read(image_files)
    x_train = x_train.astype(float)
    x_train /= 255.0

    return x_train
def load_test(path):
    files = os.listdir(path)
    count = len(files)
    image_files = random.sample(glob.glob(path + '*.jpg'), count)
    x_test = data_read(image_files)
    x_test = x_test.astype(float)
    x_test /= 255.0

    return x_test
def save_model(model, opt): #TODO: complete save model
    path = opt.path
def check_auc(labels, pred):
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, pred, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    auc = metrics.auc(fpr1, tpr1)
    eer = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    d_f1 = np.copy(pred)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)

    return auc, eer, eer_threshold1, f1_score
def recon_score(model, x_test):
    pred = model['ae'].predict(x_test)
    loss = ke.losses.MSE(x_test, pred)
    scores = ke.backend.sum(loss, axis=2)
    scores = ke.backend.sum(scores, axis=1)
    # scores = (scores - ke.backend.min(scores))/ke.backend.max(scores)

    return scores
def test_model(opt):
    opt.path = './data/pohang/unsu/test/'
    datagen = ImageDataGenerator(rescale=1. / 255)

    test_it = datagen.flow_from_directory(opt.path, class_mode='binary', target_size=(64, 64), batch_size=1)
    # batch_x, batch_y = test_it.next()
    print('data loaded')

    model = init_model(opt)
    model = load_best_model(model, opt)
    print('model loaded')

    r_score = []
    pred_checker = []
    gt = []
    for i in tqdm(range(len(test_it))):
        batch_x, batch_y = test_it.next()
        r_score.append(recon_score(model, batch_x))
        pred_checker.append(model['checker'].predict(batch_x).squeeze())
        gt.append(batch_y)

    r_score = (r_score - min(r_score))/(max(r_score)-min(r_score))

    recon_auc = check_auc(gt, r_score)
    checker_auc = check_auc(gt, pred_checker)

    print("Recon: AUC {0:.3f}, EER {1:.3f}, EER_threshold {2:.3f}, F1_score {3:.3f}".format(recon_auc[0], recon_auc[1],recon_auc[2],recon_auc[3]))
    print("Discriminator: AUC {0:.3f}, EER {1:.3f}, EER_threshold {2:.3f}, F1_score {3:.3f}".format(checker_auc[0],checker_auc[1],checker_auc[2],checker_auc[3]))
    print('done.')
def init_ped_anoamly(opt):
    model = init_model(opt)
    print("Model Initialized")
    if opt.resume:
        model = load_best_model(model, opt)
        print("Model Succesfully Loaded.")

    return model
def test_patches(model, opt, patches):  # TODO: eval score on given patch
    max_score = 0
    scores = []

    for i in range(len(patches)):
        img = np.array(patches[i])
        img = np.resize(img, (1, 64, 64, 3))
        pred = model['checker'].predict(img)
        scores.append(pred)

    max_score = max(scores)

    # patches = patches.astype(float)
    # patches = np.asarray(patches).astype(np.float)
    # patches /= 255.0
    # for i in range(len(patches)):
    #     plt.imshow(patches[i], interpolation='nearest')
    #     plt.show()

    # TODO: define Reconstruction based anomaly score
    # opt.path = './data/pohang/unsu/test/0.normal/'
    # x_test = load_test(opt.path)  # x_test ~ 2349,64,64,3 ndarray
    # r_score = recon_score(model, patches)
    # print("max {0}, min {1}".format(ke.backend.max(r_score),ke.backend.min(r_score)))
    # for i in range(len(patches)):
    #     x_tmp = np.expand_dims(patches[i], 0)
    #     recon_loss = model['ae'].evaluate(x_tmp, x_tmp, verbose=0)
    #     # print('X={0}, Prediction={1}'.format(x_test[i], y_test))
    #     if max_score < recon_loss:
    #         max_score = recon_loss
    #     print('Recon_loss={1:.3f}, Max_score={0:.3f}'.format(max_score, recon_loss))

    return max_score
def train_model(opt):
    if not os.path.isdir('./weights/save/{0}/'.format(opt.exp_name)):
        os.mkdir('./weights/save/{0}'.format(opt.exp_name))

    model = init_ped_anoamly(opt)
    x_train = load_data(opt)
    print("Training start!")

    for cnt in range(opt.train_epoch):
        epochnumber = cnt #TODO: fix naming part
        np.random.shuffle(x_train)

        for i in tqdm(range(int(len(x_train) / opt.batchsize))):
            settrainable(model['ae'], True)
            settrainable(model['enc'], True)
            settrainable(model['dec'], True)
            settrainable(model['checker'], True)

            batch = x_train[i * opt.batchsize:i * opt.batchsize + opt.batchsize]
            model['ae'].train_on_batch(batch, batch)

            clabel_true = np.zeros((opt.batchsize, 1))
            batch_fake = model['ae'].predict(batch)
            clabel_fake = np.ones((opt.batchsize, 1))

            model['checker'].train_on_batch(batch, clabel_true)
            model['checker'].train_on_batch(batch_fake, clabel_fake)

            settrainable(model['disc'], True)
            batchpred = model['enc'].predict(batch)
            fakepred = np.random.standard_normal((opt.batchsize, batchpred.shape[1], batchpred.shape[2], batchpred.shape[3]))
            discbatch_x = np.concatenate([batchpred, fakepred])
            discbatch_y = np.concatenate([np.zeros(opt.batchsize), np.ones(opt.batchsize)])
            model['disc'].train_on_batch(discbatch_x, discbatch_y)

            settrainable(model['enc_disc'], True)
            settrainable(model['enc'], True)
            settrainable(model['disc'], False)
            model['enc_disc'].train_on_batch(batch, np.ones(opt.batchsize))

        recon_loss = model['ae'].evaluate(x_train, x_train, verbose=0)
        adv_loss = model['enc_disc'].evaluate(x_train, np.ones(len(x_train)), verbose=0)
        checker_loss = model['checker'].evaluate(x_train, np.zeros(len(x_train)), verbose=0)
        print("Epoch: {0}, Recon_loss: {1:.5f}, Adv_loss: {2:.5f}, Checker_loss: {3:.5f}".format(cnt, recon_loss, adv_loss, checker_loss))
        # print("Reconstruction Loss:", model_ae.evaluate(x_train, x_train, verbose=0))
        # print("Adverserial Loss:", model_enc_disc.evaluate(x_train, np.ones(len(x_train)), verbose=0))

        if epochnumber % 10 == 0:  # TODO: make save function. preserve best and current checkpoint
            model['enc'].save_weights(
                './weights/save/{0}/new_aae_enc_64x64x3_latent'.format(opt.exp_name) + str(opt.latent_size) + '_Epoch' + str(epochnumber) + '.h5')
            model['dec'].save_weights(
                './weights/save/{0}/new_aae_dec_64x64x3_latent'.format(opt.exp_name) + str(opt.latent_size) + '_Epoch' + str(epochnumber) + '.h5')
            model['disc'].save_weights(
                './weights/save/{0}/new_aae_disc_64x64x3_latent'.format(opt.exp_name) + str(opt.latent_size) + '_Epoch' + str(epochnumber) + '.h5')
            model['ae'].save_weights(
                './weights/save/{0}/new_aae_ae_64x64x3_latent'.format(opt.exp_name) + str(opt.latent_size) + '_Epoch' + str(epochnumber) + '.h5')
            model['enc_disc'].save_weights(
                './weights/save/{0}/new_aae_enc_disc_64x64x3_latent'.format(opt.exp_name) + str(opt.latent_size) + '_Epoch' + str(epochnumber) + '.h5')
            model['checker'].save_weights(
                './weights/save/{0}/new_aae_checker_64x64x3_latent'.format(opt.exp_name) + str(opt.latent_size) + '_Epoch' + str(epochnumber) + '.h5')

        if epochnumber % 5 == 0:
            # imagegrid(model['dec'], epochnumber, opt.exp_name, opt.latent_size)
            recongrid(model['enc'], model['dec'], epochnumber, opt.exp_name, batch)



    

if __name__ == '__main__':
    main()