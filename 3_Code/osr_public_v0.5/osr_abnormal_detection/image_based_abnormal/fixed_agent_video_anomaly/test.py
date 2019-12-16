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


from train import *
import numpy as np
from sklearn.metrics import *
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, file_name='./result/confusion_matrix.jpg'):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(-0.5, cm.shape[1], 0.5),
           yticks=np.arange(-0.5, cm.shape[0], 0.5),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(file_name)
    return ax

# load test datasets
X_test = np.load('./save/UCSDped2_dataset_patch/X_test_smooth.npy')
Y_test = np.load('./save/UCSDped2_dataset_patch/Y_test_smooth.npy')
L_test = np.load('./save/UCSDped2_dataset_patch/L_test_smooth.npy')
X_test = np.expand_dims(X_test, -1)
print(X_test.shape)

X_normal = X_test[Y_test == 0]
X_abnormal = X_test[Y_test == 1]
print("# of abnormal: {} ; # of normal: {}".format(X_abnormal.shape, X_normal.shape))

if arg.model == 'AAE':
    from aae import *
    # load model
    encoder = load_model('./save_model/ucsdped2_encoder.h5')
    decoder = load_model('./save_model/ucsdped2_decoder.h5')
    discriminator = load_model('./save_model/ucsdped2_discriminator.h5')
    generator = build_generator(encoder, decoder)

    print('AAE Test')
    batch_size = arg.batch_size
    recon_normal = generator.predict(X_normal, verbose = 1, batch_size = batch_size)
    z_normal,_,_ = encoder.predict(X_normal, verbose = 1, batch_size =batch_size)
    recon_abnormal = generator.predict(X_abnormal, verbose = 1, batch_size = batch_size)
    z_abnormal,_,_ = encoder.predict(X_abnormal, verbose = 1, batch_size =batch_size)

    X_normal_w_noise = np.clip(X_normal + np.random.normal(0, np.sqrt(0.01), size=X_normal.shape), 0, 1)
    recon_normal_w_noise = generator.predict(X_normal_w_noise , verbose = 1, batch_size = batch_size)
    z_normal_w_noise,_,_ = encoder.predict(X_normal_w_noise, verbose = 1, batch_size =batch_size)

    X_abnormal_w_noise = np.clip(X_abnormal + np.random.normal(0, np.sqrt(0.01), size=X_abnormal.shape), 0, 1)
    recon_abnormal_w_noise = generator.predict(X_abnormal_w_noise, verbose = 1, batch_size = batch_size)
    z_abnormal_w_noise,_,_ = encoder.predict(X_abnormal_w_noise, verbose = 1, batch_size =batch_size)

    normal_d = discriminator.predict(z_normal, verbose = 1, batch_size = batch_size)
    abnormal_d = discriminator.predict(z_abnormal, verbose = 1, batch_size = batch_size)
    normal_w_noise_d = discriminator.predict(z_normal_w_noise , verbose = 1, batch_size = batch_size)
    abnormal_w_noise_d = discriminator.predict(z_abnormal_w_noise , verbose = 1, batch_size = batch_size)

elif arg.model == 'AE':
    from ae import *
    # load model
    autoencoder = load_model('./save_model/ucsdped2_autoencoder.h5')
    
    print('AE Test')
    recon_normal = autoencoder.predict(X_normal, verbose=1)
    recon_abnormal = autoencoder.predict(X_abnormal, verbose=1)
    
    
# reconstruction error
y_pred_n = np.mean((X_normal - recon_normal)**2, axis=(-1,-2,-3))
y_pred_ab = np.mean((X_abnormal - recon_abnormal)**2, axis=(-1,-2,-3))
y_pred = np.concatenate([y_pred_n, y_pred_ab], axis=0)
y_true = np.array([0]*y_pred_n.shape[0] + [1]*y_pred_ab.shape[0])

if not os.path.exists('./result'):
        os.mkdir('./result')

# roc curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
area = auc(fpr, tpr)
eer_threshold = fpr[np.argmin(np.abs(fpr + tpr - 1))]
plt.figure(figsize=(10,8))
plt.plot([0,1],[0,1])
plt.plot(fpr,tpr, label='AUC: %.2f ; EER: %.2f'%(area, eer_threshold))
plt.plot([0,1],[1,0])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend()
plt.savefig('./result/roc_curve.jpg')

# confusion matrix
cm_list = []
acc_l = []
f1_l = []
for t in thresholds:
    cm = [[[],[]],[[],[]]]
    cm[0][0] = np.sum(((y_pred > t) + y_true) == 2)
    cm[1][1] = np.sum(((y_pred > t) + y_true)== 0)
    cm[0][1] = np.sum(((y_pred > t) - y_true) == -1)
    cm[1][0] = np.sum(((y_pred > t) - y_true) == 1)
    cm_list.append(cm)
    
    acc = (cm[0][0] + cm[1][1])/np.sum(cm)
    acc_l.append(acc)
    recall = cm[1][1]/(np.sum(cm[1]))
    precision = cm[1][1]/(cm[0][1] + cm[1][1])
    f1 = 2*recall*precision/(precision+recall)
    f1_l.append(f1)
    
cm_list = np.array(cm_list)
classes = ['', 'normal', '', 'abnormal']
plot_confusion_matrix(cm_list[np.argmax(acc)], classes=classes)

# reconstructed images
if arg.model == 'AAE':
    ti = 3450
    plt.figure(figsize = (10,10))
    for i in range(10):
        plt.subplot(8,10,i+1)
        plt.imshow(X_normal[i+ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+11)
        plt.imshow(recon_normal[i+ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+21)
        plt.imshow(X_normal_w_noise[i+ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+31)
        plt.imshow(recon_normal_w_noise[i+ti,:,:,0], cmap='gray')
        plt.axis('off')
    ti = 50
    for i in range(10):
        plt.subplot(8,10,i+41)
        plt.imshow(X_abnormal[i + ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+51)
        plt.imshow(recon_abnormal[i + ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+61)
        plt.imshow(X_abnormal_w_noise[i + ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+71)
        plt.imshow(recon_abnormal_w_noise[i + ti,:,:,0], cmap='gray')
        plt.axis('off')
    plt.savefig('./result/recon_imgs.jpg')

elif arg.model == 'AE':
    ti = 37650
    plt.figure(figsize = (10,10))
    # normal
    for i in range(10):
        plt.subplot(8,10,i+1)
        plt.imshow(X_normal[i+ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+11)
        plt.imshow(recon_normal[i+ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+21)
        plt.imshow(((X_normal[i+ti,:,:,0] - recon_normal[i+ti,:,:,0])**2), cmap='Reds')
        plt.axis('off')

    # abnormal
    ti = 6422
    for i in range(10):
        plt.subplot(8,10,i+31)
        plt.imshow(X_abnormal[i + ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+41)
        plt.imshow(recon_abnormal[i + ti,:,:,0], cmap='gray')
        plt.axis('off')
    for i in range(10):
        plt.subplot(8,10,i+51)
        plt.imshow(((X_abnormal[i+ti,:,:,0] - recon_abnormal[i+ti,:,:,0])**2), cmap='Reds')
        plt.axis('off')
    plt.savefig('./result/recon_imgs.jpg')
