import torch
import torch.nn as nn
from torch.autograd import Variable
import ujson as json
import numpy as np
from sklearn.preprocessing import minmax_scale
from pprint import pprint
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import mat4py as mpy

batch_size = 128
#num_epochs = 150
#
# CUDA_VISIBLE_DEVICES = 1


#
# nb_samples = 100
# features = torch.randn(nb_samples, 10)
# labels = torch.empty(nb_samples, dtype=torch.long).random_(10)
# adjacency = torch.randn(nb_samples, 5)
# laplacian = torch.randn(nb_samples, 7)
#
# dataset = data_utils.TensorDataset(features, labels, adjacency, laplacian)
# loader = data_utils.DataLoader(
#     dataset,
#     batch_size=2
# )
#
# for batch_idx, (x, y, a, l) in enumerate(loader):
#     print(x.shape, y.shape, a.shape, l.shape)



#pprint(data)
# print (len(data))  # total number of videos
# print (len(data[0]["clips"][0]["features"]))  # total number of features from a video



def __init__(self, data, labels):
    #initialization
    self.data = data
    self.labels = labels

def __len__(self):
    #'Denotes the total number of samples'
    return len(self.data)

def __getitem__(self, index):
    # Select sample
    d = self.data[index]

    # Load data and get label
    x = torch.load('data/' + d + '.pt')
    y = self.labels[d]

    return x, y




def check_accuracy(output, real_labels):

    output = output.cpu()

    temp_output = output.detach().numpy().astype(float)

    real_labels = real_labels.cpu()

    output_temp = output >= 0.5
    output_labels = output_temp.numpy().astype(int)
    y = real_labels.numpy()

    similar = np.isclose(output_labels, y).astype(int)

    return sum(similar) * 100 / real_labels.size(0)


#a_x,b_y = json_data_load('ucf_train.json')


def main_function(num_epochs, learning_rate, decay_rate, momentum):


    x = torch.from_numpy(np.load('x_train_data.npy')).float()
    y = torch.from_numpy(np.load('y_train_data.npy')).float()
    x_test = torch.from_numpy(np.load('x_test_data.npy')).float()
    y_test = torch.from_numpy(np.load('y_test_data.npy')).float()
    #
    # x = torch.from_numpy(np.load('x_ucf_cliffdiving_wallpushups_train_data.npy')).float()
    # y = torch.from_numpy(np.load('y_ucf_cliffdiving_wallpushups_train_data.npy')).float()
    # x_test = torch.from_numpy(np.load('x_ucf_cliffdiving_wallpushups_test_data.npy')).float()
    # y_test = torch.from_numpy(np.load('y_ucf_cliffdiving_wallpushups_test_data.npy')).float()

    # shit = np.load('shit_1_test.npy')
    # shit2 = np.load('shit_2_test.npy')
    # shit3 = np.load('x_train_data_1.npy')
    # shit4 = np.load('x_train_data.npy')
    # shit5 = np.load('x_ucf_fighting_arrest_train_data.npy')
    # x_max = max(shit.all())

    # x = torch.randn(26083, 2048)
    # x_test = torch.randn(5820, 2048)

    train_dataset = data_utils.TensorDataset(x, y)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True
    )


    test_dataset = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size,
        shuffle=False
    )

    # for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
    #     print(x_batch.shape, y_batch.shape)




    # Defining input size, hidden layer size, output size and batch size respectively
    n_in, n_h1, n_h2,  n_out = 2048, 1024, 512, 1

    model = nn.Sequential(nn.Linear(n_in, n_h1),
                         nn.ReLU(),
                         nn.BatchNorm1d(n_h1),
                         nn.Dropout(0.5),
                         nn.Linear(n_h1, n_h2),
                         nn.BatchNorm1d(n_h2),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(n_h2, n_out),
                         nn.Sigmoid())


    model.cuda()

    model.modules[3].output

    # Construct the loss function
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.BCELoss()
    # criterion = torch.nn.NLLLoss()

    # Construct the optimizer (Stochastic Gradient Descent in this case)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay_rate, momentum=momentum)

    x_test = Variable(x_test.cuda())
    y_test = Variable(y_test.cuda())
    x = Variable(x.cuda())
    y = Variable(y.cuda())


    output_test = model(x_test)

    loss_hist = []
    test_loss_hist = []




    plt.axis([0, num_epochs, 0, 1])



    # plt.title("Training Error Graph")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Training Error")
    # plt.show(block=False)

    # Gradient Descent
    for epoch in range(num_epochs):
        train_loss = []
        #model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):

            # Forward pass: Compute predicted y by passing x to the model

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            y_pred = model(x_batch)

            # Compute and print loss
            loss = criterion(y_pred, y_batch)


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

            train_loss.append(loss.item())
            #print('batch accuracy: ', check_accuracy(y_pred, y_batch))


        #model.eval()
        output = model(x)
        #print('epoch: ', epoch)
        epoch_loss = sum(train_loss) / len(train_loss)

        loss_hist.append(epoch_loss)

        #print('loss: ', epoch_loss)

        #print('train accuracy: ', check_accuracy(output, y))


        # plt.draw(epoch, epoch_loss)


        output_test = model(x_test)
        test_loss = criterion(output_test, y_test).cpu().detach().numpy()
        test_loss_hist.append(test_loss.item())
        #print('test accuracy: ', check_accuracy(output_test, y_test))



        ## Code for run time plot update
        ##uncomment this for runtime plotting
        # plt.title("Training Error Graph")
        # plt.xlabel("Training Epochs")
        # plt.ylabel("Training Error")
        # plt.scatter(epoch, epoch_loss, linewidths=0.5, c='red', marker='.')
        # plt.scatter(epoch, test_loss,  linewidths=0.5, c='blue', marker='.')
        # plt.pause(0.00001)


    #plt.show()

    ##uncomment until here for runtime plotting

    output_test = model(x_test)

    print('test accuracy: ', check_accuracy(output_test, y_test))
    model.eval()


    output_test = model(x_test)
    test_accuracy = check_accuracy(output_test, y_test)

    print('test accuracy: ', test_accuracy)



    ohist = []
    shist = []

    ohist = [h for h in loss_hist]
    shist = [h for h in test_loss_hist]
    # ohist = [h.cpu().numpy() for h in loss_hist]
    # shist = [h.cpu().numpy() for h in scratch_hist]



    name_of_fig = 'tst_acc_'+str(test_accuracy) + 'epch_'+str(num_epochs)+'_lr_'+str(learning_rate) + '_moment_'+str(momentum) + '_decay_rate_'+str(decay_rate)

    plt.title(name_of_fig)
    plt.xlabel("Training Epochs")
    plt.ylabel("Error")
    plt.plot(range(1,num_epochs+1),ohist,label="train")
    plt.plot(range(1,num_epochs+1),shist,label="test")
    #plt.ylim((0,1.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig('./fig/'+name_of_fig+'.png')
    plt.clf()
    #plt.show()
    return test_accuracy


num_epochs = 300
learning_rate = 0.001
decay_rate = 0.01
momentum = 0.9
accuracy = main_function(num_epochs=num_epochs, learning_rate=learning_rate, decay_rate=decay_rate, momentum=momentum)
print ('test_accuracy = ', accuracy)

#
# epochs = {150, 250,  350, 400}
# lr = {0.0001, 0.0003, 0.0005, 0.0009, 0.001, 0.003, 0.005, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.2, 0.5, 0.9}
# momen = {0.01, 0.001, 0.0001, 0.05, 0.005, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95}
# decay_r = {1.0, 0.9, 0.5, 0.1, 0.09, 0.05, 0.01, 0.009, 0.005, 0.001}
# total_loop = len(epochs)*len(lr)*len(momen)*len(decay_r)
# best_lr = 0
# best_momentum = 0
# best_decay_rate = 0
# best_epochs = 0
# best_accuracy = 0
# for num_epochs in epochs:
#     for learning_rate in lr:
#         for momentum in momen:
#             for decay_rate in decay_r:
#                 print('remaining runs: %d num_epochs: %d learning_rate %f momentum %f decay_rate %f' %(total_loop,num_epochs,learning_rate, momentum, decay_rate))
#                 total_loop = total_loop-1
#                 accuracy = main_function(num_epochs=num_epochs, learning_rate=learning_rate, decay_rate=decay_rate, momentum=momentum)
#                 if best_accuracy < accuracy:
#                     best_accuracy = accuracy
#                     best_lr = learning_rate
#                     best_epochs = num_epochs
#                     best_momentum  = momentum
#                     best_decay_rate = decay_rate
#                     print('So far BEST: best_accuracy: %f num_epochs: %d learning_rate %f momentum %f decay_rate %f' % (
#                     best_accuracy, best_epochs, best_lr, best_momentum, best_decay_rate))
#
#
