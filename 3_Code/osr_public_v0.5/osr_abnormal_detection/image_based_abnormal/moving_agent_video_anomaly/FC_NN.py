import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.preprocessing import minmax_scale
from pprint import pprint



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



def json_data_load(json_file_name):


    with open(json_file_name) as f:
        data = json.load(f)

    total_feature_vectors = 0
    for j in range(len(data)):

        for i in data[j]["clips"]:
            total_feature_vectors += 1




    np_data =  np.array([], dtype=np.float64).reshape(0, len(data[0]["clips"][0]["features"]))
    #np_y =  np.array([], dtype=np.float64).reshape(0, total_feature_vectors)
    np_y =  np.zeros((total_feature_vectors,1))
    count = 0
    for j in range(len(data)):

        for i in data[j]["clips"]:
            #print (i["scores"])
            a = np.asarray(i["features"])
            #print(a)
            np_data = np.vstack([np_data, a])

            y_temp = i["ground_truth_annotaion"]
            #y_temp = np.asarray(i["ground_truth_annotaion"])
            y_tempp = np.uint8(y_temp)
            np_y[count] = y_tempp
            count += 1


    print('end')


    np_data = minmax_scale(np_data)



    x = torch.from_numpy(np_data)
    y = torch.from_numpy(np_y)

    x = x.float()
    y = y.float()

    return x, y


x,y = json_data_load('training_data.json')
x_test,y_test = json_data_load('testing_data.json')


# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h1, n_h2, n_out,batch_size = 2048, 1024, 512, 1, 10

# Create dummy input and target tensors (data)
#x_orig = torch.randn(batch_size, n_in)
#y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h1),
                     nn.ReLU(),
                     nn.Linear(n_h1, n_h2),
                     nn.ReLU(),
                     nn.Linear(n_h2, n_out),
                     nn.Sigmoid())


# Construct the loss function
criterion = torch.nn.MSELoss()

# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Gradient Descent
for epoch in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()

print ('lol')


