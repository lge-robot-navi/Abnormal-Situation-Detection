from torch import nn
import torch.nn.functional as F

n_in, n_h1, n_h2, n_out = 2048, 1024, 512, 1

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.l1 = nn.Linear(n_in, n_h1)
        self.relu_1 = nn.ReLU
        # self.bn1 = nn.BatchNorm1d()

        self.l2 = nn.Linear(n_h1, n_h2)
        self.relu_2 = nn.ReLU

        self.l3 = nn.Linear(n_h2, n_out)
        self.sigmoid_l = nn.Sigmoid()

        # self.FC_all = nn.Sequential(
        #
        #     nn.Linear(n_in, n_h1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(n_h1),
        #     # nn.Dropout(0.5),
        #     nn.Linear(n_h1, n_h2),
        #     nn.BatchNorm1d(n_h2),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(n_h2, n_out),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # x = self.FC_all(x)
        x = self.l1(x)
        x1 = F.relu(x)
        x = self.l2(x1)
        x = F.relu(x)
        x = self.l3(x)
        x = self.sigmoid_l(x)
        return x, x1
