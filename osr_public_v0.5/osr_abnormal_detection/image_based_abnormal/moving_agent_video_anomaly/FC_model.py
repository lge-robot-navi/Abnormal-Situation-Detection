from torch import nn

n_in, n_h1, n_h2, n_out = 2048, 1024, 512, 1


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.FC_all = nn.Sequential(

            nn.Linear(n_in, n_h1),
            nn.ReLU(),
            nn.BatchNorm1d(n_h1),
            # nn.Dropout(0.5),
            nn.Linear(n_h1, n_h2),
            nn.BatchNorm1d(n_h2),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(n_h2, n_out),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.FC_all(x)
        return x