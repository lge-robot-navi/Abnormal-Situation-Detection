import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        self.imgs = np.arange(len(data))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def load_data(opt, data_in):
    numpy_label = np.zeros((np.shape(data_in)[0]))
    dataset = MyDataset(data_in, numpy_label)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers), drop_last=False)
    return dataloader
