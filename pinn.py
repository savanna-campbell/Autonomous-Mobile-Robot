import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# create dataset class to deal with csv batching
class TrajectoryData(Dataset):
    def __init__(self, csv_path, skiprows=10):
        df = pd.read_csv(csv_path, skiprows=skiprows)
        self.t = torch.tensor(df["time"].values, dtype=torch.float32).unsqueeze(1)
        self.states = torch.tensor(df[['x','y','theta']].values, dtype=torch.float32)
        self.controls = torch.tensor(df[['u_R', 'u_L']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.states[idx], self.controls[idx]

# create object to hold the csv file created by the simulation
dataset = TrajectoryData('output.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# create the neural network

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # layers
        self.hidden1 = nn.Linear(3, 20)
        self.hidden2 = nn.Linear(20,20)
        self.output = nn.Linear(20, 5)

        # trainable parameters
        self.r = nn.Parameter(torch.tensor(init_r))

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x
    
model = NeuralNet()    


# physics residual


# mdmm setup

# training loop

for features_batch, labels_batch in dataloader:
    pass