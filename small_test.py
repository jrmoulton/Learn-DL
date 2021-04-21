from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


x = pd.read_csv("Data/x_train.csv", index_col=0).values.reshape(-1, 28, 28)
y = pd.read_csv("Data/y_train.csv", index_col=0).values
x_test = pd.read_csv("Data/x_test.csv", index_col=0).values.reshape(-1, 28, 28)
y_test = pd.read_csv("Data/y_test.csv", index_col=0).values
x, y, x_test, y_test = map(torch.tensor, (x, y, x_test, y_test))
x = x.float().reshape(-1, 784)
y = y.float().argmax(1)
x_test = x_test.float().reshape(-1, 784)
y_test = y_test.float().argmax(1)

train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds, 64, shuffle=True)

P = nn.Parameter


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = P(torch.randn(784, 16, requires_grad=True) / np.sqrt(784/2))
        self.b1 = P(torch.randn(16, requires_grad=True))
        self.w2 = P(torch.randn(16, 16, requires_grad=True) / np.sqrt(784/2))
        self.b2 = P(torch.randn(16, requires_grad=True))
        self.w3 = P(torch.randn(16, 10, requires_grad=True) / np.sqrt(784/2))
        self.b3 = P(torch.randn(10, requires_grad=True))

    def forward(self, xb):
        xb = F.relu(xb @ self.w1 + self.b1)
        xb = F.relu(xb @ self.w2 + self.b2)
        xb = xb @ self.w3 + self.b3
        return xb

    def update(self, epochs):
        for i in range(epochs):
            model.train()
            for xb, yb, in train_dl:
                preds = self.forward(xb)

                loss = F.cross_entropy(preds, yb)

                loss.backward()
                opt.step()
                opt.zero_grad()


model = Model()
opt = optim.SGD(model.parameters(), 0.5)

model.update(2)
