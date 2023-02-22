import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import warnings
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gpytorch
import math
import tqdm

from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from utils import BI_LSE


class MLP:
    def __init__(self, in_channels=2, hidden_channels=[100, 1], device='cpu', lr=1e-2, iters=None, patience=5, test_size=0.3, weight_decay=0, frequency=1, criterion=F.binary_cross_entropy_with_logits):
        self.model = torchvision.ops.MLP(in_channels=in_channels, hidden_channels=hidden_channels)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.iters = iters
        self.patience = patience
        self.fitted_ = False
        self.test_size = test_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.frequency = frequency
    
    def fit(self, train_xs, train_ys):
        train_xs, val_xs, train_ys, val_ys = train_test_split(
            train_xs, train_ys, test_size=self.test_size, random_state=42
        )
        train_xs = torch.from_numpy(train_xs).to(dtype=torch.float32, device=self.device)
        train_ys = torch.from_numpy(train_ys).to(dtype=torch.float32, device=self.device)
        val_xs = torch.from_numpy(val_xs).to(dtype=torch.float32, device=self.device)
        val_ys = torch.from_numpy(val_ys).to(dtype=torch.float32, device=self.device)

        acc_best = 0
        patience_akk = 0

        while self.iters is None:
            iters_cnt = 0
            # run for 'frequency' number of times before validating
            while iters_cnt < self.frequency:
                iters_cnt += 1
                self.train_epoch(train_xs, train_ys)

            acc = self.test(val_xs, val_ys)

            # if our accuracy goes down, increment the patience accumulator
            if acc_best >= acc:
                patience_akk += 1
            # or reset the accumulator and the best accuracy
            else:
                patience_akk = 0
                acc_best = acc
                save_obj = deepcopy(self.model.state_dict())

            if patience_akk >= self.patience:
                self.model.load_state_dict(save_obj)
                return None

        if self.iters is not None:
            for _ in range(self.iters):
                self.train_epoch(train_xs, train_ys)

        self.fitted_ = True
        
    def train_epoch(self, xs, ys):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(xs)
        loss = self.criterion(output.squeeze(-1), ys)
        loss.backward()
        self.optimizer.step()

    def test(self, xs, ys):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = self.model(xs)
            if self.hidden_channels[-1]==1:
                predicted = torch.gt(outputs.data, 0).squeeze(-1)
            else:
                predicted = outputs.data.argmax(-1)
        return np.sum(np.equal(predicted.numpy(), ys.numpy())) / ys.shape[0]
    
    def estimate_dropout_BI(self, xs, dropout=0.5, n_ens=10):

        xs = torch.from_numpy(xs).to(dtype=torch.float32, device=self.device)        
        self.model.train()
        state_dict = deepcopy(self.model.state_dict())
        self.model = torchvision.ops.MLP(in_channels=self.in_channels, hidden_channels=self.hidden_channels, dropout=dropout)
        self.model.load_state_dict(state_dict)
        
        preds = [self.model(xs).squeeze(-1).detach().numpy() for _ in range(n_ens)]
        preds = np.array(preds).T
        return np.array([BI_LSE(zs, bound='lower') for zs in preds])

    def predict(self, xs):
        xs = torch.from_numpy(xs).to(dtype=torch.float32, device=self.device)        
        self.model.eval()        
        return self.model(xs).squeeze(-1).detach().numpy()

    def score(self, xs, ys):
        xs = torch.from_numpy(xs).to(dtype=torch.float32, device=self.device)
        ys = torch.from_numpy(ys).to(dtype=torch.float32, device=self.device)
        return self.test(xs, ys)