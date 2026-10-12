"""Embedding net for the SBI posterior: CNN on the 2 x seq_len features + the 2 scalars -> 130 dims."""

import torch
from torch import nn


class CNNScalarEmbedding(nn.Module):
    def __init__(self, seq_len, n_scalars=2):
        super().__init__()
        self.seq_len = seq_len
        self.n_scalars = n_scalars
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
        )

    def forward(self, x):
        features = x[:, :-self.n_scalars].reshape(-1, 2, self.seq_len)
        scalars = x[:, -self.n_scalars:]
        return torch.cat([self.net(features).mean(dim=-1), scalars], dim=1)
