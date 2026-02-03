"""
Author: Your Name
HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
architecture.py
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))


class MyModel(nn.Module):
    def __init__(self, n_in_channels: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(64),
            ResidualBlock(64),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out
