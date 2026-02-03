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


class AttentionBlock(nn.Module):
    """Lightweight channel attention for better feature refinement"""
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)
        
    def forward(self, x):
        # Global average pooling
        b, c, h, w = x.shape
        avg_pool = x.mean(dim=(2, 3), keepdim=True)  # (b, c, 1, 1)
        
        # Channel attention
        att = avg_pool.view(b, c)
        att = torch.relu(self.fc1(att))
        att = torch.sigmoid(self.fc2(att))
        att = att.view(b, c, 1, 1)
        
        return x * att


class MyModel(nn.Module):
    def __init__(self, n_in_channels: int):
        super().__init__()

        # Encoder: progressively extract features with residual blocks and attention
        self.encoder = nn.Sequential(
            nn.Conv2d(n_in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64),
            ResidualBlock(64),
            AttentionBlock(64),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
            AttentionBlock(128),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),
            ResidualBlock(256),
            AttentionBlock(256),
        )

        # Decoder: progressively reconstruct with skip connections via concatenation
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
            AttentionBlock(128),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64),
            ResidualBlock(64),
            AttentionBlock(64),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out
