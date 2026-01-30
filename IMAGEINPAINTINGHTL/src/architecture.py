"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch

class MyModel(torch.nn.Module):
    # TODO: Implement the model architecture.
    def __init__(self, n_in_channels: int):
        super().__init__()

        self.cnn1 = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=6,
            kernel_size=3,
            padding=1
        )

        self.relu = torch.nn.ReLU()

        self.cnn2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        return x
    pass