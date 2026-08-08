"""Convolutional Q-network for stacked Mario game frames."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Mario pixels arrive as uint8 values in the range [0, 255].
        state = state.float() / 255.0
        return self.head(self.features(state))
