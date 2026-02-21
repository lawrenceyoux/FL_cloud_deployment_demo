"""
model.py - Shared model definition used by all FL components.
"""

import torch
import torch.nn as nn


class StrokeNet(nn.Module):
    """Simple MLP for stroke prediction. Input dim auto-detected from data."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_parameters(model: nn.Module):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters):
    import numpy as np
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state, strict=True)
