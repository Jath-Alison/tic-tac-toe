"""Simple tic-tac-toe Model definition"""

import torch
from torch import nn


class QNetwork(nn.Module):
    """Estimate a Q-value for each of the nine squares."""

    def __init__(self, hidden_size_1: int = 64, hidden_size_2: int = 64):
        super().__init__()
        # Define: 18 → hidden_size → hidden_size → 9
        # Use ReLU between hidden layers.
        self.hidden_1 = nn.Linear(18, hidden_size_1)
        self.hidden_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.output = nn.Linear(hidden_size_2, 9)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:  (18,) or (batch_size, 18)
        Output: (9,)  or (batch_size, 9)

        Return raw Q-values without softmax or legal-move masking.
        """
        x = torch.relu(self.hidden_1(features))  # 18 → 8
        x = torch.relu(self.hidden_2(x))      # 8 → 8
        return self.output(x)              # 8 → 9
