"""Agent definition for Q-network tic-tac-toe agent"""

import numpy as np
import torch

from tic_tac_toe.model import QNetwork


class Agent:
    """Choose legal actions using a Q-network and exploration."""

    def __init__(
        self,
        model: QNetwork,
        seed: int | None = None,
    ):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def encode(self, board: np.ndarray) -> torch.Tensor:
        """
        Convert the (3, 3, 2) board into 18 float32 values.
        Place the tensor on the same device as the model.
        """
        device = next(self.model.parameters()).device

        return torch.as_tensor(
            board,
            dtype=torch.float32,
            device=device,
        ).reshape(18)

    def select_action(
        self,
        board: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        """
        With probability epsilon, sample a legal move.
        Otherwise, choose the highest-scoring legal move.

        Reject invalid epsilon values and masks with no legal moves.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        if self.rng.random() < epsilon:
            return self.random_action(action_mask)

        return self.greedy_action(board, action_mask)
        

    def random_action(self, action_mask: np.ndarray) -> int:
        """Sample uniformly from legal squares."""
        legal_actions = np.flatnonzero(action_mask)
        
        if legal_actions.size == 0:
            raise ValueError("No legal moves available.")

        return int(self.rng.choice(legal_actions))

    @torch.no_grad()
    def greedy_action(
        self,
        board: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Run the model, mask illegal moves, and return argmax."""
        if not np.any(action_mask):
            raise ValueError("No legal moves available.")

        features = self.encode(board)
        q_values = self.model(features)

        legal_mask = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=q_values.device,
        )

        masked_values = q_values.masked_fill(~legal_mask, float("-inf"))
        return int(masked_values.argmax().item())
        