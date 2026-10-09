"""A small adapter around PettingZoo's turn-based tic-tac-toe game."""

from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import tictactoe_v3


@dataclass(frozen=True)
class Turn:
    player: str
    board: np.ndarray  # (3, 3, 2): mine, opponent
    action_mask: np.ndarray  # nine entries: 1 means legal
    reward: float
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def features(self) -> np.ndarray:
        """Return the 18 float inputs for a future neural network."""
        return self.board.astype(np.float32).reshape(18)


class Game:
    def __init__(self, seed: int | None = None):
        self.env = tictactoe_v3.env()
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.env.reset(seed=seed)

    @property
    def active(self) -> bool:
        # Terminal agents still need a final step(None) to leave the cycle.
        return bool(self.env.agents)

    def turn(self) -> Turn:
        if not self.active:
            raise RuntimeError("Game is finished; call reset().")
        observation, reward, terminated, truncated, _ = self.env.last()
        
        if observation is None:
            # Return empty observation turn?
            return Turn(
                self.env.agent_selection,
                np.ndarray(shape=(3,3,2)),
                np.ndarray(shape=(9)),
                float(reward),
                terminated,
                truncated,
            )
            
        return Turn(
            self.env.agent_selection,
            observation["observation"].copy(),
            observation["action_mask"].copy(),
            float(reward),
            terminated,
            truncated,
        )

    def step(self, action: int | None) -> None:
        turn = self.turn()
        if turn.done:
            if action is not None:
                raise ValueError("A finished player's action must be None.")
        elif (
            not isinstance(action, (int, np.integer))
            or not 0 <= action < 9
            or not turn.action_mask[action]
        ):
            raise ValueError("Choose a legal square numbered 0 through 8.")
        self.env.step(action)

    def close(self) -> None:
        self.env.close()
