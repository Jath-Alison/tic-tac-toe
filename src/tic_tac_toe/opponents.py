"""Baseline move selection, kept separate from the game rules."""

import numpy as np

from tic_tac_toe.game import Turn


def random_action(turn: Turn, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(turn.action_mask)
    if turn.done or not len(legal):
        raise ValueError("No move is available on this turn.")
    return int(rng.choice(legal))
