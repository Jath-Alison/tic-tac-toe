import numpy as np

from tic_tac_toe.agent import Agent
from tic_tac_toe.model import QNetwork


def main():
    model = QNetwork()
    agent = Agent(model, seed=42)

    board = np.zeros((3, 3, 2), dtype=np.int8)
    mask = np.ones(9, dtype=np.int8)

    print("Input shape:", agent.encode(board).shape)
    print("Random move:", agent.select_action(board, mask, epsilon=1))
    print("Greedy move:", agent.select_action(board, mask, epsilon=0))

    # Both selection methods must choose square 4.
    mask = np.zeros(9, dtype=np.int8)
    mask[4] = 1

    assert agent.select_action(board, mask, epsilon=1) == 4
    assert agent.select_action(board, mask, epsilon=0) == 4
    print("Legal-move checks passed.")


if __name__ == "__main__":
    main()