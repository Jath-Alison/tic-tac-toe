"""Run reproducible random-versus-random games from the terminal."""

import argparse

import numpy as np

from tic_tac_toe.game import Game, Turn
from tic_tac_toe.opponents import random_action


def format_board(turn: Turn) -> str:
    """Display global X/O symbols, although observations are player-relative."""
    mine, theirs = ("X", "O") if turn.player == "player_1" else ("O", "X")
    cells = [
        mine if cell[0] else theirs if cell[1] else str(index)
        for index, cell in enumerate(turn.board.reshape(9, 2))
    ]
    return "\n".join(" | ".join(cells[i:i + 3]) for i in range(0, 9, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    rng = np.random.default_rng(args.seed)
    game = Game(args.seed)
    counts = {"X wins": 0, "O wins": 0, "draws": 0}
    try:
        for episode in range(args.games):
            game.reset(args.seed + episode)
            final_rewards = {}
            while game.active:
                turn = game.turn()
                if turn.done:
                    final_rewards[turn.player] = turn.reward
                    game.step(None)
                else:
                    action = random_action(turn, rng)
                    if args.verbose:
                        print("-" * 40)
                        print(format_board(turn))
                        print(f"{turn.player} selects {action}\n")
                    game.step(action)
                    if args.verbose:
                        print(format_board(game.turn()), "\n")
            result = (
                "X wins" if final_rewards["player_1"] > 0
                else "O wins" if final_rewards["player_2"] > 0
                else "draws"
            )
            counts[result] += 1
        print(f"Games: {args.games}; seed: {args.seed}; results: {counts}")
    finally:
        game.close()


if __name__ == "__main__":
    main()
