# Tic-tac-toe: first implementation milestone

Python 3.12+, uv, and PettingZoo. This increment implements the environment
adapter, random opponent, terminal runner, and integration tests. Neural
network and training modules will be added in the next increments.

## Run

From this directory:

```bash
uv sync
uv run python -m tic_tac_toe.play --verbose
uv run python -m tic_tac_toe.play --games 1000 --seed 42
uv run pytest
uv run ruff check .
```

The runner currently plays random versus random; it does not accept human input.
After a successful sync, commit the generated uv.lock to record exact versions.

## Read the modules in this order

1. `game.py`: `Game.turn()` gives the current player, observation, mask,
   reward, and end flags. `Game.step(action)` applies a legal move.
2. `opponents.py`: sample uniformly from legal squares.
3. `play.py`: alternate players, collect final rewards, report results.
4. `tests/test_game.py`: ownership encoding, legality, outcomes, reset,
   and completion of 100 randomly played games.

The board observation has shape `(3, 3, 2)`. Channel zero holds the observing
player's pieces and channel one holds the opponent's pieces. `Turn.features()`
flattens this to 18 float inputs in cell-major order: mine, opponent for cell
zero, then mine, opponent for cell one, and so on. Actions 0–8 run left to right
through each row, starting at the top-left square.

A terminal player must be stepped with `None`. PettingZoo exposes each player's
final reward through `last()` on that player's final turn. The runner collects
both rewards before completing the episode. Wins yield +1, losses -1, draws 0.
The wrapper rejects illegal moves before forwarding them to PettingZoo.

## Next increment

Add `model.py`: an 18 → 64 → 64 → 9 PyTorch network with ReLU hidden layers.
Mask illegal outputs before greedy selection; exploration also samples only
legal moves. Add torch with `uv add torch` when implementing that module.
Then add training against a random opponent and a separate evaluation loop.

## Validation status

Python syntax was checked during creation. Runtime tests could not run in the
creation environment because package-index access returned HTTP 403. Run the
commands above locally to validate the PettingZoo integration.
