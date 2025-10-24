```markdown
# Minimal A3C (Asynchronous Advantage Actor-Critic) Example

This repository contains a minimal, educational A3C implementation in a single file (`a3c.py`) that trains an Actor-Critic model on `CartPole-v1`.

Note: This is a small, didactic example to demonstrate the A3C workflow (multiple processes, n-step returns, local-to-global gradient updates). It is not optimized for production use.

## Requirements

- Python 3.7+
- torch
- gym

Install requirements:
```
pip install torch gym
```

## Usage

Run training with 4 workers (example):
```
python a3c.py --workers 4 --max-episodes 500
```

Key CLI options:
- `--env-name` (default `CartPole-v1`)
- `--workers` (number of parallel worker processes)
- `--max-episodes` (global total episodes to stop training)
- `--t-max` (n-step length)
- `--gamma` (discount factor)
- `--lr` (learning rate)
- `--hidden-size` (hidden layer size)
- `--entropy-coef` (entropy bonus coefficient)
- `--value-loss-coef` (coefficient for value loss)

## Notes and compatibility

- Gym API changed around v0.26: `reset()` may return `(obs, info)` and `step()` may return `(obs, reward, terminated, truncated, info)`. The script includes basic compatibility handling.
- On Windows, multiprocessing uses the `spawn` start method; the script sets it explicitly.
- The optimizer state is placed in shared memory to allow parallel updates. This implementation uses a shared Adam optimizer and copies local gradients into the global model before stepping the shared optimizer.

If you want, I can:
- Push these files to your repository as a new branch and create a pull request.
- Split the script into modules (e.g., `model.py`, `shared_adam.py`, `worker.py`, `train.py`).
- Add logging, TensorBoard support, or an evaluation script to test the trained policy.
```
