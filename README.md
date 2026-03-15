# zimzum

An autoresearcher that optimises itself.

Give an AI agent a training script, a fixed compute budget, and a single metric — then let it run autonomously. It edits the code, trains, evaluates, keeps what works, reverts what doesn't, and repeats. The researcher *is* the experiment.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Single NVIDIA GPU, PyTorch/CUDA. For Apple Silicon (MLX), see [zimzum-mlx](https://github.com/graadient/zimzum-mlx).

## How it works

1. The agent reads `program.md` — the autonomous experiment protocol.
2. It edits `train.py` (architecture, optimizer, hyperparams, anything).
3. Runs a **fixed 5-minute** training experiment.
4. `judge.py` independently evaluates the checkpoint — the candidate never scores itself.
5. **Keeps** improvements, **reverts** failures.
6. Logs every experiment to `experiments.db` — winners, losers, and crashes.
7. Repeats indefinitely until stopped.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# one-time data + tokenizer prep (~2 min)
uv run prepare.py

# run one experiment
uv run train.py          # train + save checkpoint
uv run judge.py          # independent eval → val_bpb in metrics.json
cat metrics.json
```

Then point Claude Code (or another coding agent) at `program.md` and let it run the loop.

## Files

- `program.md` — the autonomous experiment protocol (agent instructions)
- `train.py` — model, optimizer, training loop (the only file the agent edits)
- `prepare.py` — data prep, tokenizer, dataloader, evaluation (fixed, read-only)
- `judge.py` — independent evaluator (fixed, read-only)
- `db.py` — experiment database (fixed, read-only)
- `measure_noise.py` — noise floor measurement

## Trust boundary

The candidate (`train.py`) trains and saves a checkpoint but never reports its own `val_bpb`. The immutable `judge.py` loads the checkpoint and computes the score using `evaluate_bpb` from `prepare.py`. The agent reads structured `metrics.json`, not its own stdout. This prevents the candidate from faking results.

## Noise measurement

Before trusting results, measure your hardware's noise floor:

```bash
uv run measure_noise.py --runs 3
```

This runs the same code 3 times with the same seed and reports the minimum detectable effect size — how large an improvement must be to beat hardware nondeterminism.

## Origin

zimzum is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), with an immutable judge, experiment database, and noise measurement layered on top.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat (the original concept)
- [Apple MLX team](https://github.com/ml-explore/mlx) — MLX variant lives at [zimzum-mlx](https://github.com/graadient/zimzum-mlx)

## License

MIT
