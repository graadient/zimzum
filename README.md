# zimzum

An autoresearcher that optimises itself.

zimzum is a harness that wraps [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with an honest judge, experiment evidence, and noise measurement. The objective is to optimize the research policy — how the agent picks experiments — and then apply that optimized policy to any autoresearch-compatible project.

## What's in this repo

```
judge.py       ← scores checkpoints honestly (targets never touch candidate code)
db.py          ← stores every experiment (winners, losers, crashes) in SQLite
noise.py       ← measures hardware noise floor and minimum detectable effect
program.md     ← the research policy (the thing we optimize)
```

## How to use it

1. Clone [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
2. Drop these files into it
3. Modify `train.py` to save a checkpoint + write `metrics.json` (instead of calling `evaluate_bpb` inline)
4. Point an AI agent at `program.md`

The agent runs the loop: edit `train.py` → train → judge → keep/revert → repeat.

## Trust boundary

- `train.py` saves a checkpoint but never reports its own `val_bpb`
- `judge.py` calls `model(x)` for logits only — eval targets never touch candidate code
- `judge.py` verifies only `train.py` was modified (surface check, fail-closed)
- `experiments.db` is gitignored — `git reset --hard` cannot wipe evidence
- Improvements must exceed a noise threshold to be promoted

## The objective

The repo's job is **not** to train a better GPT. Karpathy's autoresearch already does that.

The repo's job is to **optimize how the research is done** — the policy in `program.md` — and measure whether policy changes lead to better research outcomes. The inner loop (agent edits train.py) is karpathy's. The outer loop (optimizing the research strategy) is ours.

## License

MIT
