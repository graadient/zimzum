# zimzum

An autoresearcher that optimises itself.

The harness (judge, experiment DB, noise measurement, protocol generator) is generic and reusable. The project (model, data, evaluation) is pluggable via `project.yaml`. The harness is the thing the outer loop will eventually optimize. The project is what the inner loop optimizes.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). For Apple Silicon (MLX), see [zimzum-mlx](https://github.com/graadient/zimzum-mlx).

## Structure

```
harness/                          # generic, reusable, no project knowledge
  judge.py                        # loads checkpoint, calls project eval, writes metric
  db.py                           # SQLite experiment database
  noise.py                        # noise floor measurement
  surface.py                      # edit surface verification
  protocol.py                     # generates program.md from template + project.yaml
  config.py                       # loads project.yaml

projects/gpt_pretrain/            # one specific research project
  project.yaml                    # the contract between harness and project
  model.py                        # model classes (immutable, judge imports from here)
  train.py                        # candidate sandbox (agent edits this)
  prepare.py                      # data, tokenizer, dataloader (immutable)
  evaluate.py                     # evaluation logic (immutable)
```

## Quick start

**Requirements:** NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare.py                                                    # one-time data prep
uv run -m harness.protocol --project projects/gpt_pretrain/project.yaml  # generate program.md
uv run -m projects.gpt_pretrain.train                                # train + save checkpoint
uv run -m harness.judge --project projects/gpt_pretrain/project.yaml # independent eval
cat metrics.json
```

Then point an AI agent at `program.md` and let it run the loop.

## Trust boundary

- The candidate (`train.py`) trains and saves a checkpoint but never reports its own metric.
- The judge calls `model(x)` for logits only — **eval targets are never passed to candidate code**.
- The judge imports the model from `model.py` (immutable), not from `train.py` (candidate).
- The judge verifies only `train.py` was modified (surface check). Forbidden edits are rejected.
- `experiments.db` is gitignored — `git reset --hard` cannot wipe experiment history.

## Adding a new project

Create `projects/your_project/` with:
- `project.yaml` — configure metric name, commands, model module, eval module, mutable files
- `model.py` — model definition
- `train.py` — candidate sandbox
- `prepare.py` — data pipeline
- `evaluate.py` — implements `evaluate(model, cfg) -> float`

The harness works unchanged.

## License

MIT
