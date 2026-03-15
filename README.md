# zimzum

An autoresearcher that optimises itself.

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) uses an LLM to optimize a training script. zimzum uses an LLM to optimize the research policy that controls that LLM. Same loop, one level up.

```
outer loop: edits program.md policy → measures improvement per episode
  inner loop: edits train.py       → measures val_bpb
```

## Structure

```
zimzum/
  program.md          ← policy + instructions (outer edits policy, inner follows instructions)
  judge.py            ← honest scoring (targets never touch candidate code)
  db.py               ← experiments + episodes in SQLite (survives git reset)
  noise.py            ← noise floor calibration

  autoresearch/       ← karpathy/autoresearch (train.py minimally modified)
    train.py          ← inner agent edits this
    prepare.py        ← untouched
```

## Quick start

```bash
cd autoresearch && uv sync && uv run prepare.py && cd ..

# Run one experiment
cd autoresearch && uv run train.py > run.log 2>&1 && uv run python ../judge.py && cd ..
cat autoresearch/metrics.json

# Record it
RUN_TAG=baseline python db.py record --hypothesis "baseline" --category other --outcome keep
```

## Methodology

- Inner loop: n=1 per experiment (Karpathy's default — fast, self-correcting)
- Outer loop: 1v1 + confirmation (candidate vs baseline, confirm if close)
- One policy knob changed per outer iteration (credit assignment)
- All experiments in `experiments.db` with full patches (winners, losers, crashes)
- `RUN_TAG` env var or `--run-tag` flag for explicit episode tagging
- Episodes tracked with policy_hash, base_commit, and computed score

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch

## License

MIT
