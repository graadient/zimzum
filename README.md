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
  program.md          ← policy + instructions (outer loop edits policy, inner loop follows instructions)
  judge.py            ← honest checkpoint scoring (targets never touch candidate code)
  db.py               ← experiments + episodes in SQLite (survives git reset)
  noise.py            ← noise floor calibration

  autoresearch/       ← karpathy/autoresearch (forked, minimal modifications)
    train.py          ← inner agent edits this (modified: checkpoint saving + __main__ guard)
    prepare.py        ← untouched
    program.md        ← karpathy's original (reference only)
    pyproject.toml    ← untouched
```

Our code: `program.md`, `judge.py`, `db.py`, `noise.py` (4 files at root).
Karpathy's code: everything in `autoresearch/` (train.py modified minimally).

## Quick start

```bash
cd autoresearch && uv sync && uv run prepare.py && cd ..

# Inner loop: one experiment
cd autoresearch
uv run train.py > run.log 2>&1
python ../judge.py
cat metrics.json
cd ..
python db.py record --hypothesis "baseline" --category other --outcome keep
```

Point an inner agent at `program.md` and it follows the inner loop instructions.
Point an outer agent at `program.md` and it edits the policy section.

## Methodology

- Inner loop: n=1 per experiment (Karpathy's default, fast, self-correcting)
- Outer loop: n=2 per policy change (both episodes must beat baseline)
- One policy knob changed per outer iteration (credit assignment)
- All experiments preserved in `experiments.db` (winners, losers, crashes)
- History checked before each experiment (avoid retrying failures)

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch

## License

MIT
