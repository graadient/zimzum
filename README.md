# zimzum

An autoresearcher that optimises itself.

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) uses an LLM to optimize a training script. zimzum uses an LLM to optimize the research policy that controls that LLM. Same loop structure, one level up.

```
outer loop (zimzum):   edits program.md    → measures improvement per GPU-hour
  inner loop (karpathy): edits train.py    → measures val_bpb
```

## What's in this repo

```
program.md     ← outer loop instructions (how to optimize the research policy)
episode.py     ← runs one scored inner-loop session
judge.py       ← scores checkpoints honestly (targets never touch candidate code)
db.py          ← stores every experiment in SQLite (survives git reset)
noise.py       ← measures noise floor and minimum detectable effect
```

## How the two loops interact

The outer agent's mutable surface is `autoresearch/program.md` — the inner agent's research strategy. The inner agent's mutable surface is `autoresearch/train.py` — the model code. Neither can touch the other's file. Neither can touch the judge or eval harness.

```
IMMUTABLE (neither loop touches):
  prepare.py       data, tokenizer, evaluation
  judge.py         honest checkpoint scoring

OUTER LOOP edits:
  program.md       the inner agent's research strategy

INNER LOOP edits:
  train.py         the model, optimizer, training code
```

Each episode starts from a clean baseline. The outer loop's edit to `program.md` is the only variable. This is a controlled experiment.

## Quick start

```bash
# Clone the target project
git clone https://github.com/karpathy/autoresearch
cd autoresearch && uv sync && uv run prepare.py && cd ..

# Clone zimzum
git clone https://github.com/graadient/zimzum

# Copy harness files into autoresearch
cp zimzum/judge.py zimzum/db.py zimzum/noise.py autoresearch/

# Point the outer agent at zimzum/program.md
# Point the inner agent at autoresearch/program.md
```

## The objective

The repo's job is to find the `program.md` that produces the most val_bpb improvement per GPU-hour when an inner agent follows it. That optimized research policy can then be applied to any autoresearch-compatible project.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch (the inner loop)

## License

MIT
