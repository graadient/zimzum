# zimzum — outer loop

You are optimizing the research policy, not the model. The inner agent optimizes `train.py`. You optimize the instructions the inner agent follows.

## What you edit

**`autoresearch/program.md`** — the inner agent's research strategy. This is your `train.py` equivalent. Everything in it is fair game: how the agent picks experiments, when to explore vs exploit, how to handle near-misses, rerun rules, category priorities, simplicity weights, context format.

## What you don't edit

- `autoresearch/train.py` — the inner agent's sandbox (it edits this, not you)
- `autoresearch/prepare.py` — data, tokenizer, evaluation (immutable)
- `judge.py` — checkpoint scorer (immutable)
- `db.py` — experiment database (immutable)

## The metric

Your metric is **improvement per GPU-hour**: how much did val_bpb drop during an episode, divided by the wall-clock time?

A policy that makes the inner agent more timid (high keep rate, low improvement) scores badly. A policy that makes the inner agent crash constantly also scores badly. The best policy extracts the most val_bpb improvement in the least time.

## The loop

LOOP FOREVER:

1. Read the current `autoresearch/program.md` and recent episode results.
2. Propose a change to `autoresearch/program.md` — a different research strategy.
3. `git add autoresearch/program.md && git commit -m "policy: <description>"`
4. Run an episode: `python episode.py --repo autoresearch/ --tag <tag> --budget 2h`
5. Let the inner agent run for the budget. (It reads program.md and experiments autonomously.)
6. Score the episode: `python episode.py --repo autoresearch/ --tag <tag> --score`
7. Compare to baseline episodes.
8. `python db.py record --hypothesis "<policy description>" --category policy --outcome <keep|discard>`
9. If improvement per GPU-hour beats baseline: keep the policy change.
10. If not: `git reset --hard HEAD~1` to revert program.md.

## What to try

- Category ordering: should the inner agent try architecture first, or hyperparams?
- Plateau detection: "after 5 consecutive failures in one category, switch"
- Near-miss combining: "if two experiments each almost won, try them together"
- Rerun confirmation: "rerun winners once to confirm they're real before keeping"
- Context format: give the inner agent category-level win rates vs raw history
- Exploration schedules: "explore aggressively for the first hour, then exploit"
- Simplicity weighting: how much to penalize complexity in keep/discard decisions
- Failure analysis: "after a crash, try a smaller version of the same idea"

## Invariants

These hold no matter what you do:
- The judge (`judge.py`) is immutable. Neither you nor the inner agent can change how val_bpb is computed.
- `prepare.py` is immutable. The data, tokenizer, and eval function are fixed.
- Each episode starts from a clean baseline. You can't accumulate hidden advantages.
- `experiments.db` survives git reset. All evidence is preserved.

**NEVER STOP**: Work indefinitely until manually stopped.
