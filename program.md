# zimzum

policy:
  category_order: [architecture, optimizer, hyperparams, regularization, schedule]
  switch_after_failures: 5
  rerun_winner: false
  combine_near_misses: false
  simplicity_bias: 0.3
  noise_threshold: 0.001
  history_mode: summary

## Roles

This file serves two purposes depending on who reads it:

**Inner agent**: follow the experiment loop below. Edit only `autoresearch/train.py`.
**Outer agent**: edit the policy section above. Run two episodes per policy change. Keep only if both episodes improve.

## Setup

1. `cd autoresearch && uv sync && uv run prepare.py && cd ..`
2. Run baseline: `cd autoresearch && uv run train.py > run.log 2>&1 && uv run python ../judge.py && cd ..`
3. Record: `python db.py record --hypothesis "baseline" --category other --outcome keep`

## Inner loop (edit train.py)

**Edit only `autoresearch/train.py`.** Everything else is read-only.

**Goal**: lowest `val_bpb`. Fixed 5-minute training budget.

**Before proposing**: check `python db.py show --last 20` to avoid retrying failed approaches.

LOOP FOREVER:

1. Check history: `python db.py show --last 20`
2. Edit `autoresearch/train.py` with an idea.
3. `cd autoresearch && git add train.py && git commit -m "experiment: <desc>" && cd ..`
4. `cd autoresearch && rm -f checkpoint.pt checkpoint_config.json metrics.json && uv run train.py > run.log 2>&1 && cd ..`
5. If `autoresearch/metrics.json` missing: crashed. Check `tail -n 50 autoresearch/run.log`.
6. `cd autoresearch && uv run python ../judge.py >> run.log 2>&1 && cd ..`
7. `cat autoresearch/metrics.json` — check val_bpb.
8. Improvement must exceed noise_threshold to count.
9. `python db.py record --hypothesis "<desc>" --category <cat> --outcome <keep|discard|crash>`
10. If improved: commit stays.
11. If not: `cd autoresearch && git reset --hard HEAD~1 && cd ..`

**Simplicity criterion**: simpler is better. Small gain + ugly complexity = not worth it.

**NEVER STOP**: work indefinitely until manually stopped.

## Outer loop (edit this file's policy section)

**Edit only the policy section** at the top of this file. Change one knob per iteration.

1. Propose a policy change (one knob).
2. Run two episodes with the new policy. Each episode = inner agent running for a fixed budget.
3. Run two episodes with the current baseline policy.
4. Score: `python db.py score-episode --tag <tag>`
5. Compare. Both candidate episodes must beat both baseline episodes.
6. `python db.py record --hypothesis "<policy desc>" --category policy --outcome <keep|discard>`
7. If both episodes beat baseline: keep the policy change.
8. If not: revert the policy section.

The outer metric is `score = baseline_bpb - best_bpb` per episode. Higher is better.
