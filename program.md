# zimzum

policy:
  category_order: [architecture, optimizer, hyperparams, regularization, schedule]
  switch_after_failures: 5
  rerun_winner: false
  combine_near_misses: false
  simplicity_bias: 0.3
  noise_threshold: 0.001
  history_mode: summary

## Policy interpretation

Apply these knobs as follows:

- **category_order**: when choosing what to try next, prefer categories earlier in the list. Cycle through them.
- **switch_after_failures**: after this many consecutive discards in one category, move to the next category in the list.
- **rerun_winner**: if true and a win margin is within 2x noise_threshold, rerun once to confirm before keeping.
- **combine_near_misses**: if true and you've had repeated failures, try merging the best two recent discards into one experiment.
- **simplicity_bias**: when a win is small (<noise_threshold * 3), prefer the simpler code. If the diff is large and the win is tiny, discard.
- **noise_threshold**: minimum val_bpb improvement to count as a real win.
- **history_mode**: "summary" = check `python db.py show --last 20` before each experiment. "full" = read all history.

## Roles

**Inner agent**: follow the experiment loop below. Edit only `autoresearch/train.py`.
**Outer agent**: edit only the policy section above. One knob per iteration.

## Setup

1. `cd autoresearch && uv sync && uv run prepare.py && cd ..`
2. Run baseline: `cd autoresearch && uv run train.py > run.log 2>&1 && uv run python ../judge.py && cd ..`
3. Record: `RUN_TAG=baseline python db.py record --hypothesis "baseline" --category other --outcome keep`

## Inner loop (edit train.py)

**Edit only `autoresearch/train.py`.** Everything else is read-only.

**Goal**: lowest `val_bpb`. Fixed 5-minute training budget.

**Before proposing**: check history to avoid retrying failed approaches.

LOOP FOREVER:

1. `python db.py show --last 20` — check what's been tried.
2. Edit `autoresearch/train.py` with an idea. Follow category_order and switch_after_failures.
3. `cd autoresearch && git add train.py && git commit -m "experiment: <desc>" && cd ..`
4. `cd autoresearch && rm -f checkpoint.pt checkpoint_config.json metrics.json && uv run train.py > run.log 2>&1 && cd ..`
5. If `autoresearch/metrics.json` missing: crashed. Check `tail -n 50 autoresearch/run.log`.
6. `cd autoresearch && uv run python ../judge.py >> run.log 2>&1 && cd ..`
7. `cat autoresearch/metrics.json` — check val_bpb.
8. If improvement > noise_threshold: keep. If rerun_winner is true and margin < 2x noise_threshold: rerun once first.
9. `python db.py record --hypothesis "<desc>" --category <cat> --outcome <keep|discard|crash>`
10. If improved: commit stays.
11. If not: `cd autoresearch && git reset --hard HEAD~1 && cd ..`

**Simplicity**: if simplicity_bias > 0 and the win is tiny, prefer the simpler code.

**NEVER STOP**: work indefinitely until manually stopped.

## Outer loop (edit the policy section)

**Edit only the policy block** at the top of this file. Change one knob per iteration.

1. Read recent episode scores: check `experiments.db` episodes table.
2. Propose a policy change — one knob at a time.
3. Commit the updated program.md.
4. **Run 1 candidate episode** with the new policy. Tag it (e.g. `RUN_TAG=ep-cand-003`).
5. **Run 1 baseline episode** with the previous policy (revert, run, re-apply).
6. Score both: `python db.py score-episode --tag ep-cand-003` and `python db.py score-episode --tag ep-base-003`.
7. If candidate wins clearly (score > baseline score + noise_threshold): **run 1 confirmation** of each.
8. If candidate loses or margin is ambiguous: discard immediately.
9. Keep only if candidate wins in both rounds.

Outer metric: `score = baseline_bpb - best_bpb` per episode. Higher is better.

## Invariants

- `judge.py` is immutable. Neither loop changes how val_bpb is computed.
- `prepare.py` is immutable. Data, tokenizer, eval function are fixed.
- `experiments.db` is at repo root, outside autoresearch/ — inner git resets don't touch it.
- All experiments preserved: winners, losers, crashes, with full patches.
