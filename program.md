# zimzum

An autoresearcher that optimises itself. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Single NVIDIA GPU, PyTorch/CUDA.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `zimzum/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b zimzum/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `judge.py` — independent evaluator. Do not modify. This is the sole source of truth for `val_bpb`.
   - `db.py` — experiment database. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize experiments.db**: Import any existing `results.tsv` with `uv run db.py import-tsv`. Then run the baseline (see below).
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`, `judge.py`, or `db.py`. These are read-only infrastructure.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric, and `judge.py` is the sole authority that calls it.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Training and evaluation are two separate phases:

**Phase 1 — Training** (`uv run train.py`): Trains the model and saves a checkpoint (`checkpoint.pt` + `checkpoint_config.json`). Writes `metrics.json` with training stats but `val_bpb` is null.

**Phase 2 — Judging** (`uv run judge.py`): Loads the checkpoint, runs `evaluate_bpb` from `prepare.py`, and writes the final `val_bpb` into `metrics.json`.

After both phases, `metrics.json` contains:

```json
{
  "val_bpb": 0.997900,
  "training_seconds": 300.1,
  "total_seconds": 325.9,
  "peak_vram_mb": 45060.2,
  "mfu_percent": 39.80,
  "total_tokens_M": 499.6,
  "num_steps": 953,
  "num_params_M": 50.3,
  "depth": 8,
  "status": "ok",
  "judge_eval_seconds": 12.3
}
```

Read results with: `cat metrics.json`

## Logging results

When an experiment is done, log it to the experiment database:

```bash
uv run db.py record --hypothesis "description of what you tried" --category architecture --outcome keep
```

Categories: `architecture`, `optimizer`, `hyperparams`, `regularization`, `other`.
Outcomes: `keep`, `discard`, `crash`.

The `record` command automatically reads `metrics.json`, captures the git patch, and stores everything. To review experiment history:

```bash
uv run db.py show                           # all experiments
uv run db.py show --outcome keep --sort val_bpb  # winners sorted by score
uv run db.py show --last 10                 # most recent 10
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `zimzum/mar5`).

**Important**: `experiments.db` is gitignored and never committed. It lives outside git's blast radius so that `git reset --hard` cannot wipe experiment history. All evidence is preserved — winners, losers, and crashes.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code. **Only edit train.py. Do not touch prepare.py, judge.py, or db.py.**
3. `git add train.py && git commit -m "experiment: <description>"`
4. Clean stale artifacts: `rm -f checkpoint.pt checkpoint_config.json metrics.json`
5. Run training: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Check for crash: if `metrics.json` is missing or its `status` is not `"ok"`, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Run the judge: `uv run judge.py >> run.log 2>&1` — the judge verifies only `train.py` was modified, then evaluates the checkpoint. If it reports a surface violation, treat the run as a crash.
8. Read results: `python3 -c "import json; m=json.load(open('metrics.json')); print(f'val_bpb: {m[\"val_bpb\"]:.6f} | peak_vram_mb: {m[\"peak_vram_mb\"]:.1f}')"`
9. **Decide keep or discard**: An improvement counts only if `val_bpb` dropped by more than **0.001** (the noise threshold — adjust after running `measure_noise.py`). Improvements smaller than this are indistinguishable from hardware noise.
10. Record the result: `uv run db.py record --hypothesis "<description>" --category <cat> --outcome <keep|discard|crash>`
11. If val_bpb improved beyond threshold: the commit stays — the branch has advanced.
12. If val_bpb did not improve enough, or got worse: `git reset --hard HEAD~1` to discard the commit cleanly. The experiment is still preserved in `experiments.db`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~6 minutes total (5 min training + ~1 min checkpoint/judge on GPU). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the outcome, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~6 minutes then you can run approx 10/hour, for a total of about 80 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
