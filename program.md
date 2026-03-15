# zimzum

An autoresearcher that optimises itself.

## Setup

1. **Agree on a run tag** (e.g. `mar15`). Branch `zimzum/<tag>` must not exist.
2. `git checkout -b zimzum/<tag>`
3. **Read the in-scope files**: `README.md`, `prepare.py` (read-only), `train.py` (your sandbox), `judge.py` (read-only), `db.py` (read-only).
4. **Verify data**: `~/.cache/autoresearch/` should have shards and tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Confirm and go.**

## Rules

**Edit only `train.py`.** Everything else is read-only: `prepare.py`, `judge.py`, `db.py`, `noise.py`.

**Goal**: lowest `val_bpb`. Fixed 5-minute training budget. Everything in train.py is fair game.

**Simplicity criterion**: simpler is better. Small improvement + ugly complexity = not worth it.

## The loop

**Important**: `experiments.db` is gitignored. `git reset --hard` cannot wipe it.

LOOP FOREVER:

1. Edit `train.py` with an idea.
2. `git add train.py && git commit -m "experiment: <description>"`
3. `rm -f checkpoint.pt checkpoint_config.json metrics.json`
4. `uv run train.py > run.log 2>&1`
5. If `metrics.json` missing or `status != "ok"`: crashed. Check `tail -n 50 run.log`.
6. `uv run judge.py >> run.log 2>&1` — the judge verifies only `train.py` was modified, then scores the checkpoint.
7. `cat metrics.json` — read `val_bpb`.
8. Improvement must exceed **0.001** to count (noise threshold — calibrate with `noise.py`).
9. `python db.py record --hypothesis "<desc>" --category <cat> --outcome <keep|discard|crash>`
10. If improved: commit stays.
11. If not: `git reset --hard HEAD~1`. Evidence preserved in `experiments.db`.

**Timeout**: ~6 min per experiment. Kill and discard if >10 min.

**NEVER STOP**: Work indefinitely until manually stopped.
