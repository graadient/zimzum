"""
Generate program.md from template + project.yaml.
Usage: uv run -m harness.protocol --project projects/gpt_pretrain/project.yaml
"""

import argparse
import os

from .config import load_project

TEMPLATE = """# zimzum

An autoresearcher that optimises itself.

## Setup

1. **Agree on a run tag** (e.g. `mar5`). Branch `zimzum/<tag>` must not exist.
2. **Create the branch**: `git checkout -b zimzum/<tag>`
3. **Read the in-scope files**:
   - `{mutable_files}` — the file you modify (candidate sandbox)
   - `projects/{project_name}/model.py` — model definition (immutable)
   - `projects/{project_name}/prepare.py` — data, tokenizer, dataloader (immutable)
   - `projects/{project_name}/evaluate.py` — evaluation logic (immutable)
   - `projects/{project_name}/project.yaml` — project contract (immutable)
4. **Verify data exists**: `~/.cache/autoresearch/` should have data shards and tokenizer.
5. **Initialize experiments.db**: `uv run -m harness.db --project {project_yaml} import-tsv` if legacy data exists.
6. **Confirm and go.**

## Experimentation

Training runs for a **fixed time budget of {time_budget} seconds**.

**What you CAN do:**
- Modify `{mutable_files}` — everything is fair game: architecture, optimizer, hyperparams, training loop.

**What you CANNOT do:**
- Modify any other file. `model.py`, `prepare.py`, `evaluate.py`, `project.yaml`, and the harness are all read-only.
- Install new packages or add dependencies.

**The goal**: get the lowest `{primary_metric}`.

**Simplicity criterion**: simpler is better. A small improvement that adds ugly complexity is not worth it.

## Output format

Training and evaluation are two phases:

**Phase 1 — Training** (`{train_cmd}`): Trains, saves checkpoint. Writes `metrics.json` with `{primary_metric}: null`.

**Phase 2 — Judging** (`uv run -m harness.judge --project {project_yaml}`): Loads checkpoint, evaluates, writes final `{primary_metric}` to `metrics.json`.

Read results: `cat metrics.json`

## Logging

```bash
uv run -m harness.db --project {project_yaml} record --hypothesis "description" --category architecture --outcome keep
uv run -m harness.db --project {project_yaml} show
uv run -m harness.db --project {project_yaml} show --outcome keep --sort primary_metric
```

## The experiment loop

**Important**: `experiments.db` is gitignored — `git reset --hard` cannot wipe it.

LOOP FOREVER:

1. Look at git state.
2. Edit `{mutable_files}`. **Only this file.**
3. `git add {mutable_files} && git commit -m "experiment: <description>"`
4. `rm -f {checkpoint} {config} {metrics}`
5. `{train_cmd} > run.log 2>&1`
6. If `{metrics}` missing or `status != "ok"`: crashed. Check `tail -n 50 run.log`.
7. `uv run -m harness.judge --project {project_yaml} >> run.log 2>&1`
8. `cat {metrics}`
9. Improvement must exceed **{noise_threshold}** to count (noise threshold).
10. `uv run -m harness.db --project {project_yaml} record --hypothesis "<desc>" --category <cat> --outcome <keep|discard|crash>`
11. If improved: commit stays.
12. If not: `git reset --hard HEAD~1`. Experiment preserved in `experiments.db`.

**Timeout**: ~{timeout_minutes} minutes. Kill and discard if exceeded.

**NEVER STOP**: The human expects you to work indefinitely until manually stopped.
"""


def generate(cfg, project_yaml_path):
    mutable = cfg["mutable_files"][0]
    timeout = cfg["time_budget"] // 60 + 2
    text = TEMPLATE.format(
        project_name=cfg["project_name"],
        mutable_files=mutable,
        project_yaml=project_yaml_path,
        train_cmd=cfg["train_cmd"],
        primary_metric=cfg["primary_metric"],
        time_budget=cfg["time_budget"],
        noise_threshold=cfg.get("noise_threshold", 0.001),
        checkpoint=cfg["checkpoint"],
        config=cfg["config"],
        metrics=cfg["metrics"],
        timeout_minutes=timeout,
    )
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate program.md from project.yaml")
    parser.add_argument("--project", default=None, help="Path to project.yaml")
    parser.add_argument("--output", default="program.md")
    args = parser.parse_args()

    yaml_path = args.project or os.environ.get("ZIMZUM_PROJECT", "projects/gpt_pretrain/project.yaml")
    cfg = load_project(yaml_path)
    text = generate(cfg, yaml_path)

    with open(args.output, "w") as f:
        f.write(text)
    print(f"Generated {args.output} from {yaml_path}")


if __name__ == "__main__":
    main()
