"""
Generic experiment database for zimzum.
Stores every experiment with full metadata. Project-specific metrics
go in primary_metric (float) and extra_metrics (JSON blob).

Usage:
    uv run -m harness.db --project projects/gpt_pretrain/project.yaml record --hypothesis "try SiLU" --outcome keep
    uv run -m harness.db --project projects/gpt_pretrain/project.yaml show
"""

import argparse
import json
import sqlite3
import subprocess
import time

from .config import load_project

DB_PATH = "experiments.db"
OUTCOMES = ["keep", "discard", "crash"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_tag             TEXT,
    parent_commit       TEXT,
    child_commit        TEXT,
    hypothesis          TEXT,
    category            TEXT,
    patch               TEXT,
    started_at          REAL,
    finished_at         REAL,
    exit_status         INTEGER,
    primary_metric      REAL,
    primary_metric_name TEXT,
    training_seconds    REAL,
    total_seconds       REAL,
    peak_vram_mb        REAL,
    total_tokens_M      REAL,
    num_steps           INTEGER,
    num_params_M        REAL,
    extra_metrics       TEXT,
    outcome             TEXT,
    notes               TEXT
)
"""

GENERIC_SORTABLE = {
    "id", "run_tag", "primary_metric", "training_seconds", "total_seconds",
    "peak_vram_mb", "num_steps", "num_params_M", "outcome",
    "child_commit", "finished_at",
}


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _load_metrics(path="metrics.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def record_experiment(conn, cfg, hypothesis, category, outcome, notes=None):
    metrics_path = cfg.get("metrics", "metrics.json")
    metrics = _load_metrics(metrics_path)
    now = time.time()

    metric_name = cfg["primary_metric"]
    extra_cfg = cfg.get("extra_metrics", {})
    extra = {display: metrics.get(key) for display, key in extra_cfg.items() if metrics.get(key) is not None}

    row = {
        "run_tag": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "parent_commit": _git("rev-parse", "--short", "HEAD~1"),
        "child_commit": _git("rev-parse", "--short", "HEAD"),
        "hypothesis": hypothesis,
        "category": category,
        "patch": _git("diff", "HEAD~1", "HEAD"),
        "started_at": metrics.get("total_seconds", None) and (now - metrics["total_seconds"]),
        "finished_at": now,
        "exit_status": 0 if outcome != "crash" else 1,
        "primary_metric": metrics.get(metric_name),
        "primary_metric_name": metric_name,
        "training_seconds": metrics.get("training_seconds"),
        "total_seconds": metrics.get("total_seconds"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "total_tokens_M": metrics.get("total_tokens_M"),
        "num_steps": metrics.get("num_steps"),
        "num_params_M": metrics.get("num_params_M"),
        "extra_metrics": json.dumps(extra) if extra else None,
        "outcome": outcome,
        "notes": notes,
    }

    cols = list(row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    conn.execute(
        f"INSERT INTO experiments ({col_names}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return row


def query_experiments(conn, outcome=None, sort="id", last=None):
    if sort not in GENERIC_SORTABLE:
        raise ValueError(f"Invalid sort column: {sort!r} (allowed: {GENERIC_SORTABLE})")
    query = "SELECT * FROM experiments"
    params = []
    if outcome:
        query += " WHERE outcome = ?"
        params.append(outcome)
    query += f" ORDER BY {sort}"
    if last:
        query += " DESC LIMIT ?"
        params.append(last)
    cursor = conn.execute(query, params)
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _format_table(rows, cfg):
    if not rows:
        print("No experiments found.")
        return
    metric_name = cfg["primary_metric"]
    display = cfg.get("display_columns", ["id", "outcome", metric_name, "peak_vram_mb", "child_commit", "hypothesis"])

    # Map display names to row keys
    def get_val(row, col):
        if col == metric_name:
            return row.get("primary_metric")
        if col in row:
            return row[col]
        extra = json.loads(row.get("extra_metrics") or "{}") if row.get("extra_metrics") else {}
        return extra.get(col)

    widths = {}
    for c in display:
        vals = [str(get_val(r, c) or "") for r in rows]
        widths[c] = max(len(c), max((len(v) for v in vals), default=0))

    print(" | ".join(c.ljust(widths[c]) for c in display))
    print("-+-".join("-" * widths[c] for c in display))
    for r in rows:
        vals = []
        for c in display:
            v = get_val(r, c)
            if isinstance(v, float):
                v = f"{v:.6f}" if c == metric_name else f"{v:.1f}"
            vals.append(str(v if v is not None else "").ljust(widths[c]))
        print(" | ".join(vals))


def export_tsv(conn, cfg, output_path):
    rows = query_experiments(conn, sort="id")
    metric_name = cfg["primary_metric"]
    with open(output_path, "w") as f:
        f.write(f"commit\t{metric_name}\tmemory_gb\tstatus\tdescription\n")
        for r in rows:
            commit = r["child_commit"] or "unknown"
            metric = f"{r['primary_metric']:.6f}" if r["primary_metric"] is not None else "0.000000"
            mem_gb = f"{r['peak_vram_mb'] / 1024:.1f}" if r["peak_vram_mb"] else "0.0"
            status = r["outcome"] or "unknown"
            desc = r["hypothesis"] or ""
            f.write(f"{commit}\t{metric}\t{mem_gb}\t{status}\t{desc}\n")
    print(f"Exported {len(rows)} experiments to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="zimzum experiment database")
    parser.add_argument("--project", default=None, help="Path to project.yaml")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="Record an experiment")
    rec.add_argument("--hypothesis", required=True)
    rec.add_argument("--category", default="unknown")
    rec.add_argument("--outcome", required=True, choices=OUTCOMES)
    rec.add_argument("--notes")

    show = sub.add_parser("show", help="Show experiments")
    show.add_argument("--outcome", choices=OUTCOMES)
    show.add_argument("--sort", default="id")
    show.add_argument("--last", type=int)

    exp = sub.add_parser("export", help="Export to TSV")
    exp.add_argument("--output", default="results_export.tsv")

    args = parser.parse_args()
    cfg = load_project(args.project)
    conn = init_db()

    if args.command == "record":
        row = record_experiment(conn, cfg, args.hypothesis, args.category, args.outcome, notes=args.notes)
        metric_name = cfg["primary_metric"]
        print(f"Recorded: {row['outcome']} | {metric_name}={row['primary_metric']} | {row['hypothesis']}")
    elif args.command == "show":
        rows = query_experiments(conn, outcome=args.outcome, sort=args.sort, last=args.last)
        _format_table(rows, cfg)
    elif args.command == "export":
        export_tsv(conn, cfg, args.output)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
