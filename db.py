"""
Experiment database for zimzum.
Stores every experiment (winners, losers, crashes) with full metadata.

Usage:
    uv run db.py record --hypothesis "try SiLU" --category architecture --outcome keep
    uv run db.py show
    uv run db.py show --outcome keep --sort val_bpb
    uv run db.py show --last 10
    uv run db.py export --output results_export.tsv
    uv run db.py import-tsv --input results.tsv
"""

import argparse
import json
import os
import sqlite3
import subprocess
import time

DB_PATH = "experiments.db"
OUTCOMES = ["keep", "discard", "crash"]
SORTABLE_COLUMNS = {
    "id", "run_tag", "val_bpb", "training_seconds", "total_seconds",
    "peak_vram_mb", "num_steps", "num_params_M", "depth", "outcome",
    "child_commit", "finished_at",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_tag         TEXT,
    parent_commit   TEXT,
    child_commit    TEXT,
    hypothesis      TEXT,
    category        TEXT,
    patch           TEXT,
    started_at      REAL,
    finished_at     REAL,
    exit_status     INTEGER,
    val_bpb         REAL,
    training_seconds REAL,
    total_seconds   REAL,
    peak_vram_mb    REAL,
    total_tokens_M  REAL,
    num_steps       INTEGER,
    num_params_M    REAL,
    depth           INTEGER,
    outcome         TEXT,
    notes           TEXT
)
"""


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


def record_experiment(conn, hypothesis, category, outcome, run_tag=None,
                      metrics_file="metrics.json", notes=None):
    metrics = _load_metrics(metrics_file)
    now = time.time()

    row = {
        "run_tag": run_tag or _git("rev-parse", "--abbrev-ref", "HEAD"),
        "parent_commit": _git("rev-parse", "--short", "HEAD~1"),
        "child_commit": _git("rev-parse", "--short", "HEAD"),
        "hypothesis": hypothesis,
        "category": category,
        "patch": _git("diff", "HEAD~1", "HEAD", "--", "train.py"),
        "started_at": metrics.get("total_seconds", None) and (now - metrics["total_seconds"]),
        "finished_at": now,
        "exit_status": 0 if outcome != "crash" else 1,
        "val_bpb": metrics.get("val_bpb"),
        "training_seconds": metrics.get("training_seconds"),
        "total_seconds": metrics.get("total_seconds"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "total_tokens_M": metrics.get("total_tokens_M"),
        "num_steps": metrics.get("num_steps"),
        "num_params_M": metrics.get("num_params_M"),
        "depth": metrics.get("depth"),
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
    if sort not in SORTABLE_COLUMNS:
        raise ValueError(f"Invalid sort column: {sort!r} (allowed: {SORTABLE_COLUMNS})")
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


def export_tsv(conn, output_path):
    rows = query_experiments(conn, sort="id")
    with open(output_path, "w") as f:
        f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        for r in rows:
            commit = r["child_commit"] or "unknown"
            val_bpb = f"{r['val_bpb']:.6f}" if r["val_bpb"] is not None else "0.000000"
            mem_gb = f"{r['peak_vram_mb'] / 1024:.1f}" if r["peak_vram_mb"] else "0.0"
            status = r["outcome"] or "unknown"
            desc = r["hypothesis"] or ""
            f.write(f"{commit}\t{val_bpb}\t{mem_gb}\t{status}\t{desc}\n")
    print(f"Exported {len(rows)} experiments to {output_path}")


def import_tsv(conn, input_path):
    with open(input_path) as f:
        header = f.readline()
        count = 0
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            commit, val_bpb, memory_gb, status, desc = parts[0], parts[1], parts[2], parts[3], parts[4]
            conn.execute(
                """INSERT INTO experiments (child_commit, val_bpb, peak_vram_mb, outcome, hypothesis)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    commit,
                    float(val_bpb) if val_bpb != "0.000000" else None,
                    float(memory_gb) * 1024 if memory_gb != "0.0" else None,
                    status,
                    desc,
                ],
            )
            count += 1
    conn.commit()
    print(f"Imported {count} experiments from {input_path}")


def _format_table(rows):
    if not rows:
        print("No experiments found.")
        return
    cols = ["id", "outcome", "val_bpb", "peak_vram_mb", "depth", "num_steps", "child_commit", "hypothesis"]
    cols = [c for c in cols if c in rows[0]]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.6f}" if c == "val_bpb" else f"{v:.1f}"
            vals.append(str(v if v is not None else "").ljust(widths[c]))
        print(" | ".join(vals))


def main():
    parser = argparse.ArgumentParser(description="zimzum experiment database")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="Record an experiment")
    rec.add_argument("--hypothesis", required=True)
    rec.add_argument("--category", default="unknown")
    rec.add_argument("--outcome", required=True, choices=OUTCOMES)
    rec.add_argument("--run-tag")
    rec.add_argument("--metrics-file", default="metrics.json")
    rec.add_argument("--notes")

    show = sub.add_parser("show", help="Show experiments")
    show.add_argument("--outcome", choices=OUTCOMES)
    show.add_argument("--sort", default="id")
    show.add_argument("--last", type=int)

    exp = sub.add_parser("export", help="Export to TSV")
    exp.add_argument("--output", default="results_export.tsv")

    imp = sub.add_parser("import-tsv", help="Import from legacy results.tsv")
    imp.add_argument("--input", default="results.tsv")

    args = parser.parse_args()
    conn = init_db()

    if args.command == "record":
        row = record_experiment(
            conn, args.hypothesis, args.category, args.outcome,
            run_tag=args.run_tag, metrics_file=args.metrics_file, notes=args.notes,
        )
        print(f"Recorded: {row['outcome']} | val_bpb={row['val_bpb']} | {row['hypothesis']}")
    elif args.command == "show":
        rows = query_experiments(conn, outcome=args.outcome, sort=args.sort, last=args.last)
        _format_table(rows)
    elif args.command == "export":
        export_tsv(conn, args.output)
    elif args.command == "import-tsv":
        import_tsv(conn, args.input)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
