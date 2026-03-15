"""
zimzum experiment database.
Stores experiments and episodes in SQLite. Lives at repo root,
outside autoresearch/ so git operations there don't touch it.

Usage (from repo root):
    python db.py record --hypothesis "try SiLU" --outcome keep
    python db.py show
    python db.py show --outcome keep --sort val_bpb
    python db.py score-episode --tag ep-001
"""

import argparse
import json
import os
import sqlite3
import subprocess
import time

DB_PATH = "experiments.db"
OUTCOMES = ["keep", "discard", "crash"]
PROJECT_DIR = "autoresearch"

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
    val_bpb         REAL,
    training_seconds REAL,
    total_seconds   REAL,
    peak_vram_mb    REAL,
    num_steps       INTEGER,
    num_params_M    REAL,
    depth           INTEGER,
    outcome         TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tag             TEXT UNIQUE,
    base_commit     TEXT,
    policy_commit   TEXT,
    baseline_bpb    REAL,
    best_bpb        REAL,
    score           REAL,
    total_experiments INTEGER,
    keeps           INTEGER,
    discards        INTEGER,
    crashes         INTEGER,
    budget_seconds  REAL,
    started_at      REAL,
    finished_at     REAL
);
"""

SORTABLE = {
    "id", "run_tag", "val_bpb", "training_seconds", "peak_vram_mb",
    "num_steps", "depth", "outcome", "child_commit", "finished_at",
}


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", PROJECT_DIR, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _load_metrics():
    path = os.path.join(PROJECT_DIR, "metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def record(conn, hypothesis, category, outcome, run_tag=None, notes=None):
    metrics = _load_metrics()
    now = time.time()
    row = {
        "run_tag": run_tag or _git("rev-parse", "--abbrev-ref", "HEAD"),
        "parent_commit": _git("rev-parse", "--short", "HEAD~1"),
        "child_commit": _git("rev-parse", "--short", "HEAD"),
        "hypothesis": hypothesis,
        "category": category,
        "patch": _git("diff", "HEAD~1", "HEAD"),
        "started_at": (now - metrics["total_seconds"]) if metrics.get("total_seconds") else None,
        "finished_at": now,
        "val_bpb": metrics.get("val_bpb"),
        "training_seconds": metrics.get("training_seconds"),
        "total_seconds": metrics.get("total_seconds"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "num_steps": metrics.get("num_steps"),
        "num_params_M": metrics.get("num_params_M"),
        "depth": metrics.get("depth"),
        "outcome": outcome,
        "notes": notes,
    }
    cols = list(row.keys())
    conn.execute(
        f"INSERT INTO experiments ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
        [row[c] for c in cols],
    )
    conn.commit()
    print(f"Recorded: {outcome} | val_bpb={row['val_bpb']} | {hypothesis}")
    return row


def show(conn, outcome=None, sort="id", last=None):
    if sort not in SORTABLE:
        raise ValueError(f"Invalid sort: {sort!r}")
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
    rows = [dict(zip(cols, r)) for r in cursor.fetchall()]

    if not rows:
        print("No experiments found.")
        return

    display = ["id", "outcome", "val_bpb", "peak_vram_mb", "depth", "num_steps", "child_commit", "hypothesis"]
    display = [c for c in display if c in rows[0]]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in display}
    print(" | ".join(c.ljust(widths[c]) for c in display))
    print("-+-".join("-" * widths[c] for c in display))
    for r in rows:
        vals = []
        for c in display:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.6f}" if c == "val_bpb" else f"{v:.1f}"
            vals.append(str(v if v is not None else "").ljust(widths[c]))
        print(" | ".join(vals))


def score_episode(conn, tag):
    """Score an episode: baseline_bpb - best_bpb for experiments with this run_tag."""
    branch = f"zimzum/{tag}"
    cursor = conn.execute(
        "SELECT val_bpb, outcome FROM experiments WHERE run_tag = ? ORDER BY id",
        [branch],
    )
    rows = cursor.fetchall()
    if not rows:
        print(f"No experiments for {branch}.")
        return None

    keeps = [bpb for bpb, outcome in rows if outcome == "keep" and bpb is not None]
    discards = sum(1 for _, outcome in rows if outcome == "discard")
    crashes = sum(1 for _, outcome in rows if outcome == "crash")
    best_bpb = min(keeps) if keeps else None

    # Get baseline from first keep (the baseline run)
    baseline_bpb = keeps[0] if keeps else None
    score = (baseline_bpb - best_bpb) if baseline_bpb and best_bpb else None

    conn.execute("""
        INSERT OR REPLACE INTO episodes (tag, best_bpb, baseline_bpb, score,
            total_experiments, keeps, discards, crashes, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [tag, best_bpb, baseline_bpb, score, len(rows), len(keeps), discards, crashes, time.time()])
    conn.commit()

    print(f"Episode {tag}: {len(rows)} experiments, {len(keeps)} kept, "
          f"baseline={baseline_bpb}, best={best_bpb}, score={score}")
    return {"tag": tag, "score": score, "best_bpb": best_bpb, "baseline_bpb": baseline_bpb,
            "total": len(rows), "keeps": len(keeps), "discards": discards, "crashes": crashes}


def main():
    parser = argparse.ArgumentParser(description="zimzum experiment database")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record")
    rec.add_argument("--hypothesis", required=True)
    rec.add_argument("--category", default="unknown")
    rec.add_argument("--outcome", required=True, choices=OUTCOMES)
    rec.add_argument("--run-tag")
    rec.add_argument("--notes")

    sh = sub.add_parser("show")
    sh.add_argument("--outcome", choices=OUTCOMES)
    sh.add_argument("--sort", default="id")
    sh.add_argument("--last", type=int)

    ep = sub.add_parser("score-episode")
    ep.add_argument("--tag", required=True)

    args = parser.parse_args()
    conn = init_db()

    if args.command == "record":
        record(conn, args.hypothesis, args.category, args.outcome,
               run_tag=args.run_tag, notes=args.notes)
    elif args.command == "show":
        show(conn, outcome=args.outcome, sort=args.sort, last=args.last)
    elif args.command == "score-episode":
        score_episode(conn, args.tag)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
