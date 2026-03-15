"""
zimzum episode runner — sets up and scores inner-loop sessions.
Does not launch the inner agent — the outer agent or human does that.

Usage:
    python episode.py --repo /path/to/autoresearch --tag ep-001 --budget 2h
    python episode.py --repo /path/to/autoresearch --tag ep-001 --score
"""

import argparse
import json
import os
import sqlite3
import subprocess
import time


def git(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def run_episode(repo, budget_seconds, tag):
    """Set up an episode branch and record baseline state."""
    t0 = time.time()

    baseline_bpb = None
    baseline_metrics = os.path.join(repo, "metrics.json")
    try:
        with open(baseline_metrics) as f:
            baseline_bpb = json.load(f).get("val_bpb")
    except FileNotFoundError:
        pass

    baseline_commit = git(repo, "rev-parse", "--short", "HEAD")
    branch = f"zimzum/{tag}"
    git(repo, "checkout", "-b", branch)

    print(f"Episode {tag}: baseline={baseline_bpb}, commit={baseline_commit}, budget={budget_seconds}s")
    print(f"Episode branch '{branch}' created. Run the inner agent now.")
    print(f"When done, run:  python episode.py --repo {repo} --tag {tag} --score")

    return {
        "tag": tag,
        "baseline_commit": baseline_commit,
        "baseline_bpb": baseline_bpb,
        "branch": branch,
        "started_at": t0,
        "budget_seconds": budget_seconds,
    }


def score_episode(repo, tag):
    """Score a completed episode by reading its experiments.db.
    Filters by run_tag to only count experiments from this episode."""
    db_path = os.path.join(repo, "experiments.db")
    branch = f"zimzum/{tag}"

    conn = sqlite3.connect(db_path)
    # Ensure table exists (may be empty if no experiments ran)
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY, run_tag TEXT, val_bpb REAL, outcome TEXT
    )""")
    cursor = conn.execute(
        "SELECT val_bpb, outcome FROM experiments WHERE run_tag = ? ORDER BY id",
        [branch],
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No experiments recorded for {branch}.")
        return None

    keeps = [(bpb, outcome) for bpb, outcome in rows if outcome == "keep" and bpb is not None]
    discards = sum(1 for _, outcome in rows if outcome == "discard")
    crashes = sum(1 for _, outcome in rows if outcome == "crash")
    best_bpb = min(bpb for bpb, _ in keeps) if keeps else None

    result = {
        "tag": tag,
        "branch": branch,
        "total_experiments": len(rows),
        "keeps": len(keeps),
        "discards": discards,
        "crashes": crashes,
        "best_bpb": best_bpb,
    }

    print(f"Episode {tag}: {len(rows)} experiments, {len(keeps)} kept, best={best_bpb}")
    return result


def main():
    parser = argparse.ArgumentParser(description="zimzum episode runner")
    parser.add_argument("--repo", required=True, help="Path to autoresearch repo")
    parser.add_argument("--tag", required=True, help="Episode tag (e.g. ep-001)")
    parser.add_argument("--budget", default="2h", help="Time budget (e.g. 2h, 30m)")
    parser.add_argument("--score", action="store_true", help="Score a completed episode")
    args = parser.parse_args()

    if args.score:
        result = score_episode(args.repo, args.tag)
        if result:
            with open(f"episode_{args.tag}.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"Written to episode_{args.tag}.json")
    else:
        budget_str = args.budget
        if budget_str.endswith("h"):
            budget_seconds = int(float(budget_str[:-1]) * 3600)
        elif budget_str.endswith("m"):
            budget_seconds = int(float(budget_str[:-1]) * 60)
        else:
            budget_seconds = int(budget_str)

        result = run_episode(args.repo, budget_seconds, args.tag)
        with open(f"episode_{args.tag}.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
