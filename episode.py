"""
zimzum episode runner — runs one scored inner-loop session.
Resets the inner repo to a clean baseline, runs the inner agent for a
fixed budget, and reports the episode outcome.

Usage: python episode.py --repo /path/to/autoresearch --budget 2h --tag ep-001
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def git(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def run_episode(repo, budget_seconds, tag):
    """Run one inner-loop episode and return the outcome."""
    t0 = time.time()

    # Record baseline val_bpb from the current best in the repo
    baseline_metrics = os.path.join(repo, "metrics.json")
    baseline_bpb = None
    if os.path.exists(baseline_metrics):
        with open(baseline_metrics) as f:
            m = json.load(f)
        baseline_bpb = m.get("val_bpb")

    # Create episode branch from current HEAD
    baseline_commit = git(repo, "rev-parse", "--short", "HEAD")
    branch = f"zimzum/{tag}"
    git(repo, "checkout", "-b", branch)

    print(f"Episode {tag}: baseline={baseline_bpb}, commit={baseline_commit}, budget={budget_seconds}s")

    # The inner loop is driven by the agent reading program.md in the repo.
    # We don't run it here — the agent runs it externally.
    # This script just sets up and tears down the episode.
    print(f"Episode branch '{branch}' created. Run the inner agent now.")
    print(f"When done (or after {budget_seconds}s), run:")
    print(f"  python episode.py --repo {repo} --tag {tag} --score")

    return {
        "tag": tag,
        "baseline_commit": baseline_commit,
        "baseline_bpb": baseline_bpb,
        "branch": branch,
        "started_at": t0,
        "budget_seconds": budget_seconds,
    }


def score_episode(repo, tag):
    """Score a completed episode by reading its experiments.db."""
    db_path = os.path.join(repo, "experiments.db")
    if not os.path.exists(db_path):
        print(f"No experiments.db found in {repo}")
        return None

    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT val_bpb, outcome FROM experiments ORDER BY id"
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No experiments recorded.")
        return None

    keeps = [(bpb, outcome) for bpb, outcome in rows if outcome == "keep" and bpb is not None]
    discards = sum(1 for _, outcome in rows if outcome == "discard")
    crashes = sum(1 for _, outcome in rows if outcome == "crash")
    best_bpb = min(bpb for bpb, _ in keeps) if keeps else None

    result = {
        "tag": tag,
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
        # Parse budget
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
