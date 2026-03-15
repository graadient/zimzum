"""
zimzum noise measurement. Runs train + judge N times from autoresearch/
to measure val_bpb variance and compute minimum detectable effect.

Usage: python noise.py --runs 3
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

PROJECT_DIR = "autoresearch"


def run_once(run_idx):
    log_file = f"noise_run_{run_idx}.log"

    for f in ["checkpoint.pt", "checkpoint_config.json", "metrics.json"]:
        try:
            os.remove(os.path.join(PROJECT_DIR, f))
        except FileNotFoundError:
            pass

    print(f"  Run {run_idx}: training...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "train.py"], cwd=PROJECT_DIR,
        capture_output=True, text=True, timeout=900,
    )
    with open(log_file, "w") as f:
        f.write(result.stdout + result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: train crashed")
        return None

    metrics_path = os.path.join(PROJECT_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  Run {run_idx}: no metrics.json")
        return None

    print(f"  Run {run_idx}: judging...", flush=True)
    result = subprocess.run(
        ["python", "../judge.py"], cwd=PROJECT_DIR,
        capture_output=True, text=True, timeout=300,
    )
    with open(log_file, "a") as f:
        f.write(result.stdout + result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: judge crashed")
        return None

    with open(metrics_path) as f:
        val = json.load(f).get("val_bpb")
    print(f"  Run {run_idx}: val_bpb = {val:.6f} ({time.time() - t0:.0f}s)")
    return val


def main():
    parser = argparse.ArgumentParser(description="Measure val_bpb noise floor")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    n = args.runs
    if n < 2:
        print("Need at least 2 runs.")
        sys.exit(1)

    print(f"Noise measurement: {n} runs")
    values = []
    for i in range(n):
        val = run_once(i)
        if val is not None:
            values.append(val)

    if len(values) < 2:
        print("Too few successful runs.")
        sys.exit(1)

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(variance)

    # Approximate t-critical (Cornish-Fisher)
    z = 1.96  # alpha=0.05
    df = len(values) - 1
    z += (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    mde = z * std * math.sqrt(2 / len(values))

    print(f"\nmean={mean:.6f}  std={std:.6f}  MDE={mde:.6f}")
    print(f"Improvements must be > {mde:.6f} val_bpb to beat noise.")

    with open("noise_results.json", "w") as f:
        json.dump({"mean": mean, "std": std, "mde": mde, "values": values}, f, indent=2)


if __name__ == "__main__":
    main()
