"""
zimzum noise measurement — runs train + judge N times to measure
the noise floor and compute minimum detectable effect size.

Usage: python noise.py --runs 3
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time


def run_experiment(run_idx):
    log_file = f"noise_run_{run_idx}.log"

    for f in ["checkpoint.pt", "checkpoint_config.json", "metrics.json"]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    print(f"  Run {run_idx}: training...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "train.py"],
        capture_output=True, text=True, timeout=900,
    )
    with open(log_file, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: training crashed (exit {result.returncode})")
        return None

    if not os.path.exists("metrics.json"):
        print(f"  Run {run_idx}: no metrics.json produced")
        return None

    print(f"  Run {run_idx}: evaluating...", flush=True)
    result = subprocess.run(
        ["uv", "run", "judge.py"],
        capture_output=True, text=True, timeout=300,
    )
    with open(log_file, "a") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: judge crashed (exit {result.returncode})")
        return None

    with open("metrics.json") as f:
        metrics = json.load(f)

    val_bpb = metrics.get("val_bpb")
    dt = time.time() - t0
    print(f"  Run {run_idx}: val_bpb = {val_bpb:.6f} ({dt:.0f}s)")
    return val_bpb


def _t_critical(alpha, df):
    z = _norm_ppf(1 - alpha / 2)
    g1 = (z**3 + z) / (4 * df)
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    return z + g1 + g2


def _norm_ppf(p):
    if p <= 0 or p >= 1:
        return float("inf")
    t = math.sqrt(-2 * math.log(min(p, 1 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    return result if p > 0.5 else -result


def main():
    parser = argparse.ArgumentParser(description="Measure val_bpb noise floor")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    n = args.runs
    if n < 2:
        print("Need at least 2 runs.")
        sys.exit(1)

    print(f"Noise measurement: {n} runs with seed=42")
    print()

    values = []
    for i in range(n):
        val = run_experiment(i)
        if val is not None:
            values.append(val)

    print()
    if len(values) < 2:
        print("Too few successful runs.")
        sys.exit(1)

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(variance)
    df = len(values) - 1
    t_crit = _t_critical(args.alpha, df)
    mde = t_crit * std * math.sqrt(2 / len(values))

    print("Summary:")
    print(f"  mean:     {mean:.6f}")
    print(f"  std:      {std:.6f}")
    print(f"  MDE:      {mde:.6f}")
    print(f"  (improvements must be > {mde:.6f} val_bpb to beat noise)")

    with open("noise_results.json", "w") as f:
        json.dump({"mean": mean, "std": std, "mde": mde, "values": values, "alpha": args.alpha}, f, indent=2)
    print("\nWritten to noise_results.json")


if __name__ == "__main__":
    main()
