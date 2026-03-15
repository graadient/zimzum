"""
Noise measurement for zimzum experiments.
Runs train.py + judge.py N times with identical seeds and measures
val_bpb variance to establish the noise floor.

Usage:
    uv run measure_noise.py              # 3 runs (~18 min)
    uv run measure_noise.py --runs 5     # 5 runs (~30 min)
    uv run measure_noise.py --alpha 0.1  # 90% confidence MDE
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time


def run_experiment(run_idx):
    """Run train.py + judge.py once and return val_bpb."""
    log_file = f"noise_run_{run_idx}.log"

    # Clean stale artifacts
    for f in ["checkpoint.pt", "checkpoint_config.json", "metrics.json"]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    # Train
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
        print(f"  Run {run_idx}: train.py crashed (exit {result.returncode})")
        return None

    if not os.path.exists("metrics.json"):
        print(f"  Run {run_idx}: no metrics.json produced")
        return None

    # Judge
    print(f"  Run {run_idx}: evaluating...", flush=True)
    result = subprocess.run(
        ["uv", "run", "judge.py"],
        capture_output=True, text=True, timeout=300,
    )
    with open(log_file, "a") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: judge.py crashed (exit {result.returncode})")
        return None

    with open("metrics.json") as f:
        metrics = json.load(f)

    val_bpb = metrics.get("val_bpb")
    dt = time.time() - t0
    print(f"  Run {run_idx}: val_bpb = {val_bpb:.6f} ({dt:.0f}s)")
    return val_bpb


def t_critical(alpha, df):
    """Approximate t-distribution critical value using normal + correction."""
    z = _norm_ppf(1 - alpha / 2)
    g1 = (z**3 + z) / (4 * df)
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    return z + g1 + g2


def _norm_ppf(p):
    """Approximate inverse normal CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0 or p >= 1:
        return float("inf")
    t = math.sqrt(-2 * math.log(min(p, 1 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    return result if p > 0.5 else -result


def main():
    parser = argparse.ArgumentParser(description="Measure val_bpb noise floor")
    parser.add_argument("--runs", type=int, default=3, help="Number of identical runs")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for MDE")
    args = parser.parse_args()

    n = args.runs
    if n < 2:
        print("Need at least 2 runs to measure variance.")
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
        print("Too few successful runs to compute statistics.")
        sys.exit(1)

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(variance)

    df = len(values) - 1
    t_crit = t_critical(args.alpha, df)
    mde = t_crit * std * math.sqrt(2 / len(values))

    print("Summary:")
    print(f"  runs:     {len(values)}/{n}")
    print(f"  mean:     {mean:.6f}")
    print(f"  std:      {std:.6f}")
    print(f"  variance: {variance:.9f}")
    print()
    print(f"Minimum detectable effect (alpha={args.alpha}, two-tailed, n={len(values)}):")
    print(f"  MDE: {mde:.6f}")
    print(f"  (improvements must be > {mde:.6f} val_bpb to be distinguishable from noise)")

    results = {
        "n_runs": len(values),
        "n_requested": n,
        "values": values,
        "mean": mean,
        "std": std,
        "variance": variance,
        "alpha": args.alpha,
        "mde": mde,
    }
    with open("noise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Written to noise_results.json")


if __name__ == "__main__":
    main()
