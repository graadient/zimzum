"""
Generic noise measurement for zimzum experiments.
Runs train + judge N times with identical seeds and measures
variance of the primary metric to establish the noise floor.

Usage: uv run -m harness.noise --project projects/gpt_pretrain/project.yaml --runs 3
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

from .config import load_project


def run_experiment(run_idx, cfg):
    """Run train + judge once and return the primary metric value."""
    log_file = f"noise_run_{run_idx}.log"
    metric_name = cfg["primary_metric"]

    for f in [cfg["checkpoint"], cfg["config"], cfg["metrics"]]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    print(f"  Run {run_idx}: training...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cfg["train_cmd"].split(),
        capture_output=True, text=True, timeout=900,
    )
    with open(log_file, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: training crashed (exit {result.returncode})")
        return None

    if not os.path.exists(cfg["metrics"]):
        print(f"  Run {run_idx}: no {cfg['metrics']} produced")
        return None

    print(f"  Run {run_idx}: evaluating...", flush=True)
    result = subprocess.run(
        ["uv", "run", "-m", "harness.judge", "--project", cfg["_yaml_path"]],
        capture_output=True, text=True, timeout=300,
    )
    with open(log_file, "a") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  Run {run_idx}: judge crashed (exit {result.returncode})")
        return None

    with open(cfg["metrics"]) as f:
        metrics = json.load(f)

    val = metrics.get(metric_name)
    dt = time.time() - t0
    print(f"  Run {run_idx}: {metric_name} = {val:.6f} ({dt:.0f}s)")
    return val


def t_critical(alpha, df):
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
    parser = argparse.ArgumentParser(description="Measure noise floor")
    parser.add_argument("--project", default=None, help="Path to project.yaml")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_project(args.project)
    cfg["_yaml_path"] = args.project or os.environ.get("ZIMZUM_PROJECT", "projects/gpt_pretrain/project.yaml")
    metric_name = cfg["primary_metric"]
    n = args.runs

    if n < 2:
        print("Need at least 2 runs.")
        sys.exit(1)

    print(f"Noise measurement: {n} runs, metric={metric_name}")
    print()

    values = []
    for i in range(n):
        val = run_experiment(i, cfg)
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
    t_crit = t_critical(args.alpha, df)
    mde = t_crit * std * math.sqrt(2 / len(values))

    print("Summary:")
    print(f"  runs:     {len(values)}/{n}")
    print(f"  mean:     {mean:.6f}")
    print(f"  std:      {std:.6f}")
    print(f"  variance: {variance:.9f}")
    print()
    print(f"MDE (alpha={args.alpha}): {mde:.6f}")
    print(f"  (improvements must be > {mde:.6f} {metric_name} to beat noise)")

    results = {
        "metric": metric_name, "n_runs": len(values), "values": values,
        "mean": mean, "std": std, "variance": variance,
        "alpha": args.alpha, "mde": mde,
    }
    with open("noise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten to noise_results.json")


if __name__ == "__main__":
    main()
