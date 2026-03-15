"""
Generic judge for zimzum experiments.
Reads project.yaml, loads a checkpoint, calls the project's evaluate function,
and writes the result to metrics.json. No project-specific imports.

Usage: uv run -m harness.judge --project projects/gpt_pretrain/project.yaml
"""

import argparse
import importlib
import json
import time

import torch

from .config import load_project
from .surface import verify_surface


def main():
    parser = argparse.ArgumentParser(description="zimzum judge")
    parser.add_argument("--project", default=None, help="Path to project.yaml")
    args = parser.parse_args()

    cfg = load_project(args.project)
    t0 = time.time()

    # Surface check
    if not verify_surface(cfg["mutable_files"]):
        print("Aborting judge — forbidden files were modified.")
        with open(cfg["metrics"], "w") as f:
            json.dump({cfg["primary_metric"]: None, "status": "surface_violation"}, f, indent=2)
        return

    # Load checkpoint config
    with open(cfg["config"]) as f:
        config_dict = json.load(f)

    # Dynamically import model class from project's model module
    model_mod = importlib.import_module(cfg["model_module"])
    ModelClass = getattr(model_mod, cfg["model_class"])
    ConfigClass = getattr(model_mod, cfg["config_class"])

    device = torch.device("cuda")
    config = ConfigClass(**config_dict)

    with torch.device("meta"):
        model = ModelClass(config)
    model.to_empty(device=device)

    state_dict = torch.load(cfg["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f}s, running eval...")

    # Dynamically import and call project's evaluate function
    eval_mod = importlib.import_module(cfg["eval_module"])
    eval_fn = getattr(eval_mod, cfg["eval_fn"])
    metric_value = eval_fn(model, cfg)

    t1 = time.time()
    metric_name = cfg["primary_metric"]
    print(f"{metric_name}: {metric_value:.6f}  ({t1 - t0:.1f}s)")

    # Update metrics.json
    try:
        with open(cfg["metrics"]) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics[metric_name] = round(metric_value, 6)
    metrics["judge_eval_seconds"] = round(t1 - t0, 1)

    with open(cfg["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{cfg['metrics']} updated with {metric_name}={metric_value:.6f}")


if __name__ == "__main__":
    main()
