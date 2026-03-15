"""
Independent evaluator for zimzum experiments.
Loads a checkpoint saved by train.py and computes val_bpb using the
immutable evaluate_bpb function from prepare.py. Writes the result
back into metrics.json.

The candidate (train.py) never reports its own val_bpb. This script
is the sole source of truth.

Usage: uv run judge.py
"""

import json
import time

import torch

from prepare import Tokenizer, evaluate_bpb

EVAL_BATCH_SIZE = 128
CHECKPOINT_PT = "checkpoint.pt"
CHECKPOINT_CONFIG = "checkpoint_config.json"
METRICS_JSON = "metrics.json"


def main():
    t0 = time.time()

    with open(CHECKPOINT_CONFIG) as f:
        config_dict = json.load(f)

    # Import model class from train.py (safe — execution is behind __main__ guard)
    from train import GPT, GPTConfig

    device = torch.device("cuda")
    config = GPTConfig(**config_dict)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)

    state_dict = torch.load(CHECKPOINT_PT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = Tokenizer.from_directory()
    print(f"Model loaded in {time.time() - t0:.1f}s, running eval...")

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)

    t1 = time.time()
    print(f"val_bpb: {val_bpb:.6f}  ({t1 - t0:.1f}s)")

    # Update metrics.json with judge-computed val_bpb
    try:
        with open(METRICS_JSON) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics["val_bpb"] = round(val_bpb, 6)
    metrics["judge_eval_seconds"] = round(t1 - t0, 1)

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"metrics.json updated with val_bpb={val_bpb:.6f}")


if __name__ == "__main__":
    main()
