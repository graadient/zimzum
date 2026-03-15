"""
Independent evaluator for zimzum experiments.
Loads a checkpoint saved by train.py, gets logits from the model,
and computes val_bpb OUTSIDE candidate code. The model never sees
eval targets — loss is computed here, not in the candidate's forward().

Usage: uv run judge.py
"""

import json
import math
import time

import torch
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, EVAL_TOKENS, Tokenizer, make_dataloader, get_token_bytes

EVAL_BATCH_SIZE = 128
CHECKPOINT_PT = "checkpoint.pt"
CHECKPOINT_CONFIG = "checkpoint_config.json"
METRICS_JSON = "metrics.json"


@torch.no_grad()
def judge_evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte — computed entirely outside candidate code.
    The model is called as model(x) with NO targets, returning logits.
    Cross-entropy is computed here by the judge, not by the candidate.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        # model(x) returns logits — targets are NEVER passed to candidate code
        logits = model(x)
        # Loss computed here by the judge
        loss_flat = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            ignore_index=-1, reduction="none",
        )
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def verify_surface():
    """Check that only train.py was modified between parent and child commits."""
    import subprocess
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")
    except Exception:
        print("WARNING: could not verify edit surface (no git history?)")
        return True

    forbidden = [f for f in changed if f and f != "train.py"]
    if forbidden:
        print(f"SURFACE VIOLATION: files other than train.py were modified: {forbidden}")
        return False
    return True


def main():
    t0 = time.time()

    # Surface check: only train.py should have changed
    if not verify_surface():
        print("Aborting judge — forbidden files were modified.")
        # Write a failed metrics.json so the loop knows
        with open(METRICS_JSON, "w") as f:
            json.dump({"val_bpb": None, "status": "surface_violation"}, f, indent=2)
        return

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
        val_bpb = judge_evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)

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
