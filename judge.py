"""
zimzum judge — scores a checkpoint honestly.
Runs from inside autoresearch/ so train.py and prepare.py are importable.
Calls model(x) for logits only — eval targets never touch candidate code.

Usage: cd autoresearch && uv run python ../judge.py
"""

import json
import math
import os
import subprocess
import sys
import time

# Ensure cwd is on the import path (not the script's directory)
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F

GENERATED_ARTIFACTS = {"checkpoint.pt", "checkpoint_config.json", "metrics.json"}


def verify_surface(mutable_files):
    """Check only allowed files were modified or appeared. Fail-closed.
    Checks committed diffs, dirty tracked files, AND untracked files."""
    has_parent = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD~1"], capture_output=True
    ).returncode == 0

    try:
        changed = set()
        if has_parent:
            # Committed changes
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            changed.update(f for f in out.split("\n") if f)
        # Dirty tracked + untracked files via git status
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        for line in status.split("\n"):
            if line and len(line) > 3:
                changed.add(line[3:].strip())
    except Exception:
        print("SURFACE CHECK FAILED: git error. Failing closed.")
        return False

    allowed = set(mutable_files) | GENERATED_ARTIFACTS
    forbidden = [f for f in changed if f not in allowed]
    if forbidden:
        print(f"SURFACE VIOLATION: {forbidden}")
        return False
    return True


def evaluate_bpb(model, tokenizer, batch_size):
    """Compute bits-per-byte outside candidate code.
    model(x) returns logits — targets never touch the candidate."""
    from prepare import MAX_SEQ_LEN, EVAL_TOKENS, make_dataloader, get_token_bytes

    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), autocast_ctx:
        for _ in range(steps):
            x, y, _ = next(val_loader)
            logits = model(x)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                ignore_index=-1, reduction="none",
            )
            nbytes = token_bytes[y.view(-1)]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def _write_error(status, error=None):
    result = {"val_bpb": None, "status": status}
    if error:
        result["error"] = str(error)
    with open("metrics.json", "w") as f:
        json.dump(result, f, indent=2)


def main():
    t0 = time.time()

    if not verify_surface(["train.py"]):
        print("Aborting — forbidden files modified.")
        _write_error("surface_violation")
        return

    with open("checkpoint_config.json") as f:
        config_dict = json.load(f)

    try:
        from train import GPT, GPTConfig
    except Exception as e:
        print(f"JUDGE ERROR: {e}")
        _write_error("judge_error", e)
        return

    device = torch.device("cuda")
    config = GPTConfig(**config_dict)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    state_dict = torch.load("checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    from prepare import Tokenizer
    tokenizer = Tokenizer.from_directory()
    print(f"Model loaded in {time.time() - t0:.1f}s, evaluating...")

    try:
        val_bpb = evaluate_bpb(model, tokenizer, 128)
    except Exception as e:
        print(f"JUDGE ERROR: {e}")
        _write_error("judge_error", e)
        return

    t1 = time.time()
    print(f"val_bpb: {val_bpb:.6f}  ({t1 - t0:.1f}s)")

    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics["val_bpb"] = round(val_bpb, 6)
    metrics["judge_seconds"] = round(t1 - t0, 1)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
