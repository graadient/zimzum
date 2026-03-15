"""
zimzum judge — scores a checkpoint honestly.
Loads the model, calls model(x) for logits only (targets never touch
candidate code), computes cross-entropy, writes val_bpb to metrics.json.

Designed to wrap karpathy/autoresearch. Drop this file into the repo
and run it after train.py saves a checkpoint.

Usage: python judge.py [--mutable train.py]
"""

import argparse
import json
import math
import subprocess
import time

import torch
import torch.nn.functional as F


def git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def verify_surface(mutable_files):
    """Check only allowed files were modified. Fail-closed."""
    has_parent = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD~1"],
        capture_output=True,
    ).returncode == 0
    if not has_parent:
        print("NOTE: first commit — surface check skipped.")
        return True

    changed_str = git("diff", "--name-only", "HEAD~1", "HEAD")
    if changed_str is None:
        print("SURFACE CHECK FAILED: could not run git diff. Failing closed.")
        return False

    allowed = set(mutable_files)
    forbidden = [f for f in changed_str.split("\n") if f and f not in allowed]
    if forbidden:
        print(f"SURFACE VIOLATION: {forbidden}")
        return False
    return True


def evaluate_bpb(model, tokenizer, batch_size):
    """Compute bits-per-byte outside candidate code.
    model(x) returns logits — targets are never passed to the candidate."""
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
            y_flat = y.view(-1)
            nbytes = token_bytes[y_flat]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def main():
    parser = argparse.ArgumentParser(description="zimzum judge")
    parser.add_argument("--mutable", nargs="*", default=["train.py"],
                        help="Files the agent is allowed to modify")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    t0 = time.time()

    if not verify_surface(args.mutable):
        print("Aborting — forbidden files were modified.")
        with open("metrics.json", "w") as f:
            json.dump({"val_bpb": None, "status": "surface_violation"}, f, indent=2)
        return

    # Load checkpoint config
    with open("checkpoint_config.json") as f:
        config_dict = json.load(f)

    # Import model from train.py (behind __main__ guard)
    try:
        from train import GPT, GPTConfig
    except Exception as e:
        print(f"JUDGE ERROR: failed to import model: {e}")
        with open("metrics.json", "w") as f:
            json.dump({"val_bpb": None, "status": "judge_error", "error": str(e)}, f, indent=2)
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
    print(f"Model loaded in {time.time() - t0:.1f}s, running eval...")

    try:
        val_bpb = evaluate_bpb(model, tokenizer, args.batch_size)
    except Exception as e:
        print(f"JUDGE ERROR: evaluation failed: {e}")
        with open("metrics.json", "w") as f:
            json.dump({"val_bpb": None, "status": "judge_error", "error": str(e)}, f, indent=2)
        return

    t1 = time.time()
    print(f"val_bpb: {val_bpb:.6f}  ({t1 - t0:.1f}s)")

    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics["val_bpb"] = round(val_bpb, 6)
    metrics["judge_eval_seconds"] = round(t1 - t0, 1)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"metrics.json updated with val_bpb={val_bpb:.6f}")


if __name__ == "__main__":
    main()
