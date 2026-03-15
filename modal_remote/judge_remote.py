"""
Thin wrapper around judge.py for Modal remote execution.
Runs in the same directory as train.py/prepare.py, so imports work directly.
Skips the git surface check (not meaningful in a Modal container).
"""

import json
import math
import time

import torch
import torch.nn.functional as F


def evaluate_bpb(model, tokenizer, batch_size):
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


def main():
    t0 = time.time()

    with open("checkpoint_config.json") as f:
        config_dict = json.load(f)

    from train import GPT, GPTConfig

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

    val_bpb = evaluate_bpb(model, tokenizer, 128)

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
