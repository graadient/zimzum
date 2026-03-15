"""
Project-specific evaluation for gpt_pretrain.
Computes bits-per-byte (BPB) entirely outside candidate code.
The model is called as model(x) with NO targets — cross-entropy
is computed here by the judge, not by the candidate.

NOTE: This intentionally duplicates the BPB accumulation logic from
prepare.evaluate_bpb. The divergence is that we never pass targets
to the model. If prepare.evaluate_bpb changes its accumulation math,
this function must be updated to match.
"""

import math

import torch
import torch.nn.functional as F

from .prepare import MAX_SEQ_LEN, EVAL_TOKENS, Tokenizer, make_dataloader, get_token_bytes


@torch.no_grad()
def evaluate(model, project_cfg):
    """
    Evaluate the model and return the primary metric value.
    This is the harness/project contract: evaluate(model, cfg) -> float.
    """
    batch_size = project_cfg.get("eval_batch_size", 128)
    tokenizer = Tokenizer.from_directory()
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast_ctx:
        for _ in range(steps):
            x, y, _ = next(val_loader)
            # model(x) returns logits — targets are NEVER passed to candidate code
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
