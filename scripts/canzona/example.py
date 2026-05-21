#!/usr/bin/env python3
"""Tiny causal-LM precision check for FSDP-Canzona.

The model is intentionally small, but it is a real Transformer-style causal LM:
token embedding, positional embedding, PyTorch TransformerEncoderLayer blocks,
final LayerNorm, and an LM head.

This script compares two optimizer paths:

1. A replicated full-parameter baseline, equivalent to single-rank training or
   DDP after gradients have been all-reduced.
2. FSDP-Canzona, where every rank owns only a uniform shard of each selected 2D
   matrix parameter.

The tiny LM produces real forward/backward gradients. FSDP-Canzona receives the
corresponding rank-local shards of those gradients, reconstructs full matrices
for Muon/SOAP, scatters update shards back, and is compared against the
replicated full-parameter baseline after every optimizer step.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matrix_based_optimizer import Muon, SOAP, is_param_use_matrix_based_optim  # noqa: E402


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_size, num_layers, num_heads, ffn_hidden_size):
        super().__init__()
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds configured context length {self.seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embed(input_ids) + self.pos_embed[positions].unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=["muon", "soap"], default="muon")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--balance", choices=["no", "single", "slot", "global"], default="global")
    parser.add_argument("--balance-cost", choices=["numel", "flops"], default="numel")
    parser.add_argument("--overlap", choices=["none", "full"], default="none")
    parser.add_argument("--no-fused-comm", action="store_true", help="Use all_gather/scatter instead of all_to_all_single.")
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--soap-precondition-frequency", type=int, default=2)
    parser.add_argument("--soap-max-precond-dim", type=int, default=256)
    return parser.parse_args()


def select_device(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        return torch.device("cuda", local_rank % torch.cuda.device_count())
    return torch.device("cpu")


def init_distributed(args, device):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Run this script with torchrun, for example: "
            "torchrun --standalone --nproc-per-node=2 scripts/canzona/example.py"
        )

    backend = args.backend
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), backend


def build_batch(args, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 17)
    tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_len + 1),
        generator=gen,
        dtype=torch.long,
    ).to(device)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def build_model(args, device):
    torch.manual_seed(args.seed)
    model = TinyCausalLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden_size,
    ).to(device)
    model.train()
    return model


def selected_matrix_params(model):
    selected = []
    for name, param in model.named_parameters():
        if not is_param_use_matrix_based_optim(name, param):
            continue
        selected.append((name, param))
    if not selected:
        raise RuntimeError("No 2D Transformer matrix parameters were selected for Canzona.")
    return selected


def validate_uniform_shards(named_params, world_size):
    for name, param in named_params:
        if param.shape[0] % world_size != 0:
            raise ValueError(
                f"Parameter {name} with shape {tuple(param.shape)} cannot be "
                f"uniformly sharded over world_size={world_size} along dim 0."
            )


def make_optimizer(args, param_groups):
    if args.optimizer == "muon":
        return Muon(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ns_steps=5,
            ns_coefficient_type="simple",
        )
    return SOAP(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        precondition_frequency=args.soap_precondition_frequency,
        max_precond_dim=args.soap_max_precond_dim,
    )


def build_reference_optimizer(args, named_params):
    use_key = "use_muon" if args.optimizer == "muon" else "use_soap"
    return make_optimizer(args, [{"params": [p for _, p in named_params], use_key: True}])


def build_fsdp_canzona_optimizer(args, named_params, rank, world_size, group):
    local_params = []
    full_shapes = []
    local_shapes = []
    shard_dims = []

    for name, full_param in named_params:
        local_rows = full_param.shape[0] // world_size
        start = rank * local_rows
        local = torch.nn.Parameter(full_param.data.narrow(0, start, local_rows).clone())
        local.fsdp_name = name
        local.fsdp_full_shape = torch.Size(full_param.shape)
        local.fsdp_local_shape = torch.Size(local.shape)
        local.fsdp_shard_dim = 0
        local_params.append(local)
        full_shapes.append(torch.Size(full_param.shape))
        local_shapes.append(torch.Size(local.shape))
        shard_dims.append(0)

    use_key = "use_muon" if args.optimizer == "muon" else "use_soap"
    param_group = {
        "params": local_params,
        use_key: True,
        "optimizer": args.optimizer,
        "is_fsdp_sharded": True,
        "fsdp_group": group,
        "fsdp_full_shapes": full_shapes,
        "fsdp_local_shapes": local_shapes,
        "fsdp_shard_dims": shard_dims,
        "fsdp_balance": args.balance,
        "fsdp_balance_cost": args.balance_cost,
        "fsdp_fused_comm": not args.no_fused_comm,
        "fsdp_overlap": args.overlap,
    }
    return local_params, make_optimizer(args, [param_group])


def compute_lm_loss(model, input_ids, targets):
    logits = model(input_ids)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def assign_shard_grads(named_params, shard_params, rank, world_size):
    for (name, full_param), shard_param in zip(named_params, shard_params):
        if full_param.grad is None:
            raise RuntimeError(f"Missing gradient for selected parameter {name}")
        local_rows = full_param.grad.shape[0] // world_size
        start = rank * local_rows
        shard_param.grad = full_param.grad.detach().narrow(0, start, local_rows).clone()


def gather_full_shards(local_params, world_size):
    gathered = []
    for local in local_params:
        shards = [torch.empty_like(local.data) for _ in range(world_size)]
        dist.all_gather(shards, local.data.contiguous())
        gathered.append(torch.cat(shards, dim=0))
    return gathered


def compare_params(named_params, fsdp_full_params, device):
    max_abs = torch.zeros(1, device=device)
    max_rel = torch.zeros(1, device=device)
    worst_name = ""
    worst_abs = -1.0
    for (name, full_param), fsdp_full in zip(named_params, fsdp_full_params):
        diff = (full_param.data - fsdp_full).abs()
        denom = full_param.data.abs().clamp_min(1e-8)
        local_abs = diff.max()
        local_rel = (diff / denom).max()
        if local_abs.item() > worst_abs:
            worst_abs = local_abs.item()
            worst_name = name
        max_abs = torch.maximum(max_abs, local_abs.reshape(1))
        max_rel = torch.maximum(max_rel, local_rel.reshape(1))
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_rel, op=dist.ReduceOp.MAX)
    return max_abs.item(), max_rel.item(), worst_name


def main():
    args = parse_args()
    device = select_device(args)
    rank, world_size, backend = init_distributed(args, device)

    if args.no_fused_comm is False and backend == "gloo" and rank == 0:
        print("Using fused all_to_all_single on gloo. If your PyTorch build does not support it, rerun with --no-fused-comm.")

    input_ids, targets = build_batch(args, device)
    model = build_model(args, device)
    named_params = selected_matrix_params(model)
    validate_uniform_shards(named_params, world_size)
    ref_opt = build_reference_optimizer(args, named_params)
    shard_params, shard_opt = build_fsdp_canzona_optimizer(
        args=args,
        named_params=named_params,
        rank=rank,
        world_size=world_size,
        group=dist.group.WORLD,
    )

    if rank == 0:
        print(
            f"tiny_lm layers={args.num_layers} hidden={args.hidden_size} "
            f"heads={args.num_heads} ffn={args.ffn_hidden_size} "
            f"selected_matrices={len(named_params)} optimizer={args.optimizer} "
            f"overlap={args.overlap}"
        )

    passed = True
    for step in range(1, args.steps + 1):
        model.zero_grad(set_to_none=True)
        loss = compute_lm_loss(model, input_ids, targets)
        loss.backward()
        assign_shard_grads(named_params, shard_params, rank, world_size)

        ref_opt.step()
        shard_opt.step()

        fsdp_full_params = gather_full_shards(shard_params, world_size)
        max_abs, max_rel, worst_name = compare_params(named_params, fsdp_full_params, device)
        step_passed = max_abs <= args.atol or max_rel <= args.rtol
        passed = passed and step_passed
        if rank == 0:
            print(
                f"step={step:02d} loss={loss.item():.6f} "
                f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
                f"worst={worst_name} status={'PASS' if step_passed else 'FAIL'}"
            )

    if rank == 0:
        print(f"alignment={'PASS' if passed else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
