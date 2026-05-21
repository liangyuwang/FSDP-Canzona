# FSDP-Canzona Example Scripts

This directory contains small, self-contained scripts for exercising
FSDP-Canzona on a tiny causal Transformer LM built from PyTorch modules. Unlike
Megatron-Canzona's examples, these scripts focus on precision alignment against
a single-rank or DDP-equivalent full-matrix baseline.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `example.py` | Tiny causal LM workload using `torch.nn.TransformerEncoderLayer`. Each rank owns uniform FSDP-style optimizer shards; the baseline owns replicated full matrices, equivalent to single-rank or DDP after gradient all-reduce. |
| `example.sh` | Launches one `example.py` run with environment-variable configuration. |
| `align.sh` | Runs several alignment cases, similar in spirit to the Megatron-Canzona alignment commands. |

## Precision Alignment

The alignment test builds a deterministic tiny causal LM and batch of token IDs.
It then runs real forward/backward passes and compares two optimizer paths:

1. **Baseline:** selected full Transformer matrices updated by Muon or SOAP.
2. **FSDP-Canzona:** rank-local shards of the same selected matrices updated
   through gather → compute → scatter → local update.

After each optimizer step, the script gathers the FSDP-Canzona shards and
compares them against the baseline full parameters.

### CUDA / NCCL

```bash
torchrun --standalone --nproc-per-node=2 \
  scripts/canzona/example.py \
  --optimizer muon \
  --device cuda \
  --backend nccl \
  --steps 4 \
  --balance global \
  --balance-cost flops \
  --overlap full \
  --num-layers 2 \
  --hidden-size 32 \
  --num-heads 4 \
  --ffn-hidden-size 64
```

### CPU / Gloo

For CPU-only environments, use the unfused communication path:

```bash
torchrun --standalone --nproc-per-node=2 \
  scripts/canzona/example.py \
  --optimizer soap \
  --device cpu \
  --backend gloo \
  --no-fused-comm \
  --steps 4
```

Muon may require CUDA on some PyTorch builds because its Newton-Schulz path uses
BF16 matrix multiplications.

## Shell Wrappers

Run one configured example:

```bash
OPTIMIZER=muon NPROC_PER_NODE=2 DEVICE=cuda BACKEND=nccl \
  BALANCE=global BALANCE_COST=flops OVERLAP=full FUSED_COMM=1 \
  NUM_LAYERS=2 HIDDEN_SIZE=32 NUM_HEADS=4 FFN_HIDDEN_SIZE=64 \
  bash scripts/canzona/example.sh
```

Run multiple alignment cases:

```bash
NPROC_PER_NODE=2 DEVICE=cuda BACKEND=nccl bash scripts/canzona/align.sh
```

## Useful Variants

```bash
# Compare the simple scheduling path.
torchrun --standalone --nproc-per-node=2 \
  scripts/canzona/example.py --optimizer muon --balance no

# Compare the globally balanced fused all-to-all path.
torchrun --standalone --nproc-per-node=4 \
  scripts/canzona/example.py --optimizer muon --balance global --balance-cost flops

# Exercise SOAP for multiple preconditioner updates.
torchrun --standalone --nproc-per-node=2 \
  scripts/canzona/example.py \
  --optimizer soap \
  --steps 6 \
  --soap-precondition-frequency 2
```

## Expected Output

The script prints per-step maximum absolute and relative differences:

```text
tiny_lm layers=2 hidden=32 heads=4 ffn=64 selected_matrices=8 optimizer=muon overlap=full
step=01 loss=4.941732 max_abs=0.000000e+00 max_rel=0.000000e+00 worst=blocks.layers.0.self_attn.in_proj_weight status=PASS
step=02 loss=4.889214 max_abs=0.000000e+00 max_rel=0.000000e+00 worst=blocks.layers.0.self_attn.in_proj_weight status=PASS
alignment=PASS
```

Small non-zero differences can appear when using BF16/CUDA kernels. Tune
`--atol` and `--rtol` if your hardware or PyTorch version produces slightly
different low-level math.

`OVERLAP=none` keeps the serial gather -> compute -> scatter -> update path.
`OVERLAP=full` enables the pipelined path and currently requires fused
communication (`FUSED_COMM=1`).
