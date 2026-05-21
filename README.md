# FSDP-Canzona

FSDP-Canzona is an FSDP-oriented implementation of
[Canzona](https://arxiv.org/html/2602.06079), adapted from the Megatron-based
[Megatron-Canzona](https://github.com/liangyuwang/Megatron-Canzona) prototype.

Canzona makes matrix-based optimizers such as Muon and SOAP practical under
distributed sharding. These optimizers need full 2D matrices for operations like
Newton-Schulz orthogonalization or Shampoo-style preconditioning, while FSDP
keeps only per-rank parameter shards. FSDP-Canzona bridges that mismatch by
assigning each full-matrix optimizer task to a host rank, gathering the needed
gradient shards, computing the full update, scattering update shards back, and
then applying the local shard update.

![FSDP-Canzona optimizer-step overview](image/overview.png)

The design mirrors the Megatron-Canzona TP path: both TP and FSDP split each
parameter uniformly across ranks, so the same gather → compute → scatter →
update schedule can be reused with FSDP process groups and shard metadata.

## Key Ideas

- **Full-matrix optimizer compute:** Muon/SOAP run on reconstructed full 2D
  gradients, not on partial shards.
- **Logical host assignment:** each matrix update is assigned to one rank for
  compute, independent of where the parameter shard physically lives.
- **Micro-group scheduling:** parameters are grouped so gather, compute,
  scatter, and local update can be pipelined and load-balanced.
- **Fused communication path:** shard movement can use fused `all_to_all_single`
  when enabled.
- **Optional parameter splitting:** QKV, FC1, and linear-attention input
  projections can be split into smaller 2D matrices before optimization.

## Package Layout

```text
matrix_based_optimizer/
├── load_balanced_fsdp_executor.py  # FSDP gather/compute/scatter/update executor
├── split_grad_and_state.py         # QKV/FC1/in-proj split and reassembly helpers
├── utils.py                        # parameter tagging and cost estimates
└── optimizers/
    ├── base_optim.py               # common optimizer orchestration
    ├── muon.py                     # Muon kernel
    └── soap.py                     # SOAP kernel
```

## FSDP Contract

FSDP-Canzona intentionally avoids relying on PyTorch FSDP private internals. The
training stack provides shard metadata either through parameter attributes or
param-group fields.

```python
param_group = {
    "params": local_matrix_shards,
    "use_muon": True,
    "is_fsdp_sharded": True,
    "fsdp_group": process_group,          # optional; defaults to WORLD
    "fsdp_full_shapes": full_shapes,      # list[torch.Size], one per param
    "fsdp_local_shapes": local_shapes,    # optional; defaults to p.shape
    "fsdp_shard_dims": shard_dims,        # optional; defaults to 0
}
```

Equivalent per-parameter attributes are also accepted:

```python
p.fsdp_full_shape = torch.Size([hidden_out, hidden_in])
p.fsdp_local_shape = p.shape
p.fsdp_shard_dim = 0
```

The current implementation assumes uniform per-parameter sharding:

```text
full_shape[shard_dim] == local_shape[shard_dim] * fsdp_world_size
```

All non-sharded dimensions must match exactly.

## Usage Sketch

```python
from matrix_based_optimizer import Muon

optimizer = Muon(
    [
        {
            "params": matrix_shards,
            "use_muon": True,
            "is_fsdp_sharded": True,
            "fsdp_group": fsdp_group,
            "fsdp_full_shapes": full_shapes,
            "fsdp_local_shapes": local_shapes,
            "fsdp_shard_dims": shard_dims,
            "fsdp_balance": "global",     # "no", "single", "slot", "global"
            "fsdp_fused_comm": True,      # use all_to_all_single
            "fsdp_balance_cost": "flops", # or "numel"
        }
    ],
    lr=1e-3,
)
```

For non-sharded local parameters, the optimizers behave like regular
`torch.optim.Optimizer` subclasses.

## Current Status

- FSDP-oriented executor and Muon/SOAP integration are implemented.
- Megatron TP-specific dependencies have been removed from the main path.
- Cross-world-size optimizer-state checkpoint remapping is not implemented yet.
- CUDA graph capture is disabled for the FSDP communication path.
