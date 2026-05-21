# FSDP-Canzona

This repository ports the Canzona matrix-based optimizer idea from
Megatron-Canzona to an FSDP-style sharding model.

The important assumption is the one shown in `image/overview.png`: every matrix
parameter is uniformly sharded per parameter across the FSDP process group. A
rank may own only a shard of a 2D parameter, but Muon/SOAP still run on a full
2D matrix. FSDP-Canzona therefore assigns each full-matrix optimizer task to a
host rank, gathers the corresponding gradient shards to that rank, computes the
full update, scatters update shards back to the owning ranks, and applies the
local shard update.

## Current Components

- `matrix_based_optimizer/optimizers/muon.py`: Muon compute kernel.
- `matrix_based_optimizer/optimizers/soap.py`: SOAP compute kernel.
- `matrix_based_optimizer/split_grad_and_state.py`: QKV/FC1/in-proj splitting
  before matrix optimization, then reassembly after compute.
- `matrix_based_optimizer/load_balanced_fsdp_executor.py`: FSDP Canzona
  gather/compute/scatter/update executor with micro-group load balancing.
- `matrix_based_optimizer/utils.py`: optimizer tagging and cost estimation.

## FSDP Param-Group Contract

FSDP-Canzona intentionally avoids depending on PyTorch FSDP private internals.
The training stack should pass shard metadata through parameter attributes or
param-group fields.

Required for sharded matrix params:

```python
param_group = {
    "params": local_matrix_shards,
    "is_fsdp_sharded": True,
    "fsdp_group": process_group,          # optional; defaults to WORLD
    "fsdp_full_shapes": full_shapes,      # list[torch.Size], one per param
    "fsdp_local_shapes": local_shapes,    # optional; defaults to p.shape
    "fsdp_shard_dims": shard_dims,        # optional; defaults to 0
}
```

Equivalent parameter attributes are also accepted:

```python
p.fsdp_full_shape = torch.Size([hidden_out, hidden_in])
p.fsdp_local_shape = p.shape
p.fsdp_shard_dim = 0
```

The initial implementation supports uniform per-parameter sharding only:
`full_shape[shard_dim] == local_shape[shard_dim] * fsdp_world_size`, and all
other dimensions must match.

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
        }
    ],
    lr=1e-3,
)
```

For non-sharded local parameters, use the optimizers like regular
`torch.optim.Optimizer` subclasses.

## Notes

- This is now an FSDP-Canzona path, not a Megatron TP path.
- Checkpoint remapping for optimizer states across different FSDP world sizes is
  not implemented yet.
- CUDA graph capture is disabled for the FSDP communication path.
