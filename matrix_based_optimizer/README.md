# Matrix-Based Optimizer

This package contains the FSDP-Canzona optimizer pieces:

- `optimizers/base_optim.py`: shared optimizer orchestration.
- `optimizers/muon.py`: Muon full-matrix update.
- `optimizers/soap.py`: SOAP full-matrix update.
- `load_balanced_fsdp_executor.py`: FSDP shard gather/compute/scatter/update.
- `split_grad_and_state.py`: optional matrix splitting for QKV/FC1/in-proj
  weights.
- `utils.py`: tagging helpers and cost models.

## FSDP-Canzona Flow

For a micro-group of matrix gradients:

1. Every rank starts with local FSDP shards.
2. The executor chooses host ranks for full-matrix optimizer work.
3. It gathers shards for each hosted parameter to the host rank.
4. The host rank runs Muon/SOAP on the full matrix.
5. The executor scatters update shards back to the original FSDP ranks.
6. Each rank updates its local shard.

This mirrors the Megatron-Canzona TP algorithm, but the process group and shard
metadata now come from the FSDP data-parallel shard layout.

## Param-Group Fields

Set these fields for sharded matrix params:

- `is_fsdp_sharded`: `True`.
- `fsdp_group`: process group for the shard ranks. Defaults to world group.
- `fsdp_full_shapes`: full matrix shape per local shard parameter.
- `fsdp_local_shapes`: local shard shape per parameter. Defaults to `p.shape`.
- `fsdp_shard_dims`: sharded dimension per parameter. Defaults to `0`.
- `fsdp_balance`: one of `no`, `single`, `slot`, or `global`.
- `fsdp_fused_comm`: `True` to use fused `all_to_all_single`.
- `fsdp_max_numel_per_slot`: optional slot capacity for `slot`/`global`
  scheduling.
- `fsdp_balance_cost`: `numel` or `flops`.
- `fsdp_overlap`: `none` for the serial path, or `full` for the pipelined path
  that overlaps gather for the next micro-group, compute for the current
  micro-group, and scatter/update for the previous micro-group.

Parameter attributes with the same meaning are also supported:
`fsdp_full_shape`, `fsdp_local_shape`, and `fsdp_shard_dim`.

## Limitations

- Uniform per-parameter sharding is required.
- Optimizer states are kept on the host rank for the full matrix tasks assigned
  to that rank.
- `fsdp_overlap="full"` currently requires `fsdp_fused_comm=True`.
- Cross-world-size distributed checkpoint remapping is still future work.
