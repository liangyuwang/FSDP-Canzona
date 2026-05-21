# Adding A Matrix-Based Optimizer

Add a new optimizer by subclassing `BaseOptim` and implementing two methods:

```python
from .base_optim import BaseOptim


class MyOptim(BaseOptim):
    def __init__(self, params, lr=1e-3, split_params=False, split_shape_map=None, **kwargs):
        defaults = {"lr": lr}
        super().__init__(
            params,
            defaults,
            split_params=split_params,
            split_shape_map=split_shape_map,
            **kwargs,
        )

    def _inner_single_param_step(self, name, p, grad, group):
        state = self.state[p]
        # grad is full-matrix shaped for FSDP-sharded params.
        return update

    def _single_param_update(self, p, u, group):
        # u is the local shard update for FSDP-sharded params.
        p.data.add_(u.reshape_as(p.data), alpha=-group["lr"])
```

`BaseOptim` handles normal local parameters, FSDP-sharded parameters, and
optional split-then-gather behavior through `GradAndStateSplitter`.

For FSDP-Canzona, pass the same keyword arguments supported by Muon/SOAP:

- `fsdp_sharded`
- `fsdp_group`
- `fsdp_balance`
- `fsdp_fused_comm`
- `fsdp_max_numel_per_slot`
- `fsdp_balance_cost`

If the optimizer needs parameter selection helpers, add them to `utils.py` and
export them from `matrix_based_optimizer/__init__.py`.
