import torch
import math
import os

from .base_optim import BaseOptim

import logging
logger = logging.getLogger(__name__)
# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "aol_nvidia": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
    # Qwen-tuned Polar Express (l0=1e-3, cushion=0, safety=1.01)
    "qwen_express": [
        (8.3865, -24.3696, 17.7251),
        (4.1414, -3.0173, 0.5524),
        (3.9226, -2.8672, 0.5357),
        (3.2540, -2.3922, 0.4827),
        (2.2512, -1.5963, 0.3960),
        (1.8700, -1.2279, 0.3582),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
}

# @torch.compile
def zeropower_via_newtonschulz5(G, steps=5, ns_coefficient_type="simple", ns_norm_eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Coeff semantics:
      - Take coeffs in order.
      - If steps > len(coeffs), repeat the last coeff for the remaining steps.
    """
    coeffs = _COEFFICIENT_SETS[ns_coefficient_type]

    transpose = G.size(0) > G.size(1)
    if transpose:
        G = G.T
    # Ensure spectral norm is at most 1
    X = torch.nn.functional.normalize(G, p=2, dim=(-2, -1), eps=ns_norm_eps)
    X = X.to(torch.bfloat16)
    # Perform the NS iterations
    L = len(coeffs)
    for i in range(steps):
        a, b, c = coeffs[i] if i < L else coeffs[-1]
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if transpose:
        X = X.T
    return X


class Muon(BaseOptim):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Usage:
        # params must be a list of param-groups; each group dict must include 'use_muon': bool
        opt = Muon(
            params=[{"params": muon_params, "use_muon": True},
                    {"params": adamw_params, "use_muon": False}],
            lr=1e-3, weight_decay=0.1, ...
        )
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_coefficient_type="simple",
        ns_norm_eps=1e-7,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        split_muon_params=False,
        split_muon_shape_map=None,
        fsdp_sharded=False,
        fsdp_group=None,
        fsdp_balance="global",
        fsdp_fused_comm=True,
        fsdp_max_numel_per_slot=None,
        fsdp_balance_cost="numel",
        fsdp_overlap="none",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficient_type=ns_coefficient_type,
            ns_norm_eps=ns_norm_eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        # Let base class normalize groups and fill defaults
        super().__init__(
            params,
            defaults,
            split_muon_params,
            split_muon_shape_map,
            fsdp_sharded=fsdp_sharded,
            fsdp_group=fsdp_group,
            fsdp_balance=fsdp_balance,
            fsdp_fused_comm=fsdp_fused_comm,
            fsdp_max_numel_per_slot=fsdp_max_numel_per_slot,
            fsdp_balance_cost=fsdp_balance_cost,
            fsdp_overlap=fsdp_overlap,
        )

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def get_muon_scale_factor(self, param_shape, fsdp_world_size=1):
        A, B = param_shape[:2]
        A = A * fsdp_world_size
        scale_factor = 0.2 * math.sqrt(max(A, B))
        return scale_factor

    def _single_param_update(self, p, u, group):
        weight_decay = group["weight_decay"]
        lr = group["lr"]
        p.data.mul_(1 - lr * weight_decay)
        p.data.add_(u.reshape_as(p.data), alpha=-lr)
        u = None

    def _inner_single_param_step(self, name, p, grad, group):
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        ns_coefficient_type = group["ns_coefficient_type"]
        ns_norm_eps = group["ns_norm_eps"]

        state = self.state[p]
        buf = state.setdefault(f"{name}momentum_buffer", torch.zeros_like(grad))
        buf = buf.view(grad.shape).lerp_(grad, 1 - momentum)
        if nesterov:
            grad = grad.lerp(buf, momentum)
        else:
            grad = buf

        u = zeropower_via_newtonschulz5(grad, steps=ns_steps,ns_coefficient_type=ns_coefficient_type, ns_norm_eps=ns_norm_eps)
        fsdp_world_size = group.get("fsdp_world_size", 1)
        scale_factor = self.get_muon_scale_factor(grad.shape, fsdp_world_size=fsdp_world_size)
        u = u * scale_factor
        return u
