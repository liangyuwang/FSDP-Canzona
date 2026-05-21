import torch
import os
import sys
from typing import Union
from contextlib import contextmanager


MATRIX_OPTIM_EXCLUDE_NAME_TOKENS = (
    # Generic PyTorch / HuggingFace-style embeddings and heads.
    "embed",
    "embedding",
    "lm_head",
    "classifier",
    # Common Megatron-compatible aliases kept for migrated checkpoints/models.
    "word_embeddings",
    "position_embeddings",
    "output_layer",
    # Router logits are usually small auxiliary projections, not matrix-optimizer targets.
    "router",
    "gate_weight",
)


def is_matrix_based_optim(optim_name=None):
    return optim_name in {"muon", "soap"} if optim_name is not None else False

def is_matrix_based_optim_group(param_group):
    return bool(param_group.get('use_muon', False) or param_group.get('use_soap', False))

def is_param_use_matrix_based_optim(name: str, param: torch.Tensor) -> bool:
    """Return whether a named parameter should use a matrix-based optimizer.

    This helper is intentionally framework-agnostic. It selects 2D weights and
    excludes embeddings, language-model heads, classifiers, and router logits
    across common PyTorch/HuggingFace names while retaining Megatron-compatible
    aliases for users migrating models.
    """
    if param.ndim != 2:
        return False
    name_lower = name.lower()
    return not any(token in name_lower for token in MATRIX_OPTIM_EXCLUDE_NAME_TOKENS)

def get_optim_memory_from_param(p: torch.Tensor) -> Union[int, float]:
    return p.numel()    # muon optimizer state shape is the same as p shape
                        # however, other optimizer like soap can be different

def get_optim_flops_from_param(
    p: torch.Tensor,
    optimizer: str = "muon",
    soap_precondition_frequency: int = 10,
    soap_max_precond_dim: int = 10000,
    shape=None,
) -> Union[int, float]:
    param_shape = tuple(shape if shape is not None else p.shape)
    def estimate_muon_flops(param_shape, ns_steps=5):
        if len(param_shape) < 2:
            return 0
        rows = param_shape[0]
        cols = 1
        for dim in param_shape[1:]:
            cols *= dim
        min_dim = min(rows, cols)
        max_dim = max(rows, cols)
        flops_matmul_1 = 2 * max_dim * (min_dim ** 2)
        flops_matmul_2 = 2 * max_dim * (min_dim ** 2)
        flops_per_iter = flops_matmul_1 + flops_matmul_2
        total_flops = flops_per_iter * ns_steps
        return total_flops
    def estimate_averaged_soap_flops(param_shape, update_freq=10, max_precond_dim=10000):
        assert len(param_shape) == 2
        rows, cols = param_shape
        flops_hot_path = 0
        flops_cold_path = 0
        if rows <= max_precond_dim:
            flops_hot_path += 2 * rows * rows * cols
        if cols <= max_precond_dim:
            flops_hot_path += 2 * cols * cols * rows
        if rows <= max_precond_dim:
            flops_hot_path += 2 * (2 * cols * rows * rows)
        if cols <= max_precond_dim:
            flops_hot_path += 2 * (2 * rows * cols * cols)
        if rows <= max_precond_dim:
            flops_hot_path += 2 * rows * rows * cols
        if cols <= max_precond_dim:
            flops_hot_path += 2 * cols * cols * rows
        if rows <= max_precond_dim:
            flops_cold_path += 2 * rows**3
            flops_cold_path += 2 * rows**3
            flops_cold_path += 2 * rows**3
            flops_cold_path += (4/3) * rows**3
        if cols <= max_precond_dim:
            flops_cold_path += 2 * cols**3
            flops_cold_path += 2 * cols**3
            flops_cold_path += 2 * cols**3
            flops_cold_path += (4/3) * cols**3
        avg_flops = flops_hot_path + flops_cold_path / update_freq
        return avg_flops
    if optimizer == "muon":
        return estimate_muon_flops(param_shape)
    elif optimizer == "soap":
        return estimate_averaged_soap_flops(
            param_shape,
            update_freq=soap_precondition_frequency,
            max_precond_dim=soap_max_precond_dim,
        )
    else:
        raise ValueError
