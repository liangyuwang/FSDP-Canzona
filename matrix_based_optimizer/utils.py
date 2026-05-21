import torch
import os
import sys
from typing import Union
from contextlib import contextmanager


def is_matrix_based_optim(optim_name=None):
    return optim_name in {"muon", "soap"} if optim_name is not None else False

def is_matrix_based_optim_group(param_group):
    return bool(param_group.get('use_muon', False) or param_group.get('use_soap', False))

def is_param_use_matrix_based_optim(name: str, param: torch.Tensor) -> bool:
    # assert param.ndim <= 2, 'Tensors with ndim > 2 are not supported.'
    # Matrix based optimizer for exactly 2-D weights, excluding embeddings and output head.
    use_matrix_based_opt = (
        # param.ndim in [1, 2] if args.soap_precondition_1d else param.ndim == 2
        param.ndim == 2
        and 'word_embeddings' not in name
        and 'output_layer' not in name
        and 'router' not in name
        and 'gate_weight' not in name
    )
    return use_matrix_based_opt

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
