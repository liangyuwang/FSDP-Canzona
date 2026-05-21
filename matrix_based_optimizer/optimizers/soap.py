import torch
import torch.nn as nn
import torch.optim as optim
import os
from itertools import chain
from .base_optim import BaseOptim

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py

class SOAP(BaseOptim):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer.
            Helps at large precondition_frequency (~100 in our experiments),
            but hurts performance at small precondition_frequency (~10 in our experiments).
        data_format (`str`, *optional*, defaults to `channels_first`):
            Data format of the input for convolutional layers.
            Should be "channels_last" for data_format of NHWC and "channels_first" for NCHW.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.
    """
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float= -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int=10,
        max_precond_dim: int=10000, #
        merge_dims: bool = False, # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        split_soap_params=False,
        split_soap_shape_map=None,
        async_tp=False,
        fsdp_sharded=False,
        fsdp_group=None,
        fsdp_balance="global",
        fsdp_fused_comm=True,
        fsdp_max_numel_per_slot=None,
        fsdp_balance_cost="numel",
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(
            params,
            defaults,
            split_soap_params,
            split_soap_shape_map,
            async_tp,
            fsdp_sharded=fsdp_sharded,
            fsdp_group=fsdp_group,
            fsdp_balance=fsdp_balance,
            fsdp_fused_comm=fsdp_fused_comm,
            fsdp_max_numel_per_slot=fsdp_max_numel_per_slot,
            fsdp_balance_cost=fsdp_balance_cost,
        )
        self._data_format = data_format
        assert precondition_1d is False, 'only support dim=2 for now.'
        assert merge_dims is False, 'only support merge_dims=False for now.'
        if self.use_cuda_graph:
            raise NotImplementedError

    def _single_param_update(self, p, u, group, shapes_map=None):
        weight_decay = group["weight_decay"]
        lr = group["lr"]
        p.data.add_(u.reshape_as(p.data), alpha=-lr)

        # From AdamW code: Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        # Add weight decay at the end (fixed version)
        if weight_decay > 0.0:
            p.data.add_(p.data, alpha=(-lr * weight_decay))
        u = None

    def _inner_single_param_step(self, name, p, grad, group):
        shape_map = group["shapes_map"][p]
        # assert len(grad.shape) == 2, 'More scenarios need to be supported.'
        shape_map[f"{name}exp_avg"] = grad.shape
        shape_map[f"{name}exp_avg_sq"] = grad.shape
        for idx, sh in enumerate(grad.shape):
            if sh > group['max_precond_dim']:
                pass
            else:
                shape_map[f"{name}GG_{idx}"] = torch.Size((sh,sh))
                shape_map[f"{name}Q_{idx}"] = torch.Size((sh,sh))
        state = self.state[p]
        assert len(grad.shape) == 2, 'More scenarios need to be supported.'
        """
        state.keys() = [
            "exp_avg",
            "exp_avg_sq",
            "Q_*",
            "GG_*",
            "step",
        ]
        """
        shampoo_beta = group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]
        precondition_frequency = group['precondition_frequency']
        max_precond_dim = group['max_precond_dim']
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        correct_bias = group["correct_bias"]
        normalize_grads = group["normalize_grads"]

        if f"{name}step" not in state:
            state[f"{name}step"] = torch.tensor([0], device=grad.device)
        step = state[f"{name}step"].item()

        # State initialization
        if f"{name}exp_avg" not in state:
            # Exponential moving average of gradient values
            state[f"{name}exp_avg"] = torch.zeros_like(grad)
            # Exponential moving average of squared gradient values
            state[f"{name}exp_avg_sq"] = torch.zeros_like(grad)

        if not any(key.startswith(f"{name}Q_") for key in state):
            state = self.init_preconditioner(
                grad,
                state,
                False,
                max_precond_dim=max_precond_dim,
                name=name,
            )
            state = self.update_preconditioner(grad,
                                                state,
                                                shape_map,
                                                False,
                                                step,
                                                precondition_frequency=precondition_frequency,
                                                max_precond_dim=max_precond_dim,
                                                shampoo_beta=shampoo_beta,
                                                name=name)
            return # first step is skipped so that we never use the current gradients in the projection.

        # Projecting gradients to the eigenbases of Shampoo's preconditioner
        # i.e. projecting to the eigenbases of matrices in state['GG']
        grad_projected = self.project(grad, state, shape_map, name)
        exp_avg, exp_avg_sq = state[f"{name}exp_avg"].view(shape_map[f"{name}exp_avg"]), state[f"{name}exp_avg_sq"].view(shape_map[f"{name}exp_avg_sq"])
        step += 1
        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg.lerp_(grad, weight=1.0 - beta1)
        exp_avg_sq.lerp_(grad_projected.square(), weight=1.0 - beta2)
        denom = exp_avg_sq.sqrt().add_(eps)

        # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
        # i.e. projecting to the eigenbases of matrices in state['GG']
        exp_avg_projected = self.project(exp_avg, state, shape_map, name)

        scale = 1
        if correct_bias:
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            scale = (bias_correction2 ** .5) / bias_correction1
        # Projecting back the preconditioned (by Adam) exponential moving average of gradients
        # to the original space
        norm_grad = self.project_back(exp_avg_projected / denom, state, shape_map, name=name)
        if normalize_grads:
            norm_grad = norm_grad / (1e-30+torch.mean(norm_grad**2)**0.5)
        norm_grad = norm_grad * scale

        # Update is done after the gradient step to avoid using current gradients in the projection.
        state = self.update_preconditioner(grad, state, shape_map, False, step,
                                    precondition_frequency=precondition_frequency,
                                    max_precond_dim=max_precond_dim,
                                    shampoo_beta=shampoo_beta,
                                    name=name)

        # flatten state
        for key in list(state.keys()):
            state[f"{name}step"] = torch.tensor([step], device=grad.device)
        return norm_grad


    def init_preconditioner(self, grad, state, dist_optim, max_precond_dim=10000, name=""):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        for idx,sh in enumerate(grad.shape):
            if sh > max_precond_dim:
                pass
            else:
                key = f"{name}GG_{idx}"
                if dist_optim:
                    state[key] = torch.zeros(sh, sh, device=grad.device).reshape(-1)
                else:
                    state[key] = torch.zeros(sh, sh, device=grad.device)
        return state

    def project(self, grad, state, param_to_os_shape, name=""):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        Q_list = get_mat_list(state, param_to_os_shape, "Q_", name=name)
        for mat in Q_list:
            if mat is not None:
                grad = grad.T @ mat
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        return grad

    def update_preconditioner(self,
                              grad,
                              state,
                              param_to_os_shape,
                              dist_optim,
                              step,
                              precondition_frequency=10,
                              max_precond_dim=10000,
                              shampoo_beta=-1,
                              name=""):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """

        for idx, sh in enumerate(grad.shape):
            if sh <= max_precond_dim:
                permute_order = [idx] + [i for i in range(len(grad.shape)) if i != idx]
                grad_permuted = grad.permute(permute_order)
                grad_reshaped = grad_permuted.reshape(sh, -1)
                outer_product = grad_reshaped @ grad_reshaped.T
                state[f"{name}GG_{idx}"].view(param_to_os_shape[f"{name}GG_{idx}"]).lerp_(outer_product, 1-shampoo_beta)

        if not any(key.startswith(f"{name}Q_") for key in state if isinstance(key, str)):
            gg_list = get_mat_list(state, param_to_os_shape, "GG_", name=name)
            q_list = self.get_orthogonal_matrix_Q(gg_list, name=name)
            assert len(q_list) == 2, 'only support 2D for now.'
            for idx, q in enumerate(q_list):
                if len(q) > 0:
                    if dist_optim:
                        state[f"{name}Q_{idx}"] = q.reshape(-1)
                    else:
                        state[f"{name}Q_{idx}"] = q

        if step > 0 and step % precondition_frequency == 0:
            state = self.get_orthogonal_matrix_QR(state, param_to_os_shape, dist_optim, max_precond_dim, name=name)

        return state

    def project_back(self, grad, state, param_to_os_shape, name=""):
        """
        Projects the gradient back to the original space.
        """
        Q_list = get_mat_list(state, param_to_os_shape, "Q_", name=name)

        for mat in Q_list:
            if mat is not None:
                grad = (mat @ grad).T
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        return grad

    def get_orthogonal_matrix_Q(self, mat: list, name=""):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if m is None:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(m+1e-8*torch.eye(m.shape[0], device=m.device))
            except:
                _, Q = torch.linalg.eigh(m.to(torch.float64)+1e-8*torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, param_to_os_shape, dist_optim, max_precond_dim=10000, name=""):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        """
        precond_list = get_mat_list(state, param_to_os_shape, "GG_", name=name)
        orth_list = get_mat_list(state, param_to_os_shape, "Q_", name=name)

        matrix = []
        orth_matrix = []
        for m,o in zip(precond_list, orth_list):
            if m is None:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())

        exp_avg_sq = state[f"{name}exp_avg_sq"].reshape(param_to_os_shape[f"{name}exp_avg_sq"])

        final = []
        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
            if len(m)==0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:,sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)


        state[f"{name}exp_avg_sq"] = exp_avg_sq
        for idx,q in enumerate(final):
            if len(q) > 0:
                if dist_optim:
                    state[f"{name}Q_{idx}"] = q.reshape(-1)
                else:
                    state[f"{name}Q_{idx}"] = q
        return state

def get_mat_list(state, shape_map, key, name=""):
    if name == "":
        keys = [k for k in state.keys() if k.startswith(key)]
        assert key in ['GG_', 'Q_'], 'only support GG_ and Q_ for now.'
        assert all(k in [f'{key}0', f'{key}1'] for k in keys), 'only support 2D for now.'
        mat_list = [
            state[f'{key}0'].reshape(shape_map[f'{key}0']) if f'{key}0' in state else None,
            state[f'{key}1'].reshape(shape_map[f'{key}1']) if f'{key}1' in state else None
        ]
        return mat_list
    else:
        mat_list = []
        for idx in [0, 1]:
            full_key = f"{name}{key}{idx}"
            state_tensor = state.get(full_key, None)
            if state_tensor is None:
                mat_list.append(None)
                continue
            if full_key not in shape_map:
                raise KeyError(f"Key '{full_key}' missing in shape_map for tensor {id(state_tensor)}")
            mat_list.append(state_tensor.reshape(shape_map[full_key]))
        return mat_list
