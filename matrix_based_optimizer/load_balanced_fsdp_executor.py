import functools
import heapq
import operator
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist

from .utils import get_optim_flops_from_param


@dataclass(frozen=True)
class FSDPShardSpec:
    """Metadata needed to rebuild a full matrix from per-rank shards."""

    local_shape: torch.Size
    full_shape: torch.Size
    shard_dim: int = 0


@dataclass
class MicroGroupContext:
    idx: int
    micro_param_group: object
    gather_work: object = None
    gather_send_buffer: object = None
    gather_recv_buffer: object = None
    gather_recv_split_sizes: object = None
    gather_local_numels: object = None
    full_tensors_group: object = None
    full_updates_group: object = None
    scatter_work: object = None
    scatter_flag_work: object = None
    scatter_send_buffer: object = None
    scatter_recv_buffer: object = None
    scatter_recv_split_sizes: object = None
    scatter_flag_send_buffer: object = None
    scatter_flag_recv_buffer: object = None
    scatter_flag_recv_split_sizes: object = None
    shard_updates_group: object = None


def get_numel_from_shape(shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, shape, 1)


def _get_group_rank(group) -> int:
    if group is None:
        return dist.get_rank()
    try:
        return dist.get_rank(group)
    except TypeError:
        return group.rank()


def _get_group_world_size(group) -> int:
    if group is None:
        return dist.get_world_size()
    try:
        return dist.get_world_size(group)
    except TypeError:
        return group.size()


def _get_global_rank(group, group_rank: int) -> int:
    if group is None:
        return group_rank
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(group, group_rank)
    return group_rank


def _normalize_shape(shape) -> torch.Size:
    return shape if isinstance(shape, torch.Size) else torch.Size(shape)


def _is_shape_like(value) -> bool:
    if isinstance(value, torch.Size):
        return True
    if isinstance(value, (list, tuple)) and all(isinstance(dim, int) for dim in value):
        return True
    return False


def _select_shape(value, index: int):
    if value is None:
        return None
    if _is_shape_like(value):
        return value
    return value[index]


def get_fsdp_shard_spec(group: dict, param, index: int) -> FSDPShardSpec:
    full_shapes = group.get("fsdp_full_shapes", group.get("origin_shape"))
    local_shapes = group.get("fsdp_local_shapes", group.get("local_shape"))
    shard_dims = group.get("fsdp_shard_dims", group.get("fsdp_shard_dim"))

    full_shape = _select_shape(full_shapes, index)
    full_shape = getattr(param, "fsdp_full_shape", full_shape)
    full_shape = getattr(param, "_fsdp_full_shape", full_shape)
    if full_shape is None:
        raise ValueError(
            "FSDP-sharded matrix parameters need full-shape metadata. "
            "Set param.fsdp_full_shape or pass group['fsdp_full_shapes'] / group['origin_shape']."
        )

    local_shape = _select_shape(local_shapes, index)
    local_shape = getattr(param, "fsdp_local_shape", local_shape)
    local_shape = getattr(param, "_fsdp_local_shape", local_shape)
    if local_shape is None:
        local_shape = param.shape

    shard_dim = None
    if shard_dims is not None:
        shard_dim = shard_dims[index] if isinstance(shard_dims, (list, tuple)) else shard_dims
    shard_dim = getattr(param, "fsdp_shard_dim", shard_dim)
    shard_dim = getattr(param, "_fsdp_shard_dim", shard_dim)
    shard_dim = getattr(param, "partition_dim", shard_dim)
    if shard_dim is None:
        shard_dim = 0

    full_shape = _normalize_shape(full_shape)
    local_shape = _normalize_shape(local_shape)
    if shard_dim < 0:
        shard_dim += len(full_shape)
    if len(full_shape) != len(local_shape):
        raise ValueError(
            f"FSDP shard local shape {tuple(local_shape)} and full shape "
            f"{tuple(full_shape)} must have the same rank."
        )
    for dim, (local_dim, full_dim) in enumerate(zip(local_shape, full_shape)):
        if dim == shard_dim:
            continue
        if local_dim != full_dim:
            raise ValueError(
                f"FSDP shard local shape {tuple(local_shape)} and full shape "
                f"{tuple(full_shape)} only differ along shard_dim={shard_dim}."
            )
    if dist.is_available() and dist.is_initialized():
        pg = group.get("fsdp_group", group.get("process_group"))
        world_size = _get_group_world_size(pg)
        if full_shape[shard_dim] != local_shape[shard_dim] * world_size:
            raise ValueError(
                "FSDP-Canzona currently expects uniform per-parameter sharding: "
                f"full_shape[{shard_dim}]={full_shape[shard_dim]} must equal "
                f"local_shape[{shard_dim}]={local_shape[shard_dim]} * world_size={world_size}."
            )
    return FSDPShardSpec(local_shape=local_shape, full_shape=full_shape, shard_dim=shard_dim)


def is_group_fsdp_sharded(group: dict) -> bool:
    if not group.get("is_fsdp_sharded", group.get("fsdp_sharded", False)):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    pg = group.get("fsdp_group", group.get("process_group"))
    return _get_group_world_size(pg) > 1


class FSDPShardExecutor:
    """Canzona gather-compute-scatter-update pipeline for FSDP-style shards.

    The executor assumes each optimized parameter is uniformly sharded along one
    tensor dimension across the FSDP process group. It intentionally does not
    depend on PyTorch FSDP private attributes; callers provide metadata through
    parameter attributes or param-group fields.
    """

    def __init__(
        self,
        group=None,
        balance: str = "global",
        fused_comm: bool = True,
        max_numel_per_slot: Optional[int] = None,
        cost: str = "numel",
        optimizer: str = "muon",
        soap_precondition_frequency: int = 10,
        soap_max_precond_dim: int = 10000,
        overlap: str = "none",
    ):
        self.group = group
        self.group_rank = _get_group_rank(group) if dist.is_available() and dist.is_initialized() else 0
        self.global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = _get_group_world_size(group) if dist.is_available() and dist.is_initialized() else 1
        self.balance = balance
        self.fused_comm = fused_comm
        self.max_numel_per_slot = max_numel_per_slot or int(os.environ.get("FSDP_CANZONA_FUSE_SPACE_MB", 400)) * 1024 * 1024
        self.cost_name = cost
        self.optimizer = optimizer
        self.soap_precondition_frequency = soap_precondition_frequency
        self.soap_max_precond_dim = soap_max_precond_dim
        self.overlap = overlap

    @classmethod
    def from_param_group(cls, group: dict) -> "FSDPShardExecutor":
        pg = group.get("fsdp_group", group.get("process_group"))
        optimizer_name = group.get("optimizer")
        if optimizer_name is None:
            optimizer_name = "soap" if group.get("use_soap", False) else "muon"
        return cls(
            group=pg,
            balance=group.get("fsdp_balance", "global"),
            fused_comm=group.get("fsdp_fused_comm", True),
            max_numel_per_slot=group.get("fsdp_max_numel_per_slot"),
            cost=group.get("fsdp_balance_cost", group.get("balance_cost", "numel")),
            optimizer=optimizer_name,
            soap_precondition_frequency=group.get("precondition_frequency", 10),
            soap_max_precond_dim=group.get("max_precond_dim", 10000),
            overlap=group.get("fsdp_overlap", "none"),
        )

    def execute(self, param_group, param_step_fn: Callable, param_update_fn: Callable, *args, **kwargs):
        specs = [
            get_fsdp_shard_spec(param_group, p, idx)
            for idx, p in enumerate(param_group["params"])
        ]
        micro_param_groups = self.get_micro_param_groups(
            self.balance,
            param_group["params"],
            specs,
        )
        if self.overlap == "full":
            return self.execute_full_overlap(
                micro_param_groups,
                param_group,
                param_step_fn,
                param_update_fn,
                *args,
                **kwargs,
            )
        if self.overlap != "none":
            raise ValueError(f"Unknown FSDP-Canzona overlap mode: {self.overlap}")
        for micro_param_group in micro_param_groups:
            full_tensors_group = self.gather(micro_param_group)
            full_updates_group = self.compute(
                micro_param_group,
                full_tensors_group,
                param_step_fn,
                *args,
                group=param_group,
                **kwargs,
            )
            shard_updates_group = self.scatter(micro_param_group, full_updates_group)
            self.update(micro_param_group, shard_updates_group, param_update_fn, *args, group=param_group, **kwargs)

    def execute_full_overlap(self, micro_param_groups, param_group, param_step_fn, param_update_fn, *args, **kwargs):
        if not self.fused_comm:
            raise NotImplementedError("fsdp_overlap='full' currently requires fsdp_fused_comm=True.")
        if not micro_param_groups:
            return

        contexts = [
            MicroGroupContext(idx=idx, micro_param_group=micro_param_group)
            for idx, micro_param_group in enumerate(micro_param_groups)
        ]

        self.launch_gather(contexts[0])
        for idx, ctx in enumerate(contexts):
            self.finish_gather(ctx)
            if idx + 1 < len(contexts):
                self.launch_gather(contexts[idx + 1])

            ctx.full_updates_group = self.compute(
                ctx.micro_param_group,
                ctx.full_tensors_group,
                param_step_fn,
                *args,
                group=param_group,
                **kwargs,
            )
            self.launch_scatter(ctx)

            if idx - 1 >= 0:
                prev = contexts[idx - 1]
                self.finish_scatter(prev)
                self.update(prev.micro_param_group, prev.shard_updates_group, param_update_fn, *args, group=param_group, **kwargs)

        last = contexts[-1]
        self.finish_scatter(last)
        self.update(last.micro_param_group, last.shard_updates_group, param_update_fn, *args, group=param_group, **kwargs)

    def launch_gather(self, ctx: MicroGroupContext):
        micro_param_group = ctx.micro_param_group
        ctx.full_tensors_group = [[] for _ in range(self.world_size)]
        ref = self._first_grad_or_param(micro_param_group)
        if ref is None:
            return

        send_tensors = []
        send_split_sizes = []
        for _, slot in micro_param_group:
            grads_for_host = [
                p.grad.contiguous().view(spec.local_shape).reshape(-1)
                for p, spec in slot
                if p.grad is not None
            ]
            if grads_for_host:
                tensor = torch.cat(grads_for_host).contiguous()
            else:
                tensor = torch.empty(0, dtype=ref.dtype, device=ref.device)
            send_tensors.append(tensor)
            send_split_sizes.append(tensor.numel())

        ctx.gather_send_buffer = torch.cat(send_tensors).contiguous() if send_tensors else torch.empty(0, dtype=ref.dtype, device=ref.device)
        _, my_slot = micro_param_group[self.group_rank]
        ctx.gather_local_numels = [get_numel_from_shape(spec.local_shape) for _, spec in my_slot]
        recv_per_rank = sum(ctx.gather_local_numels)
        ctx.gather_recv_split_sizes = [recv_per_rank] * self.world_size
        ctx.gather_recv_buffer = torch.empty(
            sum(ctx.gather_recv_split_sizes),
            dtype=ctx.gather_send_buffer.dtype,
            device=ctx.gather_send_buffer.device,
        )
        ctx.gather_work = dist.all_to_all_single(
            ctx.gather_recv_buffer,
            ctx.gather_send_buffer,
            output_split_sizes=ctx.gather_recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.group,
            async_op=True,
        )

    def finish_gather(self, ctx: MicroGroupContext):
        if ctx.gather_work is None:
            return ctx.full_tensors_group
        ctx.gather_work.wait()
        my_slot = ctx.micro_param_group[self.group_rank][1]
        recv_streams = torch.split(ctx.gather_recv_buffer, ctx.gather_recv_split_sizes)
        stream_offset = 0
        for (p, spec), shard_numel in zip(my_slot, ctx.gather_local_numels):
            shards = []
            for src_rank in range(self.world_size):
                shard = recv_streams[src_rank][stream_offset : stream_offset + shard_numel]
                shards.append(shard.contiguous().view(spec.local_shape))
            ctx.full_tensors_group[self.group_rank].append(torch.cat(shards, dim=spec.shard_dim).view(spec.full_shape))
            stream_offset += shard_numel
        return ctx.full_tensors_group

    def launch_scatter(self, ctx: MicroGroupContext):
        micro_param_group = ctx.micro_param_group
        full_updates_group = ctx.full_updates_group
        ctx.shard_updates_group = [[] for _ in range(self.world_size)]
        ref = self._first_grad_or_param(micro_param_group)
        if ref is None:
            return

        send_streams = [[] for _ in range(self.world_size)]
        send_flag_streams = [[] for _ in range(self.world_size)]
        for host_group_rank, slot in micro_param_group:
            if host_group_rank != self.group_rank:
                continue
            for (p, spec), full_update in zip(slot, full_updates_group[host_group_rank]):
                skip_update = full_update is None
                if full_update is None:
                    full_update = torch.zeros(spec.full_shape, dtype=ref.dtype, device=ref.device)
                shards = torch.chunk(full_update.contiguous().view(spec.full_shape), self.world_size, dim=spec.shard_dim)
                for target_rank, shard in enumerate(shards):
                    send_streams[target_rank].append(shard.contiguous().view(-1))
                    send_flag_streams[target_rank].append(
                        torch.tensor([1 if skip_update else 0], dtype=torch.uint8, device=ref.device)
                    )

        flat_send_tensors = []
        send_split_sizes = []
        for stream in send_streams:
            if stream:
                tensor = torch.cat(stream).contiguous()
            else:
                tensor = torch.empty(0, dtype=ref.dtype, device=ref.device)
            flat_send_tensors.append(tensor)
            send_split_sizes.append(tensor.numel())

        ctx.scatter_send_buffer = torch.cat(flat_send_tensors).contiguous() if flat_send_tensors else torch.empty(0, dtype=ref.dtype, device=ref.device)
        ctx.scatter_recv_split_sizes = [
            sum(get_numel_from_shape(spec.local_shape) for _, spec in slot)
            for _, slot in micro_param_group
        ]
        ctx.scatter_recv_buffer = torch.empty(
            sum(ctx.scatter_recv_split_sizes),
            dtype=ctx.scatter_send_buffer.dtype,
            device=ctx.scatter_send_buffer.device,
        )
        ctx.scatter_work = dist.all_to_all_single(
            ctx.scatter_recv_buffer,
            ctx.scatter_send_buffer,
            output_split_sizes=ctx.scatter_recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.group,
            async_op=True,
        )

        flat_send_flags = []
        flag_send_split_sizes = []
        for stream in send_flag_streams:
            if stream:
                tensor = torch.cat(stream).contiguous()
            else:
                tensor = torch.empty(0, dtype=torch.uint8, device=ref.device)
            flat_send_flags.append(tensor)
            flag_send_split_sizes.append(tensor.numel())
        ctx.scatter_flag_send_buffer = torch.cat(flat_send_flags).contiguous() if flat_send_flags else torch.empty(0, dtype=torch.uint8, device=ref.device)
        ctx.scatter_flag_recv_split_sizes = [len(slot) for _, slot in micro_param_group]
        ctx.scatter_flag_recv_buffer = torch.empty(sum(ctx.scatter_flag_recv_split_sizes), dtype=torch.uint8, device=ref.device)
        ctx.scatter_flag_work = dist.all_to_all_single(
            ctx.scatter_flag_recv_buffer,
            ctx.scatter_flag_send_buffer,
            output_split_sizes=ctx.scatter_flag_recv_split_sizes,
            input_split_sizes=flag_send_split_sizes,
            group=self.group,
            async_op=True,
        )

    def finish_scatter(self, ctx: MicroGroupContext):
        if ctx.scatter_work is None:
            return ctx.shard_updates_group
        ctx.scatter_work.wait()
        ctx.scatter_flag_work.wait()
        recv_streams = torch.split(ctx.scatter_recv_buffer, ctx.scatter_recv_split_sizes)
        flag_recv_streams = torch.split(ctx.scatter_flag_recv_buffer, ctx.scatter_flag_recv_split_sizes)
        for src_group_rank, (_, slot) in enumerate(ctx.micro_param_group):
            src_stream = recv_streams[src_group_rank]
            flag_stream = flag_recv_streams[src_group_rank]
            offset = 0
            for flag_idx, (_, spec) in enumerate(slot):
                shard_numel = get_numel_from_shape(spec.local_shape)
                skip_update = bool(flag_stream[flag_idx].item())
                if skip_update:
                    ctx.shard_updates_group[src_group_rank].append(None)
                else:
                    ctx.shard_updates_group[src_group_rank].append(
                        src_stream[offset : offset + shard_numel].contiguous().view(spec.local_shape)
                    )
                offset += shard_numel
        return ctx.shard_updates_group

    def gather(self, micro_param_group):
        if self.fused_comm:
            return self._gather_fused(micro_param_group)
        return self._gather_unfused(micro_param_group)

    def _gather_fused(self, micro_param_group):
        full_tensors_group = [[] for _ in range(self.world_size)]
        ref = self._first_grad_or_param(micro_param_group)
        if ref is None:
            return full_tensors_group

        send_tensors = []
        send_split_sizes = []
        for _, slot in micro_param_group:
            grads_for_host = [
                p.grad.contiguous().view(spec.local_shape).reshape(-1)
                for p, spec in slot
                if p.grad is not None
            ]
            if grads_for_host:
                tensor = torch.cat(grads_for_host).contiguous()
            else:
                tensor = torch.empty(0, dtype=ref.dtype, device=ref.device)
            send_tensors.append(tensor)
            send_split_sizes.append(tensor.numel())

        send_buffer = torch.cat(send_tensors).contiguous() if send_tensors else torch.empty(0, dtype=ref.dtype, device=ref.device)
        _, my_slot = micro_param_group[self.group_rank]
        local_numels = [get_numel_from_shape(spec.local_shape) for _, spec in my_slot]
        recv_per_rank = sum(local_numels)
        recv_split_sizes = [recv_per_rank] * self.world_size
        recv_buffer = torch.empty(sum(recv_split_sizes), dtype=send_buffer.dtype, device=send_buffer.device)

        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.group,
        )

        recv_streams = torch.split(recv_buffer, recv_split_sizes)
        stream_offset = 0
        for p, spec in my_slot:
            shard_numel = get_numel_from_shape(spec.local_shape)
            shards = []
            for src_rank in range(self.world_size):
                shard = recv_streams[src_rank][stream_offset : stream_offset + shard_numel]
                shards.append(shard.contiguous().view(spec.local_shape))
            full_tensors_group[self.group_rank].append(torch.cat(shards, dim=spec.shard_dim).view(spec.full_shape))
            stream_offset += shard_numel
        return full_tensors_group

    def _gather_unfused(self, micro_param_group):
        full_tensors_group = [[] for _ in range(self.world_size)]
        for host_group_rank, slot in micro_param_group:
            for p, spec in slot:
                if p.grad is None:
                    full_tensors_group[host_group_rank].append(None)
                    continue
                local_grad = p.grad.contiguous().view(spec.local_shape)
                shards = [torch.empty_like(local_grad) for _ in range(self.world_size)]
                dist.all_gather(shards, local_grad, group=self.group)
                if host_group_rank == self.group_rank:
                    full_tensors_group[host_group_rank].append(torch.cat(shards, dim=spec.shard_dim).view(spec.full_shape))
                else:
                    full_tensors_group[host_group_rank].append(None)
        return full_tensors_group

    def compute(self, micro_param_group, full_tensors_group, param_step_fn, *args, **kwargs):
        full_updates_group = [[] for _ in range(self.world_size)]
        for host_group_rank, slot in micro_param_group:
            full_tensors_slot = full_tensors_group[host_group_rank]
            for (p, spec), full_grad in zip(slot, full_tensors_slot):
                if host_group_rank == self.group_rank and full_grad is not None:
                    full_updates_group[host_group_rank].append(
                        param_step_fn(p, spec.full_shape, *args, g=full_grad, **kwargs)
                    )
                else:
                    full_updates_group[host_group_rank].append(None)
        return full_updates_group

    def scatter(self, micro_param_group, full_updates_group):
        if self.fused_comm:
            return self._scatter_fused(micro_param_group, full_updates_group)
        return self._scatter_unfused(micro_param_group, full_updates_group)

    def _scatter_fused(self, micro_param_group, full_updates_group):
        shard_updates_group = [[] for _ in range(self.world_size)]
        ref = self._first_grad_or_param(micro_param_group)
        if ref is None:
            return shard_updates_group

        send_streams = [[] for _ in range(self.world_size)]
        send_flag_streams = [[] for _ in range(self.world_size)]
        for host_group_rank, slot in micro_param_group:
            if host_group_rank != self.group_rank:
                continue
            for (p, spec), full_update in zip(slot, full_updates_group[host_group_rank]):
                skip_update = full_update is None
                if full_update is None:
                    full_update = torch.zeros(spec.full_shape, dtype=ref.dtype, device=ref.device)
                shards = torch.chunk(full_update.contiguous().view(spec.full_shape), self.world_size, dim=spec.shard_dim)
                for target_rank, shard in enumerate(shards):
                    send_streams[target_rank].append(shard.contiguous().view(-1))
                    send_flag_streams[target_rank].append(
                        torch.tensor([1 if skip_update else 0], dtype=torch.uint8, device=ref.device)
                    )

        flat_send_tensors = []
        send_split_sizes = []
        for stream in send_streams:
            if stream:
                tensor = torch.cat(stream).contiguous()
            else:
                tensor = torch.empty(0, dtype=ref.dtype, device=ref.device)
            flat_send_tensors.append(tensor)
            send_split_sizes.append(tensor.numel())

        send_buffer = torch.cat(flat_send_tensors).contiguous() if flat_send_tensors else torch.empty(0, dtype=ref.dtype, device=ref.device)
        recv_split_sizes = [
            sum(get_numel_from_shape(spec.local_shape) for _, spec in slot)
            for _, slot in micro_param_group
        ]
        recv_buffer = torch.empty(sum(recv_split_sizes), dtype=send_buffer.dtype, device=send_buffer.device)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.group,
        )

        flat_send_flags = []
        flag_send_split_sizes = []
        for stream in send_flag_streams:
            if stream:
                tensor = torch.cat(stream).contiguous()
            else:
                tensor = torch.empty(0, dtype=torch.uint8, device=ref.device)
            flat_send_flags.append(tensor)
            flag_send_split_sizes.append(tensor.numel())
        flag_send_buffer = torch.cat(flat_send_flags).contiguous() if flat_send_flags else torch.empty(0, dtype=torch.uint8, device=ref.device)
        flag_recv_split_sizes = [len(slot) for _, slot in micro_param_group]
        flag_recv_buffer = torch.empty(sum(flag_recv_split_sizes), dtype=torch.uint8, device=ref.device)
        dist.all_to_all_single(
            flag_recv_buffer,
            flag_send_buffer,
            output_split_sizes=flag_recv_split_sizes,
            input_split_sizes=flag_send_split_sizes,
            group=self.group,
        )

        recv_streams = torch.split(recv_buffer, recv_split_sizes)
        flag_recv_streams = torch.split(flag_recv_buffer, flag_recv_split_sizes)
        for src_group_rank, (_, slot) in enumerate(micro_param_group):
            src_stream = recv_streams[src_group_rank]
            flag_stream = flag_recv_streams[src_group_rank]
            offset = 0
            for flag_idx, (_, spec) in enumerate(slot):
                shard_numel = get_numel_from_shape(spec.local_shape)
                skip_update = bool(flag_stream[flag_idx].item())
                if skip_update:
                    shard_updates_group[src_group_rank].append(None)
                else:
                    shard_updates_group[src_group_rank].append(
                        src_stream[offset : offset + shard_numel].contiguous().view(spec.local_shape)
                    )
                offset += shard_numel
        return shard_updates_group

    def _scatter_unfused(self, micro_param_group, full_updates_group):
        shard_updates_group = [[] for _ in range(self.world_size)]
        for host_group_rank, slot in micro_param_group:
            host_global_rank = _get_global_rank(self.group, host_group_rank)
            for (p, spec), full_update in zip(slot, full_updates_group[host_group_rank]):
                local_update = torch.empty(spec.local_shape, dtype=p.dtype, device=p.device)
                skip_flag = torch.zeros(1, dtype=torch.uint8, device=p.device)
                if host_group_rank == self.group_rank:
                    if full_update is None:
                        skip_flag.fill_(1)
                        full_update = torch.zeros(spec.full_shape, dtype=p.dtype, device=p.device)
                    shards = [
                        shard.contiguous()
                        for shard in torch.chunk(full_update.view(spec.full_shape), self.world_size, dim=spec.shard_dim)
                    ]
                else:
                    shards = None
                dist.scatter(local_update, shards, src=host_global_rank, group=self.group)
                dist.broadcast(skip_flag, src=host_global_rank, group=self.group)
                shard_updates_group[host_group_rank].append(None if bool(skip_flag.item()) else local_update)
        return shard_updates_group

    def update(self, micro_param_group, shard_updates_group, param_update_fn, *args, **kwargs):
        for (_, slot), shard_updates_slot in zip(micro_param_group, shard_updates_group):
            for (p, _), shard_update in zip(slot, shard_updates_slot):
                if shard_update is not None:
                    param_update_fn(p, shard_update, *args, **kwargs)

    def get_micro_param_groups(self, balance: str, params: Sequence, specs: Sequence[FSDPShardSpec]):
        if balance == "no":
            slots_groups = self._simple_groups(params, specs)
        elif balance == "single":
            slots_groups = self._single_balanced_groups(params, specs)
        elif balance == "slot":
            slots_groups = self._slot_balanced_groups(params, specs)
        elif balance == "global":
            slots_groups = self._globally_balanced_groups(params, specs)
        else:
            raise ValueError(f"Unknown FSDP Canzona balance mode: {balance}")
        return [list(enumerate(slots)) for slots in slots_groups]

    def _simple_groups(self, params, specs):
        items = [(self.cost_fn(p, spec), p, spec) for p, spec in zip(params, specs) if p.grad is not None]
        groups = []
        for start in range(0, len(items), self.world_size):
            slots = [[] for _ in range(self.world_size)]
            for rank, (_, p, spec) in enumerate(items[start : start + self.world_size]):
                slots[rank].append((p, spec))
            groups.append(slots)
        return groups

    def _single_balanced_groups(self, params, specs):
        items = [(self.cost_fn(p, spec), p, spec) for p, spec in zip(params, specs) if p.grad is not None]
        items.sort(key=lambda item: item[0], reverse=True)
        groups = []
        for start in range(0, len(items), self.world_size):
            slots = [[] for _ in range(self.world_size)]
            for rank, (_, p, spec) in enumerate(items[start : start + self.world_size]):
                slots[rank].append((p, spec))
            groups.append(slots)
        return groups

    def _slot_balanced_groups(self, params, specs):
        items = [(self.cost_fn(p, spec), get_numel_from_shape(spec.local_shape), p, spec) for p, spec in zip(params, specs) if p.grad is not None]
        items.sort(key=lambda item: item[0], reverse=True)
        groups = []
        slots = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        heap = [(0, rank) for rank in range(self.world_size)]
        heapq.heapify(heap)
        for cost, comm, p, spec in items:
            load, rank = heapq.heappop(heap)
            slots[rank].append((p, spec))
            loads[rank] = load + cost
            heapq.heappush(heap, (loads[rank], rank))
            if max(loads) >= self.max_numel_per_slot:
                groups.append(slots)
                slots = [[] for _ in range(self.world_size)]
                loads = [0] * self.world_size
                heap = [(0, rank) for rank in range(self.world_size)]
                heapq.heapify(heap)
        if any(slots):
            groups.append(slots)
        return groups

    def _globally_balanced_groups(self, params, specs):
        items = [(self.cost_fn(p, spec), idx, p, spec) for idx, (p, spec) in enumerate(zip(params, specs)) if p.grad is not None]
        items.sort(key=lambda item: (item[0], item[1]), reverse=True)
        groups = []
        current = []
        current_sum = 0
        for item in items:
            cost = item[0]
            candidate = current + [item]
            candidate_sum = current_sum + cost
            allocation, max_load = self._partition(candidate)
            if current and (candidate_sum / self.world_size > self.max_numel_per_slot or max_load > self.max_numel_per_slot):
                groups.append(self._partition(current)[0])
                current = [item]
                current_sum = cost
            else:
                current = candidate
                current_sum = candidate_sum
        if current:
            groups.append(self._partition(current)[0])
        return groups

    def _partition(self, items):
        slots = [[] for _ in range(self.world_size)]
        loads = [0] * self.world_size
        heap = [(0, rank) for rank in range(self.world_size)]
        heapq.heapify(heap)
        for cost, _idx, p, spec in sorted(items, key=lambda item: item[0], reverse=True):
            load, rank = heapq.heappop(heap)
            slots[rank].append((p, spec))
            loads[rank] = load + cost
            heapq.heappush(heap, (loads[rank], rank))
        return slots, max(loads) if loads else 0

    def cost_fn(self, param, spec: FSDPShardSpec):
        if self.cost_name == "numel":
            return get_numel_from_shape(spec.full_shape)
        if self.cost_name == "flops":
            return get_optim_flops_from_param(
                param,
                optimizer=self.optimizer,
                soap_precondition_frequency=self.soap_precondition_frequency,
                soap_max_precond_dim=self.soap_max_precond_dim,
                shape=spec.full_shape,
            )
        raise ValueError(f"Unknown FSDP Canzona cost: {self.cost_name}")

    @staticmethod
    def _first_grad_or_param(micro_param_group):
        for _, slot in micro_param_group:
            if slot:
                param = slot[0][0]
                return param.grad if param.grad is not None else param
        return None
