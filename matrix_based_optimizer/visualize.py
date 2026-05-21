import functools
import logging
import math
import operator
from typing import Callable, Optional, Sequence

from .utils import get_optim_flops_from_param


logger = logging.getLogger(__name__)


def _numel_from_shape(shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, shape, 1)


def _is_shape_like(value) -> bool:
    if hasattr(value, "__iter__") and all(isinstance(dim, int) for dim in value):
        return True
    return False


def _shape_from_slot_item(param, meta):
    if hasattr(meta, "full_shape"):
        return meta.full_shape
    if hasattr(meta, "shape") and _is_shape_like(meta.shape):
        return meta.shape
    if _is_shape_like(meta):
        return meta
    return param.shape


class FSDPLoadVisualizer:
    """Render load-balance reports for Canzona micro-groups.

    The expected micro-group format is the same as FSDPShardExecutor emits:
    ``[(rank, [(param, spec_or_shape), ...]), ...]``.

    ``spec_or_shape`` may be an FSDPShardSpec, a torch.Size, or any shape-like
    sequence. This keeps the visualizer useful for both the FSDP executor and
    tests that build synthetic groups.
    """

    def __init__(
        self,
        world_size: int,
        rank: int = 0,
        rank_to_log: int = 0,
        cost_fn: Optional[Callable] = None,
        optimizer: str = "muon",
        soap_precondition_frequency: int = 10,
        soap_max_precond_dim: int = 10000,
        dtype_bytes: int = 4,
        log_fn: Optional[Callable[[str], None]] = None,
        use_color: bool = False,
    ):
        self.world_size = world_size
        self.rank = rank
        self.rank_to_log = rank_to_log
        self.cost_fn = cost_fn
        self.optimizer = optimizer
        self.soap_precondition_frequency = soap_precondition_frequency
        self.soap_max_precond_dim = soap_max_precond_dim
        self.dtype_bytes = dtype_bytes
        self.log_fn = log_fn
        self.use_color = use_color
        self.colors = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "fail": "\033[91m",
            "end": "\033[0m",
        }
        self.bar_char = "#"
        self.empty_char = "."

    def _emit(self, text: str) -> None:
        if self.rank != self.rank_to_log:
            return
        sink = self.log_fn or logger.info
        for line in text.splitlines():
            sink(line)

    def _human_readable_size(self, numel: float) -> str:
        size_bytes = numel * self.dtype_bytes
        if size_bytes <= 0:
            return "0 B"
        size_names = ("B", "KB", "MB", "GB", "TB", "PB")
        idx = min(int(math.floor(math.log(size_bytes, 1024))), len(size_names) - 1)
        scaled = round(size_bytes / math.pow(1024, idx), 2)
        return f"{scaled} {size_names[idx]}"

    def _human_readable_unit(self, value: float, unit: str = "") -> str:
        if value == 0:
            return f"0 {unit}".rstrip()
        if abs(value) < 1000:
            return f"{int(value)} {unit}".rstrip()
        prefixes = ("", "K", "M", "B", "T", "P")
        idx = min(int(math.floor(math.log(abs(value), 1000))), len(prefixes) - 1)
        scaled = round(value / math.pow(1000, idx), 2)
        return f"{scaled} {prefixes[idx]}{unit}".rstrip()

    def _human_readable(self, cost_name: str, value: float) -> str:
        if cost_name == "numel":
            return self._human_readable_size(value)
        if cost_name == "flops":
            return self._human_readable_unit(value, "FLOPs")
        raise ValueError(f"Unknown visualization cost: {cost_name}")

    def _maybe_external_cost(self, param, meta, cost_name: str):
        if self.cost_fn is None:
            return None
        try:
            return self.cost_fn(param, meta, cost_name)
        except TypeError:
            pass
        try:
            return self.cost_fn(param, meta)
        except (TypeError, AttributeError, ValueError):
            pass
        try:
            return self.cost_fn(param, cost_name)
        except TypeError:
            pass
        return None

    def _cost(self, param, meta, cost_name: str) -> float:
        external_cost = self._maybe_external_cost(param, meta, cost_name)
        if external_cost is not None:
            return external_cost
        shape = _shape_from_slot_item(param, meta)
        if cost_name == "numel":
            return _numel_from_shape(shape)
        if cost_name == "flops":
            return get_optim_flops_from_param(
                param,
                optimizer=self.optimizer,
                soap_precondition_frequency=self.soap_precondition_frequency,
                soap_max_precond_dim=self.soap_max_precond_dim,
                shape=shape,
            )
        raise ValueError(f"Unknown visualization cost: {cost_name}")

    def _colorize_bar(self, bar: str, load: float, max_load: float, min_load: float) -> str:
        if not self.use_color:
            return bar
        if max_load == min_load:
            color = self.colors["green"]
        else:
            ratio = (load - min_load) / (max_load - min_load + 1e-6)
            if ratio > 0.8:
                color = self.colors["fail"]
            elif ratio < 0.2:
                color = self.colors["blue"]
            else:
                color = self.colors["green"]
        return f"{color}{bar}{self.colors['end']}"

    def render_cost(
        self,
        micro_param_groups,
        cost_name: str,
        title: str = "FSDP Load Balance Report",
        max_bar_width: int = 40,
    ) -> str:
        lines = [f"\n=== {title} ===", f"Total Micro Groups: {len(micro_param_groups)}", ""]
        total_load = 0.0
        total_idle_load = 0.0

        for group_idx, group in enumerate(micro_param_groups):
            rank_loads = [
                sum(self._cost(param, meta, cost_name) for param, meta in slot)
                for _, slot in group
            ]
            if not rank_loads:
                continue

            max_load = max(rank_loads)
            min_load = min(rank_loads)
            avg_load = sum(rank_loads) / len(rank_loads)
            imbalance_ratio = max_load / avg_load if avg_load > 0 else 1.0
            total_load += sum(rank_loads)
            total_idle_load += sum(max_load - load for load in rank_loads)

            lines.append(
                f"Micro Group [{group_idx}] "
                f"(Max Imbalance: {imbalance_ratio:.4f}x | "
                f"Spread: {self._human_readable(cost_name, max_load - min_load)})"
            )
            for rank_idx, ((rank, _), load) in enumerate(zip(group, rank_loads)):
                bar_len = int((load / max_load) * max_bar_width) if max_load > 0 else 0
                bar = self.bar_char * bar_len + self.empty_char * (max_bar_width - bar_len)
                bar = self._colorize_bar(bar, load, max_load, min_load)
                lines.append(f"  Rank {rank} (slot {rank_idx}): [{bar}] {self._human_readable(cost_name, load)}")
            lines.append("-" * 60)

        ideal_total = total_load + total_idle_load
        efficiency = (total_load / ideal_total) * 100 if ideal_total > 0 else 100.0
        lines.append("Summary:")
        lines.append(f"  Total Processed: {self._human_readable(cost_name, total_load)}")
        lines.append(f"  Approx. Computational Efficiency: {efficiency:.4f}%")
        lines.append("=" * 60)
        return "\n".join(lines)

    def visualize_cost(self, micro_param_groups, cost_name: str, title: str = "FSDP Load Balance Report") -> str:
        report = self.render_cost(micro_param_groups, cost_name=cost_name, title=title)
        self._emit(report)
        return report

    def visualize(self, micro_param_groups, title: str = "FSDP Load Balance Report") -> str:
        memory_report = self.visualize_cost(
            micro_param_groups,
            cost_name="numel",
            title=f"{title} (Memory View)",
        )
        flops_report = self.visualize_cost(
            micro_param_groups,
            cost_name="flops",
            title=f"{title} (FLOPs View)",
        )
        return f"{memory_report}\n{flops_report}"


LoadVisualizer = FSDPLoadVisualizer
