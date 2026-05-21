import torch

from ..load_balanced_fsdp_executor import FSDPShardExecutor, get_fsdp_shard_spec, is_group_fsdp_sharded
from ..split_grad_and_state import GradAndStateSplitter
import gc
import os
import copy

class BaseOptim(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        defaults,
        split_params = False,
        split_shape_map = None,
        async_tp = False,
        fsdp_sharded = False,
        fsdp_group = None,
        fsdp_balance = "global",
        fsdp_fused_comm = True,
        fsdp_max_numel_per_slot = None,
        fsdp_balance_cost = "numel",
        fsdp_overlap = "none",
    ):
        # Let base class normalize groups and fill defaults
        super().__init__(params, defaults)
        self.grad_and_state_splitter = GradAndStateSplitter(split_params, split_shape_map)
        self.fsdp_defaults = {
            "is_fsdp_sharded": fsdp_sharded,
            "fsdp_group": fsdp_group,
            "fsdp_balance": fsdp_balance,
            "fsdp_fused_comm": fsdp_fused_comm,
            "fsdp_max_numel_per_slot": fsdp_max_numel_per_slot,
            "fsdp_balance_cost": fsdp_balance_cost,
            "fsdp_overlap": fsdp_overlap,
        }
        # add cuda graph
        self.use_cuda_graph = int(os.environ.get('USE_CUDA_GRAPH_OPTIM', 0)) == 1
        if self.use_cuda_graph:
            self._cuda_graphs = {}
            self._cuda_graph_warmup_steps = 3
            self._cuda_graph_current_step = 0
            self._group_meta = copy.deepcopy(defaults)
            self._mempool = torch.cuda.graph_pool_handle()
    
    def _single_param_update(self, p, u, group):
        p.data.add_(u.reshape_as(p.data))

    def _single_param_step(self, p, s, group, g=None):
        if is_group_fsdp_sharded(group) and g is None:
            raise RuntimeError("'g' must be provided when using FSDP-sharded parameters.")
        g = p.grad.view(s) if g is None else g

        true_attrs = self.grad_and_state_splitter.get_split_param_methods(p)
        if true_attrs:
            assert len(true_attrs) == 1, f"Only one of {self.grad_and_state_splitter.get_attrs()} can be set for a param, got {true_attrs}"
            split_method = true_attrs[0]
            grads = self.grad_and_state_splitter.split(g, split_method, g.shape)  # use g.shape, not s
            grads = [self._inner_single_param_step(f"{split_method}.{idx}.", p, gg, group)
                for idx, gg in enumerate(grads)]
            if None in grads:
                return None
            u = self.grad_and_state_splitter.gather(grads, split_method, g.shape)  # use g.shape, not s

        else:
            u = self._inner_single_param_step("", p, g, group)
            if u is None:
                return None
        return u.to(g.dtype)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self.use_cuda_graph:
            return self.step_with_cuda_graph(loss)
        
        for group in self.param_groups:
            self._apply_fsdp_defaults(group)
            # Increment step counter to stay consistent with TE FusedAdam.
            # _synchronize_steps() in ChainedOptimizer relies on this for
            # cross-optimizer step alignment.
            group["step"] = group.get("step", 0) + 1
            shapes = self._get_group_shapes(group)
            shapes_map = {p: {'origin_shape': s} for p, s in zip(group['params'], shapes)}
            group["shapes_map"] = shapes_map
            if is_group_fsdp_sharded(group):
                FSDPShardExecutor.from_param_group(group).execute(
                    group,
                    self._single_param_step,
                    self._single_param_update,
                )
            else:
                for p, s in zip(group["params"], shapes):
                    if p.grad is None:
                        continue
                    tensor_to_update_p = self._single_param_step(p, s, group)
                    if tensor_to_update_p is not None:
                        self._single_param_update(p, tensor_to_update_p, group)
        return loss

    def _inner_single_param_step(self, name, p, grad, group):
        """
        API for a single param's optimizer step
        """
        raise NotImplementedError("_inner_single_param_step must be implemented in subclass.")

    def _get_graph_specs(self):
        """Return the list of (name, kwargs) pairs describing graphs to capture, in order.

        Each entry causes one capture step. During replay, _select_graph_for_replay()
        picks which graph to use based on the current replay step counter.

        Default: a single graph with no extra kwargs (one-path optimizers like Muon).
        Override in subclasses for multi-path optimizers (e.g. SOAP uses two graphs).
        """
        return [('default', {})]

    def _select_graph_for_replay(self, group, graphs, replay_step):
        """Return the CUDAGraph to replay for this step.

        graphs: dict[name -> CUDAGraph] built by step_with_cuda_graph during capture.
        replay_step: 0-based counter that increments once per call after all captures.

        Default: always replay the single 'default' graph.
        Override in subclasses to implement step-dependent dispatch.
        """
        return graphs['default']

    def step_with_cuda_graph(self, loss):
        specs = self._get_graph_specs()          # [(name, kwargs), ...]
        num_captures = len(specs)

        for i, group in enumerate(self.param_groups):
            self._apply_fsdp_defaults(group)
            if is_group_fsdp_sharded(group):
                raise NotImplementedError("CUDA graph capture is not implemented for FSDP-Canzona communication.")
            group["step"] = group.get("step", 0) + 1
            # Build shapes_map consistently with step() so subclasses have access to it
            shapes = self._get_group_shapes(group)
            shapes_map = {p: {'origin_shape': s} for p, s in zip(group['params'], shapes)}
            group["shapes_map"] = shapes_map

            self._upload_group_meta_to_cuda_graph(group, shapes)

            current_step = self._cuda_graph_current_step

            if current_step < self._cuda_graph_warmup_steps:
                # Run warmup on a dedicated stream so lazy CUDA inits (cuDNN benchmarking,
                # memory allocator warm-up) are never recorded into the graph.
                if current_step == 0:
                    torch.cuda.synchronize()  # flush all pending ops before warmup
                warmup_stream = torch.cuda.Stream()
                with torch.cuda.stream(warmup_stream):
                    self._run_param_updates(group, shapes)
                if current_step == self._cuda_graph_warmup_steps - 1:
                    torch.cuda.synchronize()  # ensure warmup fully done before capture

            elif current_step < self._cuda_graph_warmup_steps + num_captures:
                # Capture phase: one capture step per graph spec, in order.
                # Disable GC to avoid PyTorch bug (pytorch/pytorch#161037).
                capture_idx = current_step - self._cuda_graph_warmup_steps
                name, kwargs = specs[capture_idx]
                graph = torch.cuda.CUDAGraph()
                gc_enabled = gc.isenabled()
                if gc_enabled:
                    gc.disable()
                with torch.cuda.graph(graph, pool=self._mempool):
                    self._run_param_updates(group, shapes, **kwargs)
                if gc_enabled:
                    gc.enable()
                torch.cuda.synchronize()
                if i not in self._cuda_graphs:
                    self._cuda_graphs[i] = {}
                self._cuda_graphs[i][name] = graph

            else:
                # Replay: delegate graph selection to the subclass hook.
                replay_step = current_step - self._cuda_graph_warmup_steps - num_captures
                graph = self._select_graph_for_replay(group, self._cuda_graphs[i], replay_step)
                graph.replay()

            self._offload_group_meta_from_cuda_graph(group, shapes)

        self._cuda_graph_current_step += 1
        return loss

    def _run_param_updates(self, group, shapes, **kwargs):
        for p, s in zip(group["params"], shapes):
            if p.grad is None:
                continue
            tensor_to_update_p = self._single_param_step(p, s, group, **kwargs)
            if tensor_to_update_p is not None:
                self._single_param_update(p, tensor_to_update_p, group)

    def _upload_group_meta_to_cuda_graph(self, group, shapes):  # adjust _group_meta if more meta info changes
        if "lr_tensor" not in self._group_meta:
            self._group_meta["lr_tensor"] = torch.tensor(
                group["lr"],
                dtype=torch.float32,
                device=torch.cuda.current_device()
            )
        else:
            self._group_meta["lr_tensor"].fill_(group["lr"])
        group["lr"] = self._group_meta["lr_tensor"]

    def _offload_group_meta_from_cuda_graph(self, group, shapes):
        group["lr"] = self._group_meta["lr_tensor"].item()

    def _apply_fsdp_defaults(self, group):
        for key, value in self.fsdp_defaults.items():
            if key not in group and value is not None:
                group[key] = value

    def _get_group_shapes(self, group):
        if is_group_fsdp_sharded(group):
            return [
                get_fsdp_shard_spec(group, p, idx).full_shape
                for idx, p in enumerate(group["params"])
            ]
        if 'origin_shape' in group:
            return group["origin_shape"]
        return [p.shape for p in group['params']]
