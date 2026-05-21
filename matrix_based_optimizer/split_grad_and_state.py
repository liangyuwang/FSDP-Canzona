import torch

class GradAndStateSplitter:

    SPLIT_ATTRS = [
        'is_full_attn_qkv',
        'is_shared_expert_fc1',
        'is_expert_fc1',
        'is_mlp_fc1',
        'is_linear_attn_inproj'
    ]
    SPLIT_ATTR_ALIASES = {
        'is_attention_qkv': 'is_full_attn_qkv',
        'is_moe_shared_expert_up_proj': 'is_shared_expert_fc1',
        'is_moe_expert_up_proj': 'is_expert_fc1',
        'is_mlp_up_proj': 'is_mlp_fc1',
        'is_linear_attention_in_proj': 'is_linear_attn_inproj',
    }
    SPLIT_SHAPE_KEY_ALIASES = {
        'is_attention_qkv': 'is_full_attn_qkv',
        'is_attention_qkv_num_heads': 'is_full_attn_qkv_num_heads',
        'is_moe_shared_expert_up_proj': 'is_shared_expert_fc1',
        'is_moe_expert_up_proj': 'is_expert_fc1',
        'is_mlp_up_proj': 'is_mlp_fc1',
        'is_linear_attention_in_proj': 'is_linear_attn_inproj',
        'is_linear_attention_in_proj_num_heads': 'is_linear_attn_inproj_num_heads',
    }

    def __init__(self, matrix_based_optimizer_split_params, matrix_based_optimizer_split_shape_map):
        self.matrix_based_optimizer_split_params = matrix_based_optimizer_split_params
        self.matrix_based_optimizer_split_shape_map = self.normalize_split_shape_map(matrix_based_optimizer_split_shape_map)
        self.split_attrs = self.get_attrs()

    @classmethod
    def normalize_split_shape_map(cls, split_shape_map):
        if split_shape_map is None:
            return {}
        normalized = dict(split_shape_map)
        for alias, canonical in cls.SPLIT_SHAPE_KEY_ALIASES.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized[alias]
        return normalized

    def get_attrs(self):
        return self.SPLIT_ATTRS + list(self.SPLIT_ATTR_ALIASES.keys())

    @classmethod
    def get_split_param_methods(cls, param):
        methods = [attr for attr in cls.SPLIT_ATTRS if getattr(param, attr, False)]
        methods.extend(
            canonical
            for alias, canonical in cls.SPLIT_ATTR_ALIASES.items()
            if getattr(param, alias, False)
        )
        return list(dict.fromkeys(methods))

    def compute_split_grad_shapes(self, param, split_method: str):
        split_shape_map = self.matrix_based_optimizer_split_shape_map
        if split_method not in split_shape_map:
            raise ValueError(f'Unknown split_method {split_method}')
        split_shapes = split_shape_map[split_method]
        hidden_size = param.shape[-1]
        if split_method in ['is_expert_fc1', 'is_shared_expert_fc1', 'is_mlp_fc1']:
            return [torch.Size((split_size, hidden_size)) for split_size in split_shapes]
        elif split_method == 'is_linear_attn_inproj':
            if 'is_linear_attn_inproj_num_heads' in split_shape_map: # pre head split
                num_key_heads, num_value_heads = split_shape_map['is_linear_attn_inproj_num_heads']
                # Split q,k into num_key_heads pieces
                q_split = [split_shapes[0]] * num_key_heads
                k_split = [split_shapes[1]] * num_key_heads
                # Split v,z,b,a into num_value_heads pieces
                v_split = [split_shapes[2]] * num_value_heads
                z_split = [split_shapes[3]] * num_value_heads
                b_split = [split_shapes[4]] * num_value_heads
                a_split = [split_shapes[5]] * num_value_heads
                split_shapes = q_split + k_split + v_split + z_split + b_split + a_split
            return [torch.Size((split_size, hidden_size)) for split_size in split_shapes]
        elif split_method == 'is_full_attn_qkv':
            if 'is_full_attn_qkv_num_heads' in split_shape_map: # pre head split
                channels, gate = split_shape_map['is_full_attn_qkv']
                num_head,num_query_group = split_shape_map['is_full_attn_qkv_num_heads']
                qs = [channels] * num_head
                ks = [channels] * num_query_group
                vs = [channels] * num_query_group
                if gate == 2:
                    return [torch.Size((q, hidden_size)) for q in qs] + \
                        [torch.Size((q, hidden_size)) for q in qs] + \
                        [torch.Size((k, hidden_size)) for k in ks] + \
                        [torch.Size((v, hidden_size)) for v in vs]
                else:
                    return [torch.Size((q, hidden_size)) for q in qs] + \
                        [torch.Size((k, hidden_size)) for k in ks] + \
                        [torch.Size((v, hidden_size)) for v in vs]
            else:
                channels, num_head, num_query_group, gate = split_shapes
                q_shape = torch.Size((num_head*channels, hidden_size))
                k_shape = torch.Size((channels*num_query_group, hidden_size))
                v_shape = torch.Size((channels*num_query_group, hidden_size))
                if gate == 2:
                    return [q_shape, q_shape, k_shape, v_shape]
                else:
                    return [q_shape, k_shape, v_shape]
        else:
            raise NotImplementedError(f'UNKNOWN split_method {split_method}')

    def split(self, grad_before_split, split_method, grad_shape):
        if split_method in ['is_linear_attn_inproj','is_expert_fc1', 'is_shared_expert_fc1', 'is_mlp_fc1', 'is_full_attn_qkv']:
            self.dim = 0
        elif split_method in []:
            self.dim = 1
        else:
            raise ValueError(f"Unknown split method {split_method}")
        self.split_info = []
        # for linear fc1s and linear attn in proj, split it directly
        if split_method in ['is_expert_fc1', 'is_shared_expert_fc1', 'is_mlp_fc1']:
            split_shapes = self.matrix_based_optimizer_split_shape_map[split_method]
            grads = torch.split(grad_before_split, split_shapes, dim=self.dim)
        elif split_method == 'is_linear_attn_inproj':
            split_shapes = self.matrix_based_optimizer_split_shape_map[split_method]
            if 'is_linear_attn_inproj_num_heads' in self.matrix_based_optimizer_split_shape_map: # per head split mode
                # Per head split mode
                num_key_heads, num_value_heads = self.matrix_based_optimizer_split_shape_map['is_linear_attn_inproj_num_heads']
                # Split q,k into num_key_heads pieces
                q_split = [split_shapes[0]] * num_key_heads
                k_split = [split_shapes[1]] * num_key_heads
                # Split v,z,b,a into num_value_heads pieces
                v_split = [split_shapes[2]] * num_value_heads
                z_split = [split_shapes[3]] * num_value_heads
                b_split = [split_shapes[4]] * num_value_heads
                a_split = [split_shapes[5]] * num_value_heads
                split_shapes = q_split + k_split + v_split + z_split + b_split + a_split

            grads = torch.split(grad_before_split, split_shapes, dim=self.dim)
        elif split_method in ['is_full_attn_qkv']:
            if 'is_full_attn_qkv_num_heads' in self.matrix_based_optimizer_split_shape_map:
                channels, gate = self.matrix_based_optimizer_split_shape_map['is_full_attn_qkv']
                num_head,num_query_group = self.matrix_based_optimizer_split_shape_map['is_full_attn_qkv_num_heads']
                per_head = True
            else:
                (channels, num_head, num_query_group, gate) = self.matrix_based_optimizer_split_shape_map[split_method]
                per_head = False
            hidden_size = grad_shape[1]
            self.split_info = [channels, num_head, num_query_group, gate, hidden_size, per_head]
            saved_shape = [num_query_group, (num_head // num_query_group *gate +2) * channels, hidden_size]
            g_reshaped = grad_before_split.view(*saved_shape)
            q,k,v = g_reshaped.split([num_head // num_query_group * channels *gate, channels, channels], dim=1)
            def split_heads(x,is_q):
                if not is_q:
                    xs = [t.squeeze(0) for t in torch.chunk(x, num_query_group, dim=0)]

                else:
                    x_groups = [t.squeeze(0) for t in torch.chunk(x, num_query_group, dim=0)]
                    xs = []
                    for g in x_groups:
                        xs.extend(list(torch.chunk(g, num_head // num_query_group, dim=0)))
                assert all(xx.shape == torch.Size((channels, hidden_size)) for xx in xs), f"Expected each split to have shape {(channels, hidden_size)}, but got {[xx.shape for xx in xs]}"
                return xs

            if gate == 2: # attention_output_gate
                query_part, gate_part = torch.chunk(q, 2, dim=1)
                if not per_head:
                    grads = [query_part.reshape(-1,hidden_size), gate_part.reshape(-1,hidden_size), k.reshape(-1,hidden_size), v.reshape(-1,hidden_size)]
                else:
                    ks = split_heads(k, False)
                    vs = split_heads(v, False)
                    query_parts = split_heads(query_part, True)
                    gate_parts = split_heads(gate_part, True)
                    grads = query_parts+gate_parts+ks+vs
                    assert len(grads) == num_head*2 + num_query_group*2, f"Expected {num_head*2 + num_query_group*2} splits, but got {len(grads)}"
            elif gate == 1:
                if not per_head:
                    grads = [q.reshape(-1,hidden_size),k.reshape(-1,hidden_size),v.reshape(-1,hidden_size)]
                else:
                    ks = split_heads(k, False)
                    vs = split_heads(v, False)
                    qs = split_heads(q, True)
                    grads = qs+ks+vs
                    assert len(grads) == num_head + num_query_group*2, f"Expected {num_head + num_query_group*2} splits, but got {len(grads)}"
        else:
            raise ValueError(f"Unknown split method {split_method}")
        assert all([len(gg.shape) == 2 for gg in grads]), f"All split grads must be 2D, got {[gg.shape for gg in grads]}"
        return grads

    def gather(self, grads_after_split, split_method, original_shape):
        if split_method in ['is_expert_fc1', 'is_shared_expert_fc1', 'is_mlp_fc1', 'is_linear_attn_inproj']:
            return torch.cat(grads_after_split, dim=self.dim).view(original_shape)
        elif split_method in ['is_full_attn_qkv']:
            channels, num_head, num_query_group, gate, hidden_size, per_head = self.split_info
            def merge_heads(xs, is_q):
                if not is_q:
                    return torch.stack(xs, dim=0)

                else:
                    group_size = num_head // num_query_group
                    xs_3d = [t.unsqueeze(0) for t in xs]

                    merged_groups = []
                    for i in range(num_query_group):
                        start_idx = i * group_size
                        end_idx = start_idx + group_size
                        group_merged = torch.cat(xs_3d[start_idx:end_idx], dim=1)
                        merged_groups.append(group_merged)

                    return torch.cat(merged_groups, dim=0)
            if not per_head:
                if gate ==2:
                    q = torch.cat(
                        [grads_after_split[0].reshape(num_query_group, num_head // num_query_group*channels, hidden_size),
                        grads_after_split[1].reshape(num_query_group, num_head // num_query_group*channels, hidden_size)], dim=1)
                elif gate == 1:
                    q = grads_after_split[0].reshape(num_query_group, num_head // num_query_group*channels, hidden_size)
                k = grads_after_split[-2].reshape(num_query_group, channels, hidden_size)
                v = grads_after_split[-1].reshape(num_query_group, channels, hidden_size)
            else:
                if gate == 2:
                    query_parts = grads_after_split[:num_head]
                    gate_parts = grads_after_split[num_head:num_head*2]
                    k_parts = grads_after_split[num_head*2:num_head*2+num_query_group]
                    v_parts = grads_after_split[num_head*2+num_query_group:]
                    query_part = merge_heads(query_parts, True)
                    gate_part = merge_heads(gate_parts, True)
                    q = torch.cat([query_part, gate_part], dim=1)
                elif gate == 1:
                    q_parts = grads_after_split[:num_head]
                    k_parts = grads_after_split[num_head:num_head+num_query_group]
                    v_parts = grads_after_split[num_head+num_query_group:]
                    q = merge_heads(q_parts, True)
                k = merge_heads(k_parts, False)
                v = merge_heads(v_parts, False)
            return torch.cat([q,k,v], dim=1).view(original_shape)
        else:
            raise ValueError(f"Unknown split method {split_method}")
