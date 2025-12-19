import torch
from torch import nn

from .parallel_experts import ParallelExperts, flatten_sort_count

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        bias=False,
        activation=None,
    ):
        super(MLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, hidden_size, bias=bias)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias=bias)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        num_tokens = x_shape[0] if len(x_shape) == 2 else x_shape[0] * x_shape[1]
        # moe_layer with shape info: T=tokens, E=experts, K=top_k
        moe_layer_name = f"moe_layer[T={num_tokens},E={self.num_experts},K={self.top_k}]"
        torch.cuda.nvtx.range_push(moe_layer_name)
        try:
            x = x.view(-1, x_shape[-1])
            
            # Indices-only: sort and count (not the fused kernel)
            torch.cuda.nvtx.range_push("route_indices_sort_count")
            try:
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
                    flatten_sort_count(expert_idxs, num_experts=self.num_experts)
            finally:
                torch.cuda.nvtx.range_pop()  # route_indices_sort_count

            # Fused scatter2scatter kernel (first layer)
            torch.cuda.nvtx.range_push("s2s_fused_linear_up")
            try:
                h = self.experts(
                    x, self.top_k,
                    sorted_expert_idxs, sorted_scattered_idxs,
                    expert_offsets,
                    grouped_out=True
                )
            finally:
                torch.cuda.nvtx.range_pop()  # s2s_fused_linear_up
            
            # Activation
            torch.cuda.nvtx.range_push("activation")
            try:
                h = self.activation(h)
            finally:
                torch.cuda.nvtx.range_pop()  # activation
            
            # Fused scatter2scatter kernel (second layer)
            torch.cuda.nvtx.range_push("s2s_fused_linear_down")
            try:
                y = self.output_experts(
                    h, 1, sorted_expert_idxs, sorted_scattered_idxs,
                    expert_offsets,
                    grouped_in=True,
                    gates=expert_p,
                )
            finally:
                torch.cuda.nvtx.range_pop()  # s2s_fused_linear_down
            
            y = y.view(*x_shape[:-1], y.size(-1))
            return y
        finally:
            torch.cuda.nvtx.range_pop()  # moe_layer

class GLUMLP(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_experts,
        top_k,
        bias=False,
        activation=nn.SiLU(),
    ):
        super(GLUMLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, 2 * hidden_size, bias=bias)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias=bias)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        num_tokens = x_shape[0] if len(x_shape) == 2 else x_shape[0] * x_shape[1]
        # moe_layer with shape info: T=tokens, E=experts, K=top_k
        moe_layer_name = f"moe_layer[T={num_tokens},E={self.num_experts},K={self.top_k}]"
        torch.cuda.nvtx.range_push(moe_layer_name)
        try:
            x = x.view(-1, x_shape[-1])
            
            # Indices-only: sort and count (not the fused kernel)
            torch.cuda.nvtx.range_push("route_indices_sort_count")
            try:
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
                    flatten_sort_count(expert_idxs, num_experts=self.num_experts)
            finally:
                torch.cuda.nvtx.range_pop()  # route_indices_sort_count

            # Fused scatter2scatter kernel (first layer)
            torch.cuda.nvtx.range_push("s2s_fused_linear_up")
            try:
                h, gates  = self.experts(
                    x, self.top_k,
                    sorted_expert_idxs, sorted_scattered_idxs,
                    expert_offsets,
                    grouped_out=True
                ).chunk(2, dim=-1)
            finally:
                torch.cuda.nvtx.range_pop()  # s2s_fused_linear_up
            
            # Activation
            torch.cuda.nvtx.range_push("activation")
            try:
                h = self.activation(gates) * h
            finally:
                torch.cuda.nvtx.range_pop()  # activation
            
            # Fused scatter2scatter kernel (second layer)
            torch.cuda.nvtx.range_push("s2s_fused_linear_down")
            try:
                y = self.output_experts(
                    h, 1, sorted_expert_idxs, sorted_scattered_idxs,
                    expert_offsets,
                    grouped_in=True,
                    gates=expert_p,
                )
            finally:
                torch.cuda.nvtx.range_pop()  # s2s_fused_linear_down
            
            y = y.view(*x_shape[:-1], y.size(-1))
            return y
        finally:
            torch.cuda.nvtx.range_pop()  # moe_layer

