#!/usr/bin/env python3
"""
Script to benchmark MoE Layer forward pass for tail effect analysis with Nvidia tools.
Supports Prefill scenario (long sequences ~10K tokens) and various token-expert distributions.

Usage:
    # For nsys profiling:
    nsys profile --trace=cuda,nvtx --output=moe_profile python benchmark_moe_forward.py --distribution zipf
    
    # For ncu profiling:
    ncu --set full python benchmark_moe_forward.py --distribution hotspot
    
    # For nvprof:
    nvprof python benchmark_moe_forward.py --distribution fragmented
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import json
from typing import Tuple, Optional
from scattermoe.mlp import MLP
from scattermoe.parallel_experts import flatten_sort_count
from scattermoe.analysis import compute_block_expert_stats, format_block_stats_summary
import scattermoe

# Disable TF32 for more precise measurements
scattermoe.kernels.ops.ALLOW_TF32 = False

# Block size used in scattermoe kernels
try:
    from scattermoe.kernels.ops import BLOCK_M as KERNEL_BLOCK_M
    BLOCK_M = KERNEL_BLOCK_M
except ImportError:
    BLOCK_M = 128  # Fallback default


def analyze_expert_distribution(expert_idxs: torch.Tensor, num_experts: int, block_size: int = BLOCK_M):
    """Analyze token-expert distribution and block-level fragmentation."""
    num_tokens = expert_idxs.numel()
    top_k = expert_idxs.shape[-1] if expert_idxs.dim() > 1 else 1
    
    # Flatten expert indices
    expert_idxs_flat = expert_idxs.flatten()
    
    # Global distribution: tokens per expert
    expert_counts = torch.bincount(expert_idxs_flat, minlength=num_experts)
    tokens_per_expert = expert_counts.float() / num_tokens
    
    # Block-level fragmentation analysis
    # Group tokens into blocks of size block_size
    num_blocks = (num_tokens + block_size - 1) // block_size
    unique_experts_per_block = []
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_tokens)
        block_experts = expert_idxs_flat[start_idx:end_idx]
        unique_count = len(torch.unique(block_experts))
        unique_experts_per_block.append(unique_count)
    
    unique_experts_per_block = torch.tensor(unique_experts_per_block)
    
    return {
        'tokens_per_expert': tokens_per_expert,
        'expert_counts': expert_counts,
        'unique_experts_per_block': unique_experts_per_block,
        'mean_unique_per_block': unique_experts_per_block.float().mean().item(),
        'max_unique_per_block': unique_experts_per_block.max().item(),
        'min_unique_per_block': unique_experts_per_block.min().item(),
        'std_unique_per_block': unique_experts_per_block.float().std().item(),
    }


def generate_expert_routing_uniform(
    num_tokens: int, num_experts: int, top_k: int, 
    device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1. Uniform distribution: tokens evenly distributed across experts."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate uniform random logits
    logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    weights = torch.softmax(logits.float(), dim=-1).to(dtype)
    k_weights, k_idxs = torch.topk(weights, top_k, dim=-1)
    
    return k_weights, k_idxs


def generate_expert_routing_zipf(
    num_tokens: int, num_experts: int, top_k: int,
    exponent: float = 1.5, device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2. Zipf / power-law distribution: long-tail, few experts get most tokens."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate Zipf-distributed probabilities
    ranks = torch.arange(1, num_experts + 1, dtype=torch.float32, device=device)
    probs = 1.0 / (ranks ** exponent)
    probs = probs / probs.sum()
    
    # Sample expert assignments
    expert_assignments = torch.multinomial(probs.unsqueeze(0).expand(num_tokens, -1), top_k, replacement=True)
    
    # Generate weights (higher for lower-ranked experts)
    weights = torch.softmax(torch.randn(num_tokens, num_experts, dtype=dtype, device=device), dim=-1)
    k_weights, _ = torch.topk(weights, top_k, dim=-1)
    k_idxs = expert_assignments
    
    return k_weights, k_idxs


def generate_expert_routing_hotspot(
    num_tokens: int, num_experts: int, top_k: int,
    hotspot_ratio: float = 0.75, device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3. Hotspot + background: 60-90% tokens go to expert 0, rest uniform."""
    if seed is not None:
        torch.manual_seed(seed)
    
    num_hotspot = int(num_tokens * hotspot_ratio)
    num_background = num_tokens - num_hotspot
    
    # Hotspot tokens go to expert 0
    hotspot_idxs = torch.zeros(num_hotspot, top_k, dtype=torch.long, device=device)
    hotspot_weights = torch.ones(num_hotspot, top_k, dtype=dtype, device=device) / top_k
    
    # Background tokens uniformly distributed
    background_logits = torch.randn(num_background, num_experts, dtype=dtype, device=device)
    background_weights = torch.softmax(background_logits.float(), dim=-1).to(dtype)
    background_k_weights, background_k_idxs = torch.topk(background_weights, top_k, dim=-1)
    
    # Combine
    k_idxs = torch.cat([hotspot_idxs, background_k_idxs], dim=0)
    k_weights = torch.cat([hotspot_weights, background_k_weights], dim=0)
    
    # Shuffle to mix hotspot and background
    perm = torch.randperm(num_tokens, device=device)
    k_idxs = k_idxs[perm]
    k_weights = k_weights[perm]
    
    return k_weights, k_idxs


def generate_expert_routing_bimodal(
    num_tokens: int, num_experts: int, top_k: int,
    num_large_experts: int = 2, large_expert_ratio: float = 0.7,
    device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """4. Bimodal: small group of experts get most tokens, rest get few."""
    if seed is not None:
        torch.manual_seed(seed)
    
    num_large_tokens = int(num_tokens * large_expert_ratio)
    num_small_tokens = num_tokens - num_large_tokens
    
    # Large experts (first num_large_experts)
    large_expert_idxs = torch.randint(0, num_large_experts, (num_large_tokens,), device=device)
    large_k_idxs = large_expert_idxs.unsqueeze(1).expand(-1, top_k)
    large_k_weights = torch.ones(num_large_tokens, top_k, dtype=dtype, device=device) / top_k
    
    # Small experts (rest)
    small_expert_idxs = torch.randint(num_large_experts, num_experts, (num_small_tokens,), device=device)
    small_k_idxs = small_expert_idxs.unsqueeze(1).expand(-1, top_k)
    small_k_weights = torch.ones(num_small_tokens, top_k, dtype=dtype, device=device) / top_k
    
    # Combine
    k_idxs = torch.cat([large_k_idxs, small_k_idxs], dim=0)
    k_weights = torch.cat([large_k_weights, small_k_weights], dim=0)
    
    # Shuffle
    perm = torch.randperm(num_tokens, device=device)
    k_idxs = k_idxs[perm]
    k_weights = k_weights[perm]
    
    return k_weights, k_idxs


def generate_expert_routing_fragmented(
    num_tokens: int, num_experts: int, top_k: int,
    block_size: int = BLOCK_M, device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """5. Fragmented / Max-unique: maximize unique experts per block."""
    if seed is not None:
        torch.manual_seed(seed)
    
    num_blocks = (num_tokens + block_size - 1) // block_size
    k_idxs_list = []
    k_weights_list = []
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_tokens)
        block_size_actual = end_idx - start_idx
        
        # Within each block, maximize unique experts
        # Cycle through experts to maximize diversity
        block_experts = torch.arange(block_size_actual, device=device) % num_experts
        block_k_idxs = block_experts.unsqueeze(1).expand(-1, top_k)
        
        # Ensure we don't exceed num_experts
        block_k_idxs = block_k_idxs % num_experts
        
        block_k_weights = torch.ones(block_size_actual, top_k, dtype=dtype, device=device) / top_k
        k_idxs_list.append(block_k_idxs)
        k_weights_list.append(block_k_weights)
    
    k_idxs = torch.cat(k_idxs_list, dim=0)
    k_weights = torch.cat(k_weights_list, dim=0)
    
    return k_weights, k_idxs


def generate_expert_routing_block_aligned(
    num_tokens: int, num_experts: int, top_k: int,
    block_size: int = BLOCK_M, device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-aligned routing: each block (in flattened expert_idxs) contains only one expert.
    Expert assignment uses round-robin: e = b % num_experts
    
    This ensures that after flatten_sort_count(), each BLOCK_M-sized block in
    sorted_expert_idxs will contain tokens from a single expert, minimizing
    kernel overhead from expert switching.
    """
    # Total (token, expert) pairs
    L = num_tokens * top_k
    num_blocks = (L + block_size - 1) // block_size
    
    # Block-wise expert assignment (round-robin)
    block_experts = (torch.arange(num_blocks, device=device, dtype=torch.int64) % num_experts).to(torch.long)
    expert_idxs_flat = block_experts.repeat_interleave(block_size)[:L]  # Truncate to exactly L
    
    # Reshape to [T, K]
    expert_idxs = expert_idxs_flat.view(num_tokens, top_k)
    
    # Gate weights: simplest uniform weights
    expert_p = torch.full((num_tokens, top_k), 1.0 / top_k, device=device, dtype=dtype)
    
    return expert_p, expert_idxs


def generate_expert_routing_shuffled_after_group(
    num_tokens: int, num_experts: int, top_k: int,
    block_size: int = BLOCK_M, device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """7. Shuffled-after-group: global grouping but shuffled within blocks."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # First create block-aligned distribution
    k_weights, k_idxs = generate_expert_routing_block_aligned(
        num_tokens, num_experts, top_k, block_size, device, dtype, seed
    )
    
    # Shuffle within each block to break locality
    num_blocks = (num_tokens + block_size - 1) // block_size
    shuffled_indices = []
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_tokens)
        block_indices = torch.arange(start_idx, end_idx, device=device)
        perm = torch.randperm(len(block_indices), device=device)
        shuffled_indices.append(block_indices[perm])
    
    shuffled_indices = torch.cat(shuffled_indices)
    k_idxs = k_idxs[shuffled_indices]
    k_weights = k_weights[shuffled_indices]
    
    return k_weights, k_idxs


def generate_expert_routing_sequence_correlated(
    num_tokens: int, num_experts: int, top_k: int,
    burst_length: int = 100, device='cuda', dtype=torch.float32, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """8. Sequence-correlated / Bursty: consecutive tokens prefer same experts."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Pre-allocate all tensors on GPU to avoid .item() calls
    k_idxs = torch.zeros(num_tokens, top_k, dtype=torch.long, device=device)
    k_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device) / top_k
    
    # Generate expert groups for each burst without using .item()
    num_bursts = (num_tokens + burst_length - 1) // burst_length
    burst_expert_groups = torch.randint(0, num_experts, (num_bursts,), device=device)
    
    for i in range(num_tokens):
        burst_id = i // burst_length
        current_expert_group = burst_expert_groups[burst_id]
        
        # Assign to current expert group with some variation (all on GPU)
        expert_idxs = (current_expert_group + torch.randint(0, min(3, num_experts), (top_k,), device=device)) % num_experts
        k_idxs[i] = expert_idxs
    
    return k_weights, k_idxs


DISTRIBUTION_GENERATORS = {
    'uniform': generate_expert_routing_uniform,
    'zipf': generate_expert_routing_zipf,
    'hotspot': generate_expert_routing_hotspot,
    'bimodal': generate_expert_routing_bimodal,
    'fragmented': generate_expert_routing_fragmented,
    'block-aligned': generate_expert_routing_block_aligned,
    'shuffled': generate_expert_routing_shuffled_after_group,
    'bursty': generate_expert_routing_sequence_correlated,
}


def benchmark_moe_forward(
    num_tokens=10000,
    input_size=512,
    hidden_size=2048,
    num_experts=8,
    top_k=2,
    distribution='uniform',
    distribution_params=None,
    num_iterations=100,
    warmup_iterations=10,
    dtype=torch.float32,
    device='cuda',
    enable_nvtx=True,
    analyze_distribution=True,
    seed=None,
    dump_block_stats=False,
    block_m=None,
    dump_block_stats_json=None,
):
    """Benchmark MoE Layer forward pass with various token-expert distributions."""
    
    if distribution_params is None:
        distribution_params = {}
    
    print(f"\n{'='*60}")
    print(f"MoE Forward Benchmark - Prefill Scenario")
    print(f"{'='*60}")
    print(f"  Number of tokens: {num_tokens:,}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of experts: {num_experts}")
    print(f"  Top-k: {top_k}")
    print(f"  Distribution: {distribution}")
    print(f"  Dtype: {dtype}")
    print(f"  Device: {device}")
    print(f"  Block size: {BLOCK_M}")
    
    # Initialize MoE Layer
    mlp = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        bias=False,
        activation=nn.GELU(),
    ).to(device).to(dtype)
    
    # Generate input (flattened: num_tokens x input_size)
    x = torch.randn(num_tokens, input_size, dtype=dtype, device=device, requires_grad=False)
    
    # Store generator and params for re-computing router in each iteration
    generator = DISTRIBUTION_GENERATORS[distribution]
    generator_params = {
        'num_tokens': num_tokens,
        'num_experts': num_experts,
        'top_k': top_k,
        'device': device,
        'dtype': dtype,
        **distribution_params
    }
    
    # TEMPORARY: Generate routing ONCE outside loop for debugging
    # This allows us to measure group_by_expert_reorder cost without router overhead
    print(f"\n[DEBUG] Generating routing ONCE (fixed routing for all iterations)...")
    expert_p, expert_idxs = generator(
        seed=seed,
        **generator_params
    )
    # Ensure tensors are on GPU
    assert expert_idxs.is_cuda, f"expert_idxs must be on GPU, got device={expert_idxs.device}"
    assert expert_p.is_cuda, f"expert_p must be on GPU, got device={expert_p.device}"
    
    # Set block_m for statistics
    if block_m is None:
        block_m = BLOCK_M
    
    # Analyze distribution
    if analyze_distribution:
        print(f"\n{'='*60}")
        print(f"Token-Expert Distribution Analysis")
        print(f"{'='*60}")
        dist_stats = analyze_expert_distribution(expert_idxs, num_experts, BLOCK_M)
        
        print(f"\nGlobal distribution (tokens per expert):")
        for e in range(num_experts):
            count = dist_stats['expert_counts'][e].item()
            pct = dist_stats['tokens_per_expert'][e].item() * 100
            print(f"  Expert {e:2d}: {count:6d} tokens ({pct:5.2f}%)")
        
        print(f"\nBlock-level fragmentation:")
        print(f"  Mean unique experts per block: {dist_stats['mean_unique_per_block']:.2f}")
        print(f"  Min unique experts per block: {dist_stats['min_unique_per_block']}")
        print(f"  Max unique experts per block: {dist_stats['max_unique_per_block']}")
        print(f"  Std unique experts per block: {dist_stats['std_unique_per_block']:.2f}")
    
    # Warmup
    print(f"\n{'='*60}")
    print(f"Warming up ({warmup_iterations} iterations)...")
    torch.cuda.synchronize()
    for _ in range(warmup_iterations):
        with torch.cuda.stream(torch.cuda.Stream()):
            _ = mlp(x, expert_p, expert_idxs)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"\n{'='*60}")
    print(f"Running benchmark ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    
    events = []
    block_stats_computed = False
    for i in range(num_iterations):
        if enable_nvtx:
            torch.cuda.nvtx.range_push(f"iteration_{i}")
        try:
            if enable_nvtx:
                torch.cuda.nvtx.range_push("prefill")
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                
                # TEMPORARY: Router generation moved outside loop for debugging
                # Router: topk / softmax / indices
                # In real scenario: router_logits -> softmax -> topk -> expert_idxs, expert_p
                # if enable_nvtx:
                #     torch.cuda.nvtx.range_push("router")
                # try:
                #     # Re-compute router in each iteration to simulate real scenario
                #     # Use same seed to maintain distribution consistency across iterations
                #     expert_p_iter, expert_idxs_iter = generator(
                #         seed=seed,  # Use same seed for consistent distribution
                #         **generator_params
                #     )
                #     # Ensure tensors are on GPU (avoid HtoD/DtoH transfers)
                #     assert expert_idxs_iter.is_cuda, f"expert_idxs_iter must be on GPU, got device={expert_idxs_iter.device}"
                #     assert expert_p_iter.is_cuda, f"expert_p_iter must be on GPU, got device={expert_p_iter.device}"
                # finally:
                #     if enable_nvtx:
                #         torch.cuda.nvtx.range_pop()  # router
                
                # Compute block statistics (only on first iteration to avoid overhead)
                if dump_block_stats and not block_stats_computed:
                    # Get sorted_expert_idxs (actual tensor passed to kernel)
                    sorted_expert_idxs, _, _ = flatten_sort_count(expert_idxs, num_experts=num_experts)
                    block_stats = compute_block_expert_stats(
                        expert_idxs=sorted_expert_idxs,
                        block_m=block_m,
                        num_experts=num_experts,
                        device=device,
                    )
                    # Print summary
                    summary_str = format_block_stats_summary(
                        block_stats,
                        distribution=distribution,
                        num_tokens=num_tokens,
                        num_experts=num_experts,
                        top_k=top_k,
                        block_m=block_m,
                    )
                    print(summary_str)
                    
                    # Save to JSON if requested
                    if dump_block_stats_json:
                        json_data = {
                            'distribution': distribution,
                            'num_tokens': num_tokens,
                            'num_experts': num_experts,
                            'top_k': top_k,
                            'block_m': block_m,
                            'summary': {
                                'span': {k: float(v) for k, v in block_stats['summary']['span'].items()},
                                'unique': {k: float(v) for k, v in block_stats['summary']['unique'].items()},
                                'transitions': {k: float(v) for k, v in block_stats['summary']['transitions'].items()},
                                'major_ratio': {k: float(v) for k, v in block_stats['summary']['major_ratio'].items()},
                                'frac_mixed': float(block_stats['summary']['frac_mixed']),
                                'frac_heavy': float(block_stats['summary']['frac_heavy']),
                                'num_blocks': int(block_stats['summary']['num_blocks']),
                                'num_valid_blocks': int(block_stats['summary']['num_valid_blocks']),
                            }
                        }
                        with open(dump_block_stats_json, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        print(f"[block_stats] Saved to {dump_block_stats_json}")
                    
                    block_stats_computed = True
                
                # MoE Layer forward (use fixed routing from outside loop)
                if enable_nvtx:
                    torch.cuda.nvtx.range_push("moe_forward")
                try:
                    output = mlp(x, expert_p, expert_idxs)
                finally:
                    if enable_nvtx:
                        torch.cuda.nvtx.range_pop()  # moe_forward
                
                end_event.record()
            finally:
                if enable_nvtx:
                    torch.cuda.nvtx.range_pop()  # prefill
        finally:
            if enable_nvtx:
                torch.cuda.nvtx.range_pop()  # iteration
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        events.append(elapsed_time)
    
    # Statistics
    # Note: These statistics measure variation ACROSS multiple batches,
    # not within a single batch. Each iteration processes one complete batch,
    # and we measure the execution time of each batch forward pass.
    # This helps identify tail effects where some batches take significantly
    # longer than others (e.g., due to imbalanced expert assignments).
    times_ms = torch.tensor(events)
    mean_time = times_ms.mean().item()
    std_time = times_ms.std().item()
    min_time = times_ms.min().item()
    max_time = times_ms.max().item()
    max_time_iter = times_ms.argmax().item()  # Find iteration with max time
    p50_time = times_ms.median().item()
    p95_time = torch.quantile(times_ms, 0.95).item()
    p99_time = torch.quantile(times_ms, 0.99).item()
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results (across {num_iterations} iterations)")
    print(f"{'='*60}")
    print(f"Mean time: {mean_time:.3f} ms")
    print(f"Std time: {std_time:.3f} ms")
    print(f"Min time: {min_time:.3f} ms")
    print(f"Max time: {max_time:.3f} ms (iteration {max_time_iter})")
    print(f"P50 time: {p50_time:.3f} ms")
    print(f"P95 time: {p95_time:.3f} ms")
    print(f"P99 time: {p99_time:.3f} ms")
    print(f"\nTail effect metrics (batch-to-batch variation):")
    print(f"  Max/Mean ratio: {max_time/mean_time:.2f}x")
    print(f"  P99/Mean ratio: {p99_time/mean_time:.2f}x")
    print(f"  P95/Mean ratio: {p95_time/mean_time:.2f}x")
    print(f"{'='*60}\n")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark MoE Layer forward pass for Prefill scenario with various distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distribution types:
  uniform          - Tokens evenly distributed across experts
  zipf             - Power-law / long-tail distribution (few experts get most tokens)
  hotspot          - Single hotspot expert (60-90% tokens) + uniform background
  bimodal          - Small group of large experts + many small experts
  fragmented       - Maximize unique experts per block (worst locality)
  block-aligned    - Each block (in flattened expert_idxs) contains only one expert (best locality)
  shuffled         - Global grouping but shuffled within blocks
  bursty           - Sequence-correlated, consecutive tokens prefer same experts

Example usage:
  python benchmark_moe_forward.py --distribution zipf --num-tokens 10000
  python benchmark_moe_forward.py --distribution hotspot --hotspot-ratio 0.8
  python benchmark_moe_forward.py --distribution fragmented --num-tokens 10000
        """
    )
    
    parser.add_argument('--num-tokens', type=int, default=10000,
                        help='Number of tokens (sequence length) (default: 10000)')
    parser.add_argument('--input-size', type=int, default=512,
                        help='Input dimension (default: 512)')
    parser.add_argument('--hidden-size', type=int, default=2048,
                        help='Hidden dimension (default: 2048)')
    parser.add_argument('--num-experts', type=int, default=8,
                        help='Number of experts (default: 8)')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Top-k experts to use (default: 2)')
    parser.add_argument('--distribution', type=str, default='uniform',
                        choices=list(DISTRIBUTION_GENERATORS.keys()),
                        help='Token-expert distribution type (default: uniform)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type (default: float32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--disable-nvtx', action='store_true',
                        help='Disable NVTX markers for profiling')
    parser.add_argument('--no-analyze', action='store_true',
                        help='Disable distribution analysis')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dump-block-stats', action='store_true',
                        help='Compute and print block-level expert distribution statistics')
    parser.add_argument('--block-m', type=int, default=None,
                        help=f'Block size for statistics (default: {BLOCK_M} from kernel)')
    parser.add_argument('--dump-block-stats-json', type=str, default=None,
                        help='Path to save block statistics as JSON (optional)')
    
    # Distribution-specific parameters
    parser.add_argument('--zipf-exponent', type=float, default=1.5,
                        help='Zipf distribution exponent (default: 1.5)')
    parser.add_argument('--hotspot-ratio', type=float, default=0.75,
                        help='Hotspot ratio: fraction of tokens going to expert 0 (default: 0.75)')
    parser.add_argument('--bimodal-large-count', type=int, default=2,
                        help='Number of large experts in bimodal distribution (default: 2)')
    parser.add_argument('--bimodal-large-ratio', type=float, default=0.7,
                        help='Fraction of tokens going to large experts (default: 0.7)')
    parser.add_argument('--burst-length', type=int, default=100,
                        help='Burst length for sequence-correlated distribution (default: 100)')
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Build distribution parameters
    distribution_params = {}
    if args.distribution == 'zipf':
        distribution_params['exponent'] = args.zipf_exponent
    elif args.distribution == 'hotspot':
        distribution_params['hotspot_ratio'] = args.hotspot_ratio
    elif args.distribution == 'bimodal':
        distribution_params['num_large_experts'] = args.bimodal_large_count
        distribution_params['large_expert_ratio'] = args.bimodal_large_ratio
    elif args.distribution == 'bursty':
        distribution_params['burst_length'] = args.burst_length
    elif args.distribution == 'block-aligned':
        # Use specified block_m or default BLOCK_M
        distribution_params['block_size'] = args.block_m if args.block_m is not None else BLOCK_M
    
    # Run benchmark
    benchmark_moe_forward(
        num_tokens=args.num_tokens,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        distribution=args.distribution,
        distribution_params=distribution_params,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        dtype=dtype,
        device=args.device,
        enable_nvtx=not args.disable_nvtx,
        analyze_distribution=not args.no_analyze,
        seed=args.seed,
        dump_block_stats=args.dump_block_stats,
        block_m=args.block_m,
        dump_block_stats_json=args.dump_block_stats_json,
    )


if __name__ == '__main__':
    main()
