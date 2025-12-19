"""
Block-level expert distribution statistics for scatter2scatter kernel analysis.

This module computes statistics about expert distribution within each BLOCK_M-sized
block, which directly affects the performance of the _scatter2scatter Triton kernel.
"""

import torch
from typing import Dict, Optional, Union


def compute_block_expert_stats(
    expert_idxs: torch.Tensor,
    *,
    block_m: int,
    num_experts: int,
    pad_value: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_counts: bool = False,
) -> Dict:
    """
    Compute block-level expert distribution statistics.
    
    Args:
        expert_idxs: 1D tensor, length = FAN_OUT * M (same tensor passed to kernel)
        block_m: Triton BLOCK_M (block size)
        num_experts: Number of experts (E)
        pad_value: Padding value for incomplete blocks (default: num_experts, treated as invalid)
        device: Device to use for computation (default: same as expert_idxs)
        return_counts: Whether to return per-block expert counts (default: False, can be large)
    
    Returns:
        Dictionary containing:
            - "span": tensor [num_blocks] - expert span per block (max - min + 1)
            - "unique": tensor [num_blocks] - number of unique experts per block
            - "transitions": tensor [num_blocks] - number of expert id transitions per block
            - "major_ratio": tensor [num_blocks] - ratio of most frequent expert
            - "valid_count": tensor [num_blocks] - number of valid rows per block
            - "summary": dict with p50/p95/p99/max/min statistics and frac_mixed/frac_heavy
            - "counts": (optional) tensor [num_blocks, num_experts] - expert counts per block
    """
    if device is None:
        device = expert_idxs.device
    
    # Ensure 1D
    expert_idxs = expert_idxs.reshape(-1).to(device)
    original_length = expert_idxs.shape[0]
    
    # Set pad_value
    if pad_value is None:
        pad_value = num_experts
    
    # Calculate padding
    num_blocks = (original_length + block_m - 1) // block_m
    padded_length = num_blocks * block_m
    
    # Pad if necessary
    if original_length < padded_length:
        padding = torch.full(
            (padded_length - original_length,),
            pad_value,
            dtype=expert_idxs.dtype,
            device=device
        )
        expert_idxs = torch.cat([expert_idxs, padding], dim=0)
    
    # Reshape to [num_blocks, block_m]
    Eblk = expert_idxs.view(num_blocks, block_m)
    
    # Compute valid mask
    valid = (Eblk < num_experts)
    valid_count = valid.sum(dim=1)  # [num_blocks]
    
    # Compute expert counts per block
    if num_experts <= 128:
        # Use one-hot encoding for small E
        counts = _compute_counts_onehot(Eblk, valid, num_experts, device)
    else:
        # Fallback for large E (slower but works)
        counts = _compute_counts_fallback(Eblk, valid, num_experts, device)
    
    # Compute statistics per block
    # emin: minimum expert id (only valid values)
    # emax: maximum expert id (only valid values)
    # Use num_experts as sentinel for invalid values in min, -1 for max
    Eblk_masked = torch.where(valid, Eblk, num_experts)  # Invalid -> num_experts (high value)
    emin = Eblk_masked.min(dim=1)[0]  # [num_blocks]
    # For max, use -1 sentinel for invalid
    Eblk_masked_max = torch.where(valid, Eblk, -1)  # Invalid -> -1 (low value)
    emax = Eblk_masked_max.max(dim=1)[0]  # [num_blocks]
    
    # For invalid blocks (valid_count == 0), emin will be num_experts, emax will be -1
    # Set them to proper sentinel values
    emin = torch.where(valid_count > 0, emin, torch.tensor(num_experts, device=device))
    emax = torch.where(valid_count > 0, emax, torch.tensor(-1, device=device))
    
    # Span = (emax - emin + 1) for valid blocks, 0 for invalid
    span = torch.where(
        valid_count > 0,
        (emax - emin + 1).clamp(min=0),
        torch.tensor(0, dtype=torch.long, device=device)
    )
    
    # Unique experts per block
    unique = (counts > 0).sum(dim=1)  # [num_blocks]
    
    # Transitions per block: count expert id changes within each block
    # transitions[b] = sum(Eblk[b, i] != Eblk[b, i-1] for i in range(1, block_m))
    # Only count transitions between valid entries
    # Vectorized computation: compare Eblk[:, 1:] with Eblk[:, :-1]
    Eblk_prev = Eblk[:, :-1]  # [num_blocks, block_m-1]
    Eblk_next = Eblk[:, 1:]   # [num_blocks, block_m-1]
    valid_prev = valid[:, :-1]  # [num_blocks, block_m-1]
    valid_next = valid[:, 1:]   # [num_blocks, block_m-1]
    
    # Transition occurs when: (Eblk_prev != Eblk_next) AND both are valid
    transitions_mask = (Eblk_prev != Eblk_next) & valid_prev & valid_next
    transitions = transitions_mask.sum(dim=1)  # [num_blocks]
    
    # Major expert count and ratio
    major_count = counts.max(dim=1)[0]  # [num_blocks]
    major_ratio = torch.where(
        valid_count > 0,
        major_count.float() / valid_count.float(),
        torch.tensor(0.0, device=device)
    )
    
    # Compute summary statistics
    summary = _compute_summary_stats(
        span=span,
        unique=unique,
        transitions=transitions,
        major_ratio=major_ratio,
        valid_count=valid_count,
        device=device
    )
    
    result = {
        "span": span,
        "unique": unique,
        "transitions": transitions,
        "major_ratio": major_ratio,
        "valid_count": valid_count,
        "summary": summary,
    }
    
    if return_counts:
        result["counts"] = counts
    
    return result


def _compute_counts_onehot(
    Eblk: torch.Tensor,
    valid: torch.Tensor,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute expert counts using one-hot encoding (fast for small E)."""
    num_blocks, block_m = Eblk.shape
    
    # Create one-hot encoding: [num_blocks, block_m, num_experts]
    # Only count valid entries
    Eblk_clamped = torch.clamp(Eblk, 0, num_experts - 1)
    onehot = torch.zeros(
        num_blocks, block_m, num_experts,
        dtype=torch.long,
        device=device
    )
    
    # Create indices for scatter: [num_blocks, block_m, 1]
    indices = Eblk_clamped.unsqueeze(-1)  # [num_blocks, block_m, 1]
    # Create source values: 1 for valid, 0 for invalid
    src = valid.long().unsqueeze(-1)  # [num_blocks, block_m, 1]
    
    # Scatter: for each (block, row), add 1 to the expert index position
    onehot.scatter_add_(
        dim=2,
        index=indices,
        src=src
    )
    
    # Sum over block_m: [num_blocks, num_experts]
    counts = onehot.sum(dim=1)
    return counts


def _compute_counts_fallback(
    Eblk: torch.Tensor,
    valid: torch.Tensor,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Fallback method for large E (slower but works)."""
    num_blocks, block_m = Eblk.shape
    counts = torch.zeros(num_blocks, num_experts, dtype=torch.long, device=device)
    
    for b in range(num_blocks):
        block_experts = Eblk[b][valid[b]]
        if len(block_experts) > 0:
            block_counts = torch.bincount(block_experts, minlength=num_experts)
            counts[b] = block_counts
    
    return counts


def _compute_summary_stats(
    span: torch.Tensor,
    unique: torch.Tensor,
    transitions: torch.Tensor,
    major_ratio: torch.Tensor,
    valid_count: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute p50/p95/p99/max/min statistics and fraction metrics."""
    # Filter out invalid blocks (valid_count == 0)
    valid_mask = valid_count > 0
    
    def compute_percentiles(tensor, mask):
        valid_tensor = tensor[mask]
        if len(valid_tensor) == 0:
            return {
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'max': 0.0,
                'min': 0.0,
            }
        # Convert to float for quantile computation
        valid_tensor_float = valid_tensor.float()
        return {
            'p50': valid_tensor_float.median().item(),
            'p95': valid_tensor_float.quantile(0.95).item(),
            'p99': valid_tensor_float.quantile(0.99).item(),
            'max': valid_tensor.max().item(),  # Keep original dtype for max/min
            'min': valid_tensor.min().item(),
        }
    
    span_stats = compute_percentiles(span, valid_mask)
    unique_stats = compute_percentiles(unique, valid_mask)
    transitions_stats = compute_percentiles(transitions, valid_mask)
    major_ratio_stats = compute_percentiles(major_ratio, valid_mask)
    
    # Compute fraction metrics
    valid_unique = unique[valid_mask]
    
    if len(valid_unique) > 0:
        frac_mixed = (valid_unique > 1).float().mean().item()
        frac_heavy = (valid_unique >= 4).float().mean().item()
    else:
        frac_mixed = 0.0
        frac_heavy = 0.0
    
    return {
        'span': span_stats,
        'unique': unique_stats,
        'transitions': transitions_stats,
        'major_ratio': major_ratio_stats,
        'frac_mixed': frac_mixed,
        'frac_heavy': frac_heavy,
        'num_blocks': int(valid_mask.sum().item()),
        'num_valid_blocks': int(valid_mask.sum().item()),
    }


def format_block_stats_summary(
    stats: Dict,
    distribution: str = "unknown",
    num_tokens: int = 0,
    num_experts: int = 0,
    top_k: int = 0,
    block_m: int = 128,
) -> str:
    """Format block statistics summary as a readable string."""
    summary = stats['summary']
    
    lines = [
        f"[block_stats] dist={distribution} tokens={num_tokens} E={num_experts} topk={top_k} BLOCK_M={block_m}",
        f"  span:       p50={summary['span']['p50']:.0f} p95={summary['span']['p95']:.0f} "
        f"p99={summary['span']['p99']:.0f} max={summary['span']['max']:.0f}",
        f"  unique:     p50={summary['unique']['p50']:.0f} p95={summary['unique']['p95']:.0f} "
        f"p99={summary['unique']['p99']:.0f} max={summary['unique']['max']:.0f}",
        f"  transitions: p50={summary['transitions']['p50']:.0f} p95={summary['transitions']['p95']:.0f} "
        f"p99={summary['transitions']['p99']:.0f} max={summary['transitions']['max']:.0f}",
        f"  major_ratio: p50={summary['major_ratio']['p50']:.2f} p95={summary['major_ratio']['p95']:.2f} "
        f"p99={summary['major_ratio']['p99']:.2f} min={summary['major_ratio']['min']:.2f}",
        f"  frac_mixed: {summary['frac_mixed']:.3f} (blocks with unique > 1)",
        f"  frac_heavy: {summary['frac_heavy']:.3f} (blocks with unique >= 4)",
    ]
    
    return "\n".join(lines)

