# nvtx_demo.py
import os
import time
import torch

def fake_router(x, num_experts: int, topk: int):
    """模擬 router：產生 (token, expert) pairs 的 indices。"""
    B, H = x.shape
    num_tokens = B

    # 產生 topk expert index for each token
    # shape: [num_tokens, topk]
    expert_idxs = torch.randint(
        low=0, high=num_experts, size=(num_tokens, topk), device=x.device, dtype=torch.int32
    )

    # (token, expert) pairs 展平成一維：length = num_tokens * topk
    token_idxs = torch.arange(num_tokens, device=x.device, dtype=torch.int32).repeat_interleave(topk)
    expert_idxs_flat = expert_idxs.reshape(-1)

    return token_idxs, expert_idxs_flat


def group_by_expert(token_idxs, expert_idxs, num_experts: int):
    """把 pairs 依 expert 排序（這裡用 sort 模擬你的 group-by-expert/reorder）。"""
    # 依 expert 排序
    order = torch.argsort(expert_idxs)
    token_sorted = token_idxs[order]
    expert_sorted = expert_idxs[order]

    # 找每個 expert 的 offset（方便後續 per-expert chunk）
    counts = torch.bincount(expert_sorted, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, device=counts.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return token_sorted, expert_sorted, offsets


def per_expert_gemm(x, token_sorted, expert_sorted, offsets, W):
    """
    模擬 per-expert GEMM：
    對每個 expert，把該 expert 的 token rows gather 出來做 matmul，再 scatter 回去。
    """
    B, H = x.shape
    out = torch.zeros_like(x)

    for e in range(W.shape[0]):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if end <= start:
            continue

        tok = token_sorted[start:end].to(torch.int64)

        # gather
        xe = x.index_select(0, tok)               # [n_e, H]
        # gemm
        ye = xe @ W[e]                            # [n_e, H]
        # scatter back (sum for simplicity)
        out.index_add_(0, tok, ye)

    return out


def main():
    device = "cuda"
    torch.manual_seed(0)

    # 參數
    B = 4096          # tokens
    H = 1024          # hidden
    E = 64            # experts
    topk = 2
    iters = 5
    warmup = 2

    # 資料
    x = torch.randn(B, H, device=device, dtype=torch.float16)

    # 每個 expert 一個權重矩陣 [H, H]（這裡用 fp16，會看到 GEMM kernel）
    W = torch.randn(E, H, H, device=device, dtype=torch.float16)

    # warmup（避免第一次編譯/初始化干擾）
    for _ in range(warmup):
        _ = x @ W[0]
    torch.cuda.synchronize()

    for i in range(iters):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        try:
            torch.cuda.nvtx.range_push("prefill")
            try:
                # router
                torch.cuda.nvtx.range_push("router")
                token_idxs, expert_idxs = fake_router(x, E, topk)
                torch.cuda.nvtx.range_pop()

                # group by expert / reorder
                torch.cuda.nvtx.range_push("group_by_expert")
                token_sorted, expert_sorted, offsets = group_by_expert(token_idxs, expert_idxs, E)
                torch.cuda.nvtx.range_pop()

                # per-expert gemm
                torch.cuda.nvtx.range_push("expert_gemm")
                y = per_expert_gemm(x, token_sorted, expert_sorted, offsets, W)
                torch.cuda.nvtx.range_pop()

                # scatter_back（這裡已包含在 per_expert_gemm 的 index_add，另外放個 range 示意）
                torch.cuda.nvtx.range_push("scatter_back")
                z = y.relu_()
                torch.cuda.nvtx.range_pop()

            finally:
                torch.cuda.nvtx.range_pop()  # prefill

        finally:
            torch.cuda.nvtx.range_pop()      # iter_i

    torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    # 可選：讓 CUDA launch 更同步，方便看 timeline（但會變慢）
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
