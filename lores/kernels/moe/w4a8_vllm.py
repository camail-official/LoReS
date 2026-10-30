# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 expert GEMM, derived from vLLM v0.26.0's fused_moe.py."""

import math
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_w4a8(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    a_scale_ptr=None,
    stride_asm=0,
):
    """Multiply routed activations by packed W4 expert weights with A8 quantization.

    ``sorted_token_ids`` repeats routed tokens, sorted by expert and padded to
    ``BLOCK_SIZE_M``. ``expert_ids`` selects each block's expert; output is
    indexed by token, top-k route, and output channel.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type
        )
        return
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + offs_k[:, None] // 2 * stride_bk + offs_bn[None, :] * stride_bn
    b_shifter = offs_k[:, None] % 2 * 4
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs)
        b = b >> b_shifter & 15
        b_scale_i = tl.load(
            b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn + BLOCK_SIZE_K * k // group_size * stride_bsk
        ).to(tl.float32)
        b_i8 = (b.to(tl.int32) - 8).to(tl.int8)
        accumulator += tl.dot(a, b_i8).to(tl.float32) * b_scale_i[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K // 2 * stride_bk
    a_scale = tl.load(a_scale_ptr + offs_token // top_k * stride_asm, mask=token_mask, other=0.0)
    accumulator = accumulator * a_scale[:, None]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_w4a8_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    block_shape: list[int] | None,
    A_scale: torch.Tensor | None = None,
):
    from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_wna16_block_config

    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None, "int4 x int8 path: symmetric weights only"
    assert block_shape is not None and block_shape[0] == 0
    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )
    assert A.dtype == torch.int8 and A_scale is not None and (A_scale.dim() == 2), (A.dtype, A_scale)
    # Each integer dot block must stay within one weight-scale group.
    bk = math.gcd(math.gcd(config["BLOCK_SIZE_K"], block_shape[1]), A.size(1))
    assert bk >= 16, (config["BLOCK_SIZE_K"], block_shape[1], A.size(1))
    config["BLOCK_SIZE_K"] = bk
    fused_moe_kernel_w4a8[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        a_scale_ptr=A_scale,
        stride_asm=A_scale.stride(0) if A_scale is not None else 0,
        **config,
    )
