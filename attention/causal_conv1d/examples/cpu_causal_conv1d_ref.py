"""
CPU Causal Conv1d Implementation

对齐 vllm 和 sglang 两个框架的因果卷积特性，用于 Qwen3-Next 的 Gated DeltaNet 线性注意力。

当前规划规格：所有参数都使用 channel-last 布局，dim 在尾轴且连续（SIMD 友好）。

数据布局：
- x: (batch, seqlen, dim) 或 (cu_seqlen, dim) 变长打包
- weight: (width, dim)
- bias: (dim,)
- conv_states (prefill): (num_cache_lines, width-1, dim)
- conv_state (decode): (num_cache_lines, state_len, dim)

API:
- causal_conv1d_fn: Prefill 模式，支持变长序列批处理
- causal_conv1d_update: Decode 模式，支持增量推理和状态滚动
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

# Padding slot ID，与 vllm/sglang 保持一致
PAD_SLOT_ID = -1


def _require_last_dim_contiguous(x: torch.Tensor, *, name: str) -> None:
    # Planned spec: `dim` is the last axis and must be contiguous for SIMD-friendly access.
    if x.stride(-1) != 1:
        raise ValueError(
            f"{name} must be contiguous in the last (dim) axis: stride[-1] == 1. "
            f"Got shape={tuple(x.shape)}, stride={x.stride()}"
        )


# ============================================================
# NPU SIMD-Friendly Core Primitives
# ============================================================


def _conv1d_channel_last_simd(
    x_padded: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seqlen: int,
) -> torch.Tensor:
    """
    NPU SIMD-friendly causal convolution for batched input.

    Args:
        x_padded: (batch, width-1+seqlen, dim) - padded input tensor
        weight: (width, dim) - convolution weights
        bias: (dim,) or None - optional bias
        seqlen: output sequence length

    Returns:
        out: (batch, seqlen, dim) - convolution output

    Memory access pattern (maps to NPU Vector Core):
    - Loop over kernel width (small, typically 2-6)
    - Each iteration: vector multiply-add on dim axis
    - x_slice[:, t, :] is contiguous dim-sized vector

    NPU mapping:
    - weight[k] * x_slice → Vector Multiply (VMUL)
    - out += ... → Vector Add (VADD) or fused VMLA
    """
    batch = x_padded.shape[0]
    dim = x_padded.shape[2]
    width = weight.shape[0]

    # Allocate output buffer
    out = torch.zeros(batch, seqlen, dim, dtype=x_padded.dtype, device=x_padded.device)

    # Bias: vector broadcast-add (SIMD-friendly)
    if bias is not None:
        out = out + bias  # (batch, seqlen, dim) + (dim,) → broadcast

    # Convolution: width vector multiply-adds.
    #
    # Align with vLLM/Triton (PyTorch conv1d correlation) convention:
    #   out[t] = bias + w[0]*x_ext[t-(width-1)] + ... + w[width-1]*x_ext[t]
    # With padding/initial-states placed at the left side of x_padded, this is:
    #   out += w[k] * x_padded[k : k+seqlen]
    for k in range(width):
        x_slice = x_padded[:, k:k + seqlen, :]  # (batch, seqlen, dim)
        out = out + weight[k] * x_slice

    return out


def _conv1d_channel_last_simd_1d(
    x_padded: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seqlen: int,
) -> torch.Tensor:
    """
    NPU SIMD-friendly causal convolution for single sequence (no batch dim).

    Args:
        x_padded: (padded_len, dim) - padded input tensor
        weight: (width, dim) - convolution weights
        bias: (dim,) or None - optional bias
        seqlen: output sequence length

    Returns:
        out: (seqlen, dim) - convolution output

    Used in decode mode for per-element processing.
    """
    dim = x_padded.shape[1]
    width = weight.shape[0]

    out = torch.zeros(seqlen, dim, dtype=x_padded.dtype, device=x_padded.device)

    if bias is not None:
        out = out + bias

    for k in range(width):
        x_slice = x_padded[k:k + seqlen, :]  # (seqlen, dim)
        out = out + weight[k] * x_slice

    return out


def _prepare_padded_input(
    x: torch.Tensor,
    initial_states: torch.Tensor | None,
    width: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prepare zero-padded or state-padded input for causal convolution.

    Args:
        x: (batch, seqlen, dim) - input tensor
        initial_states: (batch, width-1, dim) or None - initial states
        width: convolution kernel width
        target_dtype: target dtype for computation

    Returns:
        x_padded: (batch, width-1+seqlen, dim) - padded input

    Uses F.pad for efficient single-allocation zero padding.
    """
    x_converted = x.to(target_dtype)
    if initial_states is not None:
        return torch.cat([initial_states.to(target_dtype), x_converted], dim=1)
    # F.pad: (left_dim2, right_dim2, left_dim1, right_dim1) for 3D tensor
    return F.pad(x_converted, (0, 0, width - 1, 0), mode='constant', value=0)


def _prepare_padded_input_1d(
    x: torch.Tensor,
    state: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prepare padded input for single sequence (decode mode).

    Args:
        x: (seqlen, dim) - input tensor
        state: (state_len, dim) - state tensor
        target_dtype: target dtype for computation

    Returns:
        x_padded: (state_len+seqlen, dim) - padded input
    """
    return torch.cat([state.to(target_dtype), x.to(target_dtype)], dim=0)


def _extract_final_states(
    x: torch.Tensor,
    initial_states: torch.Tensor | None,
    width: int,
    dtype_out: torch.dtype,
) -> torch.Tensor:
    """
    Extract final (width-1) elements for state continuation.

    Args:
        x: (batch, seqlen, dim) - input tensor
        initial_states: (batch, width-1, dim) or None - initial states
        width: convolution kernel width
        dtype_out: output dtype

    Returns:
        final_states: (batch, width-1, dim) - final states for next iteration
    """
    batch, seqlen, dim = x.shape
    state_len = width - 1
    if seqlen >= state_len:
        return x[:, -state_len:, :].to(dtype_out).contiguous()
    # Short sequence: combine with initial states
    if initial_states is not None:
        combined = torch.cat([initial_states, x], dim=1)
    else:
        combined = F.pad(x, (0, 0, state_len, 0), mode='constant', value=0)
    return combined[:, -state_len:, :].to(dtype_out).contiguous()


def _apply_activation(out: torch.Tensor, activation: str | None) -> torch.Tensor:
    """
    Apply activation function.

    Args:
        out: input tensor
        activation: "silu", "swish", or None

    Returns:
        activated tensor

    Maps to NPU SILU instruction.
    """
    if activation in ["silu", "swish"]:
        return F.silu(out)
    return out


# ============================================================
# Reference Implementations
# ============================================================


def causal_conv1d_ref_channel_last(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    NPU SIMD 友好的 channel-last 因果卷积参考实现。

    数据布局 (channel_last):
        x: (batch, seqlen, dim) - dim 在最后，便于 SIMD 向量化
        weight: (width, dim) - dim 在尾轴，SIMD 向量化友好
        initial_states: (batch, width-1, dim)
        output: (batch, seqlen, dim)

    内存访问模式：
        - 每个时间步 t，访问 x[b, t, :] 是连续的 dim 个元素
        - 权重 weight[k, :] 直接是第 k 个卷积核位置的所有 dim 权重，内存连续
        - 适合 NPU Vector Core 的向量乘加指令

    Args:
        x: (batch, seqlen, dim) 输入张量
        weight: (width, dim) 卷积权重
        bias: (dim,) 可选偏置
        initial_states: (batch, width-1, dim) 初始状态
        return_final_states: 是否返回最终状态
        final_states_out: (batch, width-1, dim) 输出最终状态的缓冲区
        activation: 激活函数，None/"silu"/"swish"

    Returns:
        out: (batch, seqlen, dim) 输出张量
        final_states: (batch, width-1, dim) 最终状态（如果 return_final_states=True）
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    if x.dim() != 3:
        raise ValueError(f"x must have shape (batch, seqlen, dim), got {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must have shape (width, dim), got {tuple(weight.shape)}")
    _require_last_dim_contiguous(x, name="x")
    _require_last_dim_contiguous(weight, name="weight")
    if bias is not None:
        _require_last_dim_contiguous(bias, name="bias")
    if initial_states is not None:
        _require_last_dim_contiguous(initial_states, name="initial_states")

    dtype_in = x.dtype
    batch, seqlen, dim = x.shape
    width, dim_w = weight.shape
    if dim_w != dim:
        raise ValueError(
            f"dim mismatch: x has dim={dim}, but weight has dim={dim_w} (shape={tuple(weight.shape)})"
        )
    if bias is not None and bias.shape != (dim,):
        raise ValueError(f"bias must have shape (dim,), got {tuple(bias.shape)}")
    if initial_states is not None and initial_states.shape != (batch, width - 1, dim):
        raise ValueError(
            f"initial_states must have shape (batch, width-1, dim), got {tuple(initial_states.shape)}"
        )

    # 准备输入序列（包含初始状态）- 使用 SIMD 友好的 helper
    x_padded = _prepare_padded_input(x, initial_states, width, weight.dtype)

    # 执行因果卷积 - NPU SIMD 友好的实现
    out = _conv1d_channel_last_simd(x_padded, weight, bias, seqlen)

    # 提取最终状态
    final_states_result = None
    if return_final_states:
        final_states = _extract_final_states(x, initial_states, width, dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
            final_states_result = final_states_out
        else:
            final_states_result = final_states

    # 应用激活函数
    out = _apply_activation(out, activation)

    out = out.to(dtype_in)
    return (out, None) if not return_final_states else (out, final_states_result)


def causal_conv1d_update_ref_channel_last(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    NPU SIMD 友好的 channel-last Decode 模式参考实现。

    数据布局 (channel_last):
        x: (batch, dim) 或 (batch, seqlen, dim)
        conv_state: (batch, state_len, dim) - state_len >= width-1
        weight: (width, dim)

    Args:
        x: (batch, dim) 或 (batch, seqlen, dim) 输入张量
        conv_state: (batch, state_len, dim) 状态缓存
        weight: (width, dim) 卷积权重
        bias: (dim,) 可选偏置
        activation: 激活函数，None/"silu"/"swish"
        cache_seqlens: (batch,) int32，循环缓冲区索引（暂不支持）

    Returns:
        out: 与输入 x 形状相同的输出张量
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    if cache_seqlens is not None:
        raise NotImplementedError("cache_seqlens not supported in channel_last mode yet")

    if weight.dim() != 2:
        raise ValueError(f"weight must have shape (width, dim), got {tuple(weight.shape)}")
    if conv_state.dim() != 3:
        raise ValueError(
            f"conv_state must have shape (batch, state_len, dim), got {tuple(conv_state.shape)}"
        )
    _require_last_dim_contiguous(weight, name="weight")
    _require_last_dim_contiguous(conv_state, name="conv_state")
    if bias is not None:
        _require_last_dim_contiguous(bias, name="bias")

    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)  # (batch, 1, dim)

    if x.dim() != 3:
        raise ValueError(f"x must have shape (batch, seqlen, dim) or (batch, dim), got {tuple(x.shape)}")
    _require_last_dim_contiguous(x, name="x")

    batch, seqlen, dim = x.shape
    width = weight.shape[0]
    state_len = conv_state.shape[1]

    assert conv_state.shape == (batch, state_len, dim)
    assert weight.shape == (width, dim)
    if state_len < width - 1:
        raise ValueError(f"conv_state state_len must be >= width-1 ({width - 1}), got {state_len}")

    # 拼接状态和新输入
    x_new = torch.cat(
        [conv_state.to(weight.dtype), x.to(weight.dtype)], dim=1
    )  # (batch, state_len + seqlen, dim)

    # 更新状态（滚动窗口）
    conv_state.copy_(x_new[:, -state_len:, :].to(dtype_in))

    # 执行卷积 - NPU SIMD 友好的实现
    # x_new 的有效卷积窗口从 state_len - (width - 1) 开始
    # 截取正确的 padded 输入区域
    conv_start = state_len - (width - 1)
    x_padded = x_new[:, conv_start:, :]  # (batch, width-1+seqlen, dim)
    out = _conv1d_channel_last_simd(x_padded, weight, bias, seqlen)

    # 应用激活函数
    out = _apply_activation(out, activation)

    out = out.to(dtype_in)

    if unsqueeze:
        out = out.squeeze(1)

    return out


# Public reference helpers (channel-last layout with dim on last axis)
causal_conv1d_ref = causal_conv1d_ref_channel_last
causal_conv1d_update_ref = causal_conv1d_update_ref_channel_last


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """
    Prefill 模式因果卷积，支持变长序列批处理和连续批处理。

    Args:
        x: 输入张量
            - (batch, seqlen, dim) 固定长度
            - (cu_seqlen, dim) 变长序列打包
        weight: (width, dim) 卷积权重
        bias: (dim,) 可选偏置
        conv_states: 状态缓存
            - (num_cache_lines, width-1, dim)
        query_start_loc: (batch+1,) int32 累积序列长度
        cache_indices: (batch,) int32 状态索引
        has_initial_state: (batch,) bool 是否使用初始状态
        activation: 激活函数，None/"silu"/"swish"
        pad_slot_id: padding 标识，默认 -1

    Returns:
        out: 与输入 x 形状相同的输出张量
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # NPU SIMD 友好的 channel-last 模式
    return _causal_conv1d_fn_channel_last(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
    )


def _causal_conv1d_fn_channel_last(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
    pad_slot_id: int,
) -> torch.Tensor:
    """
    channel_last 模式的内部实现。

    数据布局：
        x: (batch, seqlen, dim) 或 (cu_seqlen, dim)
        conv_states: (num_cache_lines, width-1, dim)
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must have shape (width, dim), got {tuple(weight.shape)}")
    width, dim = weight.shape
    _require_last_dim_contiguous(weight, name="weight")
    if bias is not None:
        _require_last_dim_contiguous(bias, name="bias")
    if x.dim() not in (2, 3):
        raise ValueError(f"x must have shape (batch, seqlen, dim) or (cu_seqlen, dim), got {tuple(x.shape)}")
    _require_last_dim_contiguous(x, name="x")
    if x.shape[-1] != dim:
        raise ValueError(f"dim mismatch: x has dim={x.shape[-1]}, but weight has dim={dim}")
    if conv_states is not None:
        if conv_states.dim() != 3:
            raise ValueError(
                f"conv_states must have shape (num_cache_lines, width-1, dim), got {tuple(conv_states.shape)}"
            )
        if conv_states.shape[-1] != dim or conv_states.shape[-2] != width - 1:
            raise ValueError(
                f"conv_states must have shape (num_cache_lines, width-1, dim), got {tuple(conv_states.shape)}"
            )
        _require_last_dim_contiguous(conv_states, name="conv_states")
    is_varlen = query_start_loc is not None

    if not is_varlen:
        # 固定长度模式：x 是 (batch, seqlen, dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch = x.shape[0]

        if conv_states is None:
            # 无状态模式
            out, _ = causal_conv1d_ref_channel_last(x, weight, bias, activation=activation)
            return out

        # 有状态模式
        outputs = []
        for i in range(batch):
            cache_idx = cache_indices[i].item() if cache_indices is not None else i
            if cache_idx == pad_slot_id:
                # 跳过 padding，但仍需要输出占位
                outputs.append(torch.zeros_like(x[i]))
                continue

            use_initial = (
                has_initial_state[i].item()
                if has_initial_state is not None
                else False
            )
            # conv_states: (num_cache_lines, width-1, dim)
            initial = conv_states[cache_idx].unsqueeze(0) if use_initial else None

            out_i, final_state = causal_conv1d_ref_channel_last(
                x[i].unsqueeze(0),
                weight,
                bias,
                initial_states=initial,
                return_final_states=True,
                activation=activation,
            )
            outputs.append(out_i.squeeze(0))

            # 更新状态
            if final_state is not None:
                conv_states[cache_idx].copy_(final_state.squeeze(0))

        return torch.stack(outputs, dim=0)

    # 变长序列模式：x 是 (cu_seqlen, dim)
    batch = query_start_loc.shape[0] - 1
    dtype_in = x.dtype
    outputs = []

    for i in range(batch):
        cache_idx = cache_indices[i].item() if cache_indices is not None else i
        if cache_idx == pad_slot_id:
            continue

        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        seq_x = x[start:end, :].unsqueeze(0)  # (1, seqlen, dim)

        use_initial = (
            has_initial_state[i].item() if has_initial_state is not None else False
        )
        initial = conv_states[cache_idx].unsqueeze(0) if use_initial else None

        out_i, final_state = causal_conv1d_ref_channel_last(
            seq_x,
            weight,
            bias,
            initial_states=initial,
            return_final_states=True,
            activation=activation,
        )
        outputs.append(out_i.squeeze(0))

        # 更新状态
        if conv_states is not None and final_state is not None:
            conv_states[cache_idx].copy_(final_state.squeeze(0))

    return torch.cat(outputs, dim=0)  # (cu_seqlen, dim)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """
    Decode 模式因果卷积，支持单 token 或少量 token 的增量推理。

    Args:
        x: 输入张量
            - (batch, dim) 或 (batch, seqlen, dim)
        conv_state: 状态缓存
            - (num_cache_lines, state_len, dim)
        weight: (width, dim) 卷积权重
        bias: (dim,) 可选偏置
        activation: 激活函数，None/"silu"/"swish"
        conv_state_indices: (batch,) int32 状态索引
        pad_slot_id: padding 标识，默认 -1

    Returns:
        out: 与输入 x 形状相同的输出张量
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    return _causal_conv1d_update_channel_last(
        x, conv_state, weight, bias, activation, conv_state_indices, pad_slot_id
    )


def _causal_conv1d_update_channel_last(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
    pad_slot_id: int,
) -> torch.Tensor:
    """
    channel_last 模式的 decode 内部实现。

    数据布局：
        x: (batch, dim) 或 (batch, seqlen, dim)
        conv_state: (num_cache_lines, state_len, dim)
    """
    dtype_in = x.dtype
    if weight.dim() != 2:
        raise ValueError(f"weight must have shape (width, dim), got {tuple(weight.shape)}")
    width, dim = weight.shape
    _require_last_dim_contiguous(weight, name="weight")
    if bias is not None:
        _require_last_dim_contiguous(bias, name="bias")

    # 处理输入维度
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)  # (batch, 1, dim)

    if x.dim() != 3:
        raise ValueError(f"x must have shape (batch, seqlen, dim) or (batch, dim), got {tuple(x.shape)}")
    _require_last_dim_contiguous(x, name="x")

    batch, seqlen, dim_x = x.shape
    assert dim_x == dim, f"dim mismatch: x has {dim_x}, weight has {dim}"

    if conv_state.dim() != 3:
        raise ValueError(
            f"conv_state must have shape (num_cache_lines, state_len, dim), got {tuple(conv_state.shape)}"
        )
    if conv_state.shape[-1] != dim:
        raise ValueError(f"dim mismatch: conv_state has dim={conv_state.shape[-1]}, weight has dim={dim}")
    _require_last_dim_contiguous(conv_state, name="conv_state")

    state_len = conv_state.shape[1]
    if state_len < width - 1:
        raise ValueError(f"conv_state state_len must be >= width-1 ({width - 1}), got {state_len}")

    outputs = []
    valid_indices = []

    for i in range(batch):
        idx = conv_state_indices[i].item() if conv_state_indices is not None else i
        if idx == pad_slot_id:
            outputs.append(None)
            continue

        valid_indices.append(i)

        # 获取当前状态: (state_len, dim)
        state_i = conv_state[idx]

        # 拼接状态和新输入: (state_len + seqlen, dim)
        x_new = _prepare_padded_input_1d(x[i], state_i, weight.dtype)

        # 更新状态（滚动窗口）
        conv_state[idx].copy_(x_new[-state_len:, :].to(dtype_in))

        # 执行卷积 - NPU SIMD 友好的实现
        conv_start = state_len - (width - 1)
        x_padded = x_new[conv_start:, :]  # (width-1+seqlen, dim)
        out_i = _conv1d_channel_last_simd_1d(x_padded, weight, bias, seqlen)

        # 应用激活函数
        out_i = _apply_activation(out_i, activation)

        outputs.append(out_i.to(dtype_in))

    # 处理输出
    if len(valid_indices) == batch:
        result = torch.stack(outputs, dim=0)
    else:
        result = torch.zeros(batch, seqlen, dim, dtype=dtype_in, device=x.device)
        for i, out in enumerate(outputs):
            if out is not None:
                result[i] = out

    if unsqueeze:
        result = result.squeeze(1)

    return result


# 便捷别名
causal_conv1d = causal_conv1d_fn


def _run_saved_case() -> int:
    import numpy as np

    F32 = np.dtype("<f4")
    I32 = np.dtype("<i4")
    U8 = np.dtype("u1")

    def read_meta_txt(path: Path) -> Dict[str, int]:
        meta: Dict[str, int] = {}
        if not path.exists():
            return meta
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = "".join(ch for ch in raw if not ch.isspace())
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                meta[k] = int(v)
            except ValueError:
                continue
        return meta

    def load_bin(path: Path, dtype: np.dtype, shape) -> np.ndarray:
        arr = np.fromfile(str(path), dtype=dtype)
        return arr.reshape(shape)

    def save_bin(path: Path, arr: np.ndarray) -> None:
        arr.astype(F32, copy=False).tofile(str(path))

    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    q = ap.add_mutually_exclusive_group()
    q.add_argument(
        "--quantize_fp16",
        dest="quantize_fp16",
        action="store_true",
        help="Quantize outputs to fp16 before saving (default).",
    )
    q.add_argument(
        "--no_quantize_fp16",
        dest="quantize_fp16",
        action="store_false",
        help="Do not fp16-quantize outputs before saving.",
    )
    ap.set_defaults(quantize_fp16=True)
    args = ap.parse_args()

    case_dir: Path = args.case_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_meta_txt(case_dir / "meta.txt")
    batch_size = int(meta.get("batch_size", 32))
    total_seqlen = int(meta.get("total_seqlen", 8192))
    dim = int(meta.get("dim", 1024))
    width = int(meta.get("width", 4))
    activation_mode = int(meta.get("activation_mode", 1))
    pad_slot_id = int(meta.get("pad_slot_id", -1))

    activation = "silu" if activation_mode != 0 else None

    x = load_bin(case_dir / "x.bin", F32, (total_seqlen, dim))
    weight = load_bin(case_dir / "weight.bin", F32, (width, dim))
    bias = load_bin(case_dir / "bias.bin", F32, (dim,))
    conv_states_in = load_bin(case_dir / "conv_states.bin", F32, (batch_size, width - 1, dim))
    qsl = load_bin(case_dir / "query_start_loc.bin", I32, (batch_size + 1,))
    cache_indices = load_bin(case_dir / "cache_indices.bin", I32, (batch_size,))
    has_initial_state = load_bin(case_dir / "has_initial_state.bin", U8, (batch_size,))

    device = torch.device("cpu")

    # Match kernel semantics: FP16 IO + FP32 compute.
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float16).to(torch.float32)
    w_t = torch.from_numpy(weight).to(device=device, dtype=torch.float16).to(torch.float32)
    b_t = torch.from_numpy(bias).to(device=device, dtype=torch.float16).to(torch.float32)
    s_t = torch.from_numpy(conv_states_in).to(device=device, dtype=torch.float16).to(torch.float32).contiguous()
    qsl_t = torch.from_numpy(qsl).to(device=device)
    cache_indices_t = torch.from_numpy(cache_indices).to(device=device)
    has_initial_state_t = torch.from_numpy(has_initial_state).to(device=device)

    y_t = torch.zeros((total_seqlen, dim), device=device, dtype=torch.float32)

    for seq in range(batch_size):
        start = int(qsl_t[seq].item())
        end = int(qsl_t[seq + 1].item())
        length = end - start
        if length <= 0:
            continue

        cache_idx = int(cache_indices_t[seq].item())
        if cache_idx == pad_slot_id:
            continue

        has_init = bool(int(has_initial_state_t[seq].item()) != 0)
        initial = s_t[cache_idx].unsqueeze(0) if has_init else None

        seq_x = x_t[start:end, :].unsqueeze(0)  # (1, seqlen, dim)
        out_i, final_state = causal_conv1d_ref_channel_last(
            seq_x,
            w_t,
            b_t,
            initial_states=initial,
            return_final_states=True,
            activation=activation,
        )
        y_t[start:end, :] = out_i.squeeze(0)
        if final_state is not None:
            s_t[cache_idx].copy_(final_state.squeeze(0))

    quantize_fp16 = bool(args.quantize_fp16)
    if quantize_fp16:
        y_save = y_t.to(torch.float16).to(torch.float32)
        s_save = s_t.to(torch.float16).to(torch.float32)
    else:
        y_save = y_t
        s_save = s_t

    y_np = y_save.cpu().numpy().astype(F32, copy=False)
    s_np = s_save.cpu().numpy().astype(F32, copy=False)
    save_bin(out_dir / "y_ref_torch.bin", y_np)
    save_bin(out_dir / "conv_states_ref_torch.bin", s_np)

    print(f"Wrote CPU reference outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_saved_case())
