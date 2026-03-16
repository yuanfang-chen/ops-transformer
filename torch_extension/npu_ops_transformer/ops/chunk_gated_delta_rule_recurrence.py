# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PTA interface for ChunkGatedDeltaRuleRecurrence.

Algorithm (per chunk i, for each (batch, head) task):
    v_prime          = k_cumdecay[h, i] @ state[b, h].T    # [cs,dk]@[dk,dv] = [cs,dv]
    attn_inter_out   = qgexp[h, i]      @ state[b, h].T    # [cs,dk]@[dk,dv] = [cs,dv]
    v_new_out        = value[h, i] - v_prime                # element-wise
    state[b, h]     *= gexp[h, i, -1, 0]                   # scalar decay
    state[b, h]     += v_new_out[h,i].T @ kgexp[h, i]     # [dv,cs]@[cs,dk] = [dv,dk]

Tensor layouts:
    initial_state  : float32  [b,  hv, dv, dk]   (modified in-place)
    kgexp          : float32  [hv, nC, cs, dk]
    value          : float32  [hv, nC, cs, dv]
    k_cumdecay     : float32  [hv, nC, cs, dk]
    qgexp          : float32  [hv, nC, cs, dk]
    gexp           : float32  [hv, nC, cs,  1]
    cu_seqlens     : int32    [b+1]
    attn_inter_out : float32  [hv, nC, cs, dv]   (output)
    v_new_out      : float32  [hv, nC, cs, dv]   (output)
"""

import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class ChunkGatedDeltaRuleRecurrenceOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_chunk_gated_delta_rule_recurrence")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/chunk_gated_delta_rule_recurrence.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "npu_chunk_gated_delta_rule_recurrence("
            "Tensor initial_state, "
            "Tensor kgexp, "
            "Tensor value, "
            "Tensor k_cumdecay, "
            "Tensor qgexp, "
            "Tensor gexp, "
            "Tensor cu_seqlens, "
            "float scale_value=1.0"
            ") -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_chunk_gated_delta_rule_recurrence_meta(
            initial_state, kgexp, value, k_cumdecay, qgexp, gexp, cu_seqlens,
            scale_value=1.0
        ):
            # initial_state [b, hv, dv, dk] → returned unchanged (modified in-place on device)
            # attn_inter_out [hv, nC, cs, dv] — same shape as value
            # v_new_out      [hv, nC, cs, dv] — same shape as value
            attn_inter_out = value.new_empty(value.shape)
            v_new_out = value.new_empty(value.shape)
            return initial_state, attn_inter_out, v_new_out


# Instantiate builder (registers schema + meta at import time)
chunk_gated_delta_rule_recurrence_op_builder = ChunkGatedDeltaRuleRecurrenceOpBuilder()
op_module = chunk_gated_delta_rule_recurrence_op_builder.load()


@impl(AS_LIBRARY, chunk_gated_delta_rule_recurrence_op_builder.name, "PrivateUse1")
def npu_chunk_gated_delta_rule_recurrence(
    initial_state, kgexp, value, k_cumdecay, qgexp, gexp, cu_seqlens,
    scale_value=1.0
):
    """
    Dispatcher implementation for NPU (PrivateUse1 = Ascend NPU backend).

    Args:
        initial_state (Tensor): float32 [b, hv, dv, dk]. Modified in-place.
        kgexp         (Tensor): float32 [hv, nChunks, cs, dk].
        value         (Tensor): float32 [hv, nChunks, cs, dv].
        k_cumdecay    (Tensor): float32 [hv, nChunks, cs, dk].
        qgexp         (Tensor): float32 [hv, nChunks, cs, dk].
        gexp          (Tensor): float32 [hv, nChunks, cs, 1].
        cu_seqlens    (Tensor): int32   [b+1].
        scale_value   (float):  default 1.0.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            initial_state  [b,  hv, dv, dk]  (in-place updated state)
            attn_inter_out [hv, nC, cs, dv]
            v_new_out      [hv, nC, cs, dv]
    """
    return op_module.npu_chunk_gated_delta_rule_recurrence(
        initial_state, kgexp, value, k_cumdecay, qgexp, gexp, cu_seqlens,
        scale_value
    )
