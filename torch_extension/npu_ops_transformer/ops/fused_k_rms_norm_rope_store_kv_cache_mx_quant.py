# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


QUANT_BLOCK_SIZE = 32
DIGIT_TWO = 2


class FusedKRmsNormRopeStoreKvCacheMxQuantOpBuilder(OpBuilder):
    def __init__(self):
        super(FusedKRmsNormRopeStoreKvCacheMxQuantOpBuilder, self).__init__(
            "npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant"
        )

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/fused_k_rms_norm_rope_store_kv_cache_mx_quant.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant("
            "Tensor qkv, Tensor cos, Tensor sin, Tensor gamma, "
            "Tensor kv_slot_mapping, Tensor v_scale_slot_mapping, "
            "Tensor(a!) k_cache, Tensor(b!) k_scale_cache, "
            "Tensor(c!) v_cache, Tensor(d!) v_scale_cache, "
            "*, float epsilon=1e-5"
            ") -> (Tensor, Tensor, Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant_meta(
            qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
            k_cache, k_scale_cache, v_cache, v_scale_cache, epsilon=1e-5
        ):
            torch._check(
                qkv.dim() == 3,
                lambda: f"qkv must be 3D tensor [T, N, D], got {qkv.dim()}D.",
            )
            torch._check(
                cos.dim() == 3,
                lambda: f"cos must be 3D tensor [T, 1, D], got {cos.dim()}D.",
            )
            torch._check(
                sin.dim() == 3,
                lambda: f"sin must be 3D tensor [T, 1, D], got {sin.dim()}D.",
            )
            torch._check(
                gamma.dim() == 1,
                lambda: f"gamma must be 1D tensor [D], got {gamma.dim()}D.",
            )
            torch._check(
                k_cache.dim() == 4,
                lambda: f"k_cache must be 4D tensor [Bn, Nk, Bs, D], got {k_cache.dim()}D.",
            )
            torch._check(
                k_scale_cache.dim() == 5,
                lambda: f"k_scale_cache must be 5D tensor [Bn, Nk, Bs, D/32/2, 2], got {k_scale_cache.dim()}D.",
            )
            torch._check(
                v_cache.dim() == 4,
                lambda: f"v_cache must be 4D tensor [Bn, Nv, Bs, D], got {v_cache.dim()}D.",
            )
            torch._check(
                v_scale_cache.dim() == 5,
                lambda: f"v_scale_cache must be 5D tensor [Bn, Nv, Bs/32/2, D, 2], got {v_scale_cache.dim()}D.",
            )

            T = qkv.size(0)
            N = qkv.size(1)
            D = qkv.size(2)
            Nk = k_cache.size(1)
            Nv = v_cache.size(1)
            Nq = N - Nk - Nv

            torch._check(
                Nq > 0,
                lambda: f"Nq must be positive, N={N}, Nk={Nk}, Nv={Nv}.",
            )

            # q output: [T, Nq, D], dtype same as k_cache (FP8_E4M3FN)
            q = qkv.new_empty((T, Nq, D), dtype=k_cache.dtype)
            # q_scale output: [T, Nq, D//32//2, 2], dtype same as k_scale_cache (FP8_E8M0)
            q_scale = qkv.new_empty(
                (T, Nq, D // QUANT_BLOCK_SIZE // DIGIT_TWO, DIGIT_TWO),
                dtype=k_scale_cache.dtype,
            )
            # k_cache, k_scale_cache, v_cache, v_scale_cache are inplace updated
            return (q, q_scale, k_cache, k_scale_cache, v_cache, v_scale_cache)


# Instantiate the builder
fused_k_rms_norm_rope_store_kv_cache_mx_quant_op_builder = FusedKRmsNormRopeStoreKvCacheMxQuantOpBuilder()
op_module = fused_k_rms_norm_rope_store_kv_cache_mx_quant_op_builder.load()


@impl(AS_LIBRARY, fused_k_rms_norm_rope_store_kv_cache_mx_quant_op_builder.name, "PrivateUse1")
def npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant(
    qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
    k_cache, k_scale_cache, v_cache, v_scale_cache, epsilon=1e-5
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant(
        qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
        k_cache, k_scale_cache, v_cache, v_scale_cache, epsilon
    )
