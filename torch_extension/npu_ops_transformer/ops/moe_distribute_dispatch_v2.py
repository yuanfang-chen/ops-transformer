# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


class MoeDistributeDispatchV2OpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDistributeDispatchV2OpBuilder, self).__init__("npu_moe_distribute_dispatch_v2")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/moe_distribute_dispatch_v2.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids, str group_ep, int ep_world_size, " \
            "int ep_rank_id, int moe_expert_num, *, Tensor? scales=None, Tensor? x_active_mask=None, " \
            "Tensor? expert_scales=None, Tensor? elastic_info=None, Tensor? performance_info=None, str group_tp=\"\", "\
            "int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, " \
            "int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, " \
            "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0, " \
            "int? y_dtype=None, int? x_dtype=None, int? scales_dtype=None) " \
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_distribute_dispatch_v2_meta(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
                                                scales, x_active_mask, expert_scales, elastic_info, performance_info,
                                                group_tp, tp_world_size, tp_rank_id, expert_shard_type,
                                                shared_expert_num, shared_expert_rank_num, quant_mode, global_bs,
                                                expert_token_nums_type, comm_alg, zero_expert_num, copy_expert_num,
                                                const_expert_num, y_dtype, x_dtype, scales_dtype):
            return torch.empty_like(x), torch.empty_like(x), torch.empty_like(x), torch.empty_like(x),\
                torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)


# Instantiate the builder
moe_distribute_dispatch_v2_op_builder = MoeDistributeDispatchV2OpBuilder()


@impl(AS_LIBRARY, moe_distribute_dispatch_v2_op_builder.name, "PrivateUse1")
def npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales=None,
                                   x_active_mask=None, expert_scales=None, elastic_info=None,
                                   performance_info=None, group_tp="", tp_world_size=0, tp_rank_id=0,
                                   expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0,
                                   quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="",
                                   zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None,
                                   x_dtype=None, scales_dtype=None):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    op_module = moe_distribute_dispatch_v2_op_builder.load()  # Compiles/loads the .so file
    return op_module.npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
                                                    scales, x_active_mask, expert_scales, elastic_info,
                                                    performance_info, group_tp, tp_world_size, tp_rank_id,
                                                    expert_shard_type, shared_expert_num, shared_expert_rank_num,
                                                    quant_mode, global_bs, expert_token_nums_type, comm_alg,
                                                    zero_expert_num, copy_expert_num, const_expert_num, y_dtype,
                                                    x_dtype, scales_dtype)
