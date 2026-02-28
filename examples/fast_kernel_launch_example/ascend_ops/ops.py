#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
from typing import List, Optional, Tuple 
import torch
from torch import Tensor

__all__ = ["groupedmatmul", "MoeDistributeDispatchV2", ]

def groupedmatmul(
    x: List[Tensor],
    weight: List[Tensor],
    bias: Optional[List[Tensor]] = None,
    scale: Optional[List[Tensor]] = None,
    offset: Optional[List[Tensor]] = None,
    antiquant_scale: Optional[List[Tensor]] = None,
    antiquant_offset: Optional[List[Tensor]] = None,
    group_list: Optional[Tensor] = None,
    per_token_scale: Optional[List[Tensor]] = None,
    split_item: int = 8,
    group_type: int = 0,
    group_list_type: int = 0,
    act_type: int = 0,
    tuning_config: Optional[List[int]] = None
) -> Tensor:
    """
    分组矩阵乘法
    
    Args:
        x: 输入张量列表（必填）
        weight: 权重张量列表（必填）
        bias: 偏置张量列表（可选）
        scale: 缩放因子列表（可选）
        offset: 偏移量列表（可选）
        antiquant_scale: 反量化缩放列表（可选）
        antiquant_offset: 反量化偏移列表（可选）
        group_list: 分组列表（可选）
        per_token_scale: 每 token 缩放因子列表（可选）
        split_item: 拆分项，默认为 8
        group_type: 分组类型，默认为 0
        group_list_type: 分组列表类型，默认为 0
        act_type: 激活类型，默认为 0
        tuning_config: 调优配置，默认为 None
    
    Returns:
        Tensor: 计算结果
    """
    return torch.ops.ascend_ops.groupedmatmul(
        x, weight, bias, scale, offset,
        antiquant_scale, antiquant_offset,
        group_list, per_token_scale,
        split_item, group_type, group_list_type, act_type,
        tuning_config
    )

def MoeDistributeDispatchV2(
    x: Tensor,
    expert_ids: Tensor,
    scales: Optional[Tensor] = None,
    x_active_mask: Optional[Tensor] = None,
    expert_scales: Optional[Tensor] = None,
    elastic_info: Optional[Tensor] = None,
    performance_info: Optional[Tensor] = None,
    group_ep: str = "",
    ep_world_size: int = 0,
    ep_rank_id: int = 0,
    moe_expert_num: int = 0,
    total_winsize_ep: int = 0,
    group_tp: str = "",
    tp_world_size: int = 0,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    global_bs: int = 0,
    expert_token_nums_type: int = 1,
    comm_alg: str = "",
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    分组矩阵乘法
    
    Args:
        x: 输入张量列表（必填）
        weight: 权重张量列表（必填）
        bias: 偏置张量列表（可选）
        scale: 缩放因子列表（可选）
        offset: 偏移量列表（可选）
        antiquant_scale: 反量化缩放列表（可选）
        antiquant_offset: 反量化偏移列表（可选）
        group_list: 分组列表（可选）
        per_token_scale: 每 token 缩放因子列表（可选）
        split_item: 拆分项，默认为 8
        group_type: 分组类型，默认为 0
        group_list_type: 分组列表类型，默认为 0
        act_type: 激活类型，默认为 0
        tuning_config: 调优配置，默认为 None
    
    Returns:
        Tensor: 计算结果
    """
    return torch.ops.ascend_ops.MoeDistributeDispatchV2(
        x,
        expert_ids,
        group_ep,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        total_winsize_ep,
        scales,
        x_active_mask,
        expert_scales,
        elastic_info,
        performance_info,
        group_tp,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        expert_token_nums_type,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num
    )
