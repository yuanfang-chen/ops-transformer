# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图

# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_incre_flash_attention.default)
def convert_npu_incre_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    padding_mask: Tensor = None, 
    atten_mask: Tensor = None, 
    pse_shift: Tensor = None, 
    actual_seq_lengths: Tensor = None,  # 不确定是tensor还是array
    antiquant_scale: Tensor = None,
    antiquant_offset: Tensor = None, 
    block_table: Tensor = None, 
    dequant_scale1: Tensor = None, 
    quant_scale1: Tensor = None, 
    dequant_scale2: Tensor = None, 
    quant_scale2: Tensor = None, 
    quant_offset2: Tensor = None, 
    kv_padding_size: Tensor = None,
    num_heads: int = 1, 
    scale_value: float = 1.0, 
    input_layout: str= "BSH", 
    num_key_value_heads: int = 0, 
    block_size: int = 0, 
    inner_precise: int = 1,
    output: Tensor = None):
    return torchair.ge.custom_op(
        "IncreFlashAttention",
        inputs={"query": query,
                "key": key,
                "value": value,
                "padding_mask": padding_mask,
                "pse_shift": pse_shift,
                "actual_seq_lengths": actual_seq_lengths,
                "antiquant_scale": antiquant_scale,
                "antiquant_offset": antiquant_offset,
                "block_table": block_table,
                "dequant_scale1": dequant_scale1,
                "quant_scale1": quant_scale1,
                "dequant_scale2": dequant_scale2,
                "quant_scale2": quant_scale2,
                "quant_offset2": quant_offset2,
                "kv_padding_size": kv_padding_size,
                },
        attrs={"num_heads": attr.Int(num_heads),
               "scale_value": attr.Float(scale_value),
               "input_layout": attr.Str(input_layout),
               "num_key_value_heads": attr.Int(num_key_value_heads),
               "block_size": attr.Int(block_size),
               "inner_precise": attr.Int(inner_precise),
               },
        outputs=['attention_out']
    )
