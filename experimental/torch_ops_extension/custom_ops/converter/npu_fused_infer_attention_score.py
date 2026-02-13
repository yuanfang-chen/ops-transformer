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
@register_fx_node_ge_converter(torch.ops.custom.npu_fused_infer_attention_score.default)
def convert_npu_fused_infer_attention_score(  query: Tensor, 
                                        key: Tensor, 
                                        value: Tensor, 
                                        *, 
                                        query_rope: Tensor = None, 
                                        key_rope: Tensor = None, 
                                        pse_shift: Tensor= None, 
                                        atten_mask: Tensor = None, 
                                        actual_seq_qlen: Tensor = None, #不确定是tensor还是array
                                        actual_seq_kvlen: Tensor = None, #不确定是tensor还是array
                                        block_table: Tensor = None, 
                                        dequant_scale_query: Tensor = None,
                                        dequant_scale_key: Tensor = None, 
                                        dequant_offset_key: Tensor = None,
                                        dequant_scale_value: Tensor = None, 
                                        dequant_offset_value: Tensor = None, 
                                        dequant_scale_key_rope: Tensor = None, 
                                        quant_scale_out: Tensor = None, 
                                        quant_offset_out: Tensor = None, 
                                        learnable_sink: Tensor = None, 
                                        num_query_heads: int = 1, 
                                        num_key_value_heads: int = 1, 
                                        softmax_scale: float = 1.0, 
                                        pre_tokens: int = 2147483647, 
                                        next_tokens: int = 2147483647,  
                                        input_layout: str = "BSH", 
                                        sparse_mode: int = 0, 
                                        block_size: int = 0, 
                                        query_quant_mode: int = 0, 
                                        key_quant_mode: int = 0, 
                                        value_quant_mode: int = 0,
                                        inner_precise: int = 0, 
                                        return_softmax_lse: bool = False, 
                                        query_dtype: int = None, 
                                        key_dtype: int = None, 
                                        value_dtype: int = None, 
                                        query_rope_dtype: int = None, 
                                        key_rope_dtype: int = None, 
                                        key_shared_prefix_dtype: int = None, 
                                        value_shared_prefix_dtype: int = None, 
                                        dequant_scale_query_dtype: int = None, 
                                        dequant_scale_key_dtype: int = None, 
                                        dequant_scale_value_dtype: int = None, 
                                        dequant_scale_key_rope_dtype: int = None):
    return torchair.ge.custom_op(
        "FusedInferAttentionScore",
        inputs={"query": query,
                "key": key,
                "value": value,
                "query_rope": query_rope,
                "key_rope": key_rope,
                "pse_shift": pse_shift,
                "atten_mask": atten_mask,
                "actual_seq_qlen": actual_seq_qlen,
                "actual_seq_kvlen": actual_seq_kvlen,
                "block_table": block_table,
                "dequant_scale_query": dequant_scale_query,
                "dequant_scale_key": dequant_scale_key,
                "dequant_offset_key": dequant_offset_key,
                "dequant_scale_value": dequant_scale_value,
                "dequant_offset_value": dequant_offset_value,
                "dequant_scale_key_rope": dequant_scale_key_rope,
                "quant_scale_out": quant_scale_out,
                "dequant_scale_key": dequant_scale_key,
                "quant_offset_out": quant_offset_out,
                "learnable_sink": learnable_sink,
                },
        attrs={"num_query_heads": attr.Int(num_query_heads),
               "num_key_value_heads": attr.Int(num_key_value_heads),
               "softmax_scale": attr.Float(softmax_scale),
               "pre_tokens": attr.Int(pre_tokens),
               "next_tokens": attr.Int(next_tokens),
               "input_layout": attr.Str(input_layout),
               "sparse_mode": attr.Int(sparse_mode),
               "block_size": attr.Int(block_size),
               "query_quant_mode": attr.Int(query_quant_mode),
               "key_quant_mode": attr.Int(key_quant_mode),
               "value_quant_mode": attr.Int(value_quant_mode),
               "inner_precise": attr.Int(inner_precise),
               "return_softmax_lse": attr.Bool(return_softmax_lse),
               "query_dtype": attr.Int(query_dtype),
               "key_dtype": attr.Int(key_dtype),
               "value_dtype": attr.Int(value_dtype),
               "query_rope_dtype": attr.Int(query_rope_dtype),
               "key_rope_dtype": attr.Int(key_rope_dtype),
               "key_shared_prefix_dtype": attr.Int(key_shared_prefix_dtype),
               "value_shared_prefix_dtype": attr.Int(value_shared_prefix_dtype),
               "dequant_scale_query_dtype": attr.Int(dequant_scale_query_dtype),
               "dequant_scale_key_dtype": attr.Int(dequant_scale_key_dtype),
               "dequant_scale_value_dtype": attr.Int(dequant_scale_value_dtype),
               "dequant_scale_key_rope_dtype": attr.Int(dequant_scale_key_rope_dtype),
               },
        outputs=['attention_out', 'softmax_lse']
    )
