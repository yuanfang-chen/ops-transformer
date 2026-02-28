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
@register_fx_node_ge_converter(torch.ops.custom.npu_fused_infer_attention_score.default)
def convert_npu_fused_infer_attention_score(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    pse_shift: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_qlen: Optional[Union[List[int], Tensor]] = None,
    actual_seq_kvlen: Optional[Union[List[int], Tensor]] = None,
    block_table: Optional[Tensor] = None,
    dequant_scale_query: Optional[Tensor] = None,
    dequant_scale_key: Optional[Tensor] = None,
    dequant_offset_key: Optional[Tensor] = None,
    dequant_scale_value: Optional[Tensor] = None,
    dequant_offset_value: Optional[Tensor] = None,
    dequant_scale_key_rope: Optional[Tensor] = None,
    quant_scale_out: Optional[Tensor] = None,
    quant_offset_out: Optional[Tensor] = None,
    learnable_sink: Optional[Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
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
    query_dtype: Optional[int] = None,
    key_dtype: Optional[int] = None,
    value_dtype: Optional[int] = None,
    query_rope_dtype: Optional[int] = None,
    key_rope_dtype: Optional[int] = None,
    key_shared_prefix_dtype: Optional[int] = None,
    value_shared_prefix_dtype: Optional[int] = None,
    dequant_scale_query_dtype: Optional[int] = None,
    dequant_scale_key_dtype: Optional[int] = None,
    dequant_scale_value_dtype: Optional[int] = None,
    dequant_scale_key_rope_dtype: Optional[int] = None,
    out_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,):

    print("convert_npu_fused_infer_attention_score")
    def to_ge_int_tensor(x, name):
        if x is None:
            return None
        # x 是 list[int]
        vals = [int(v) for v in x]
        # 下面函数名可能是 Const / const / make_const，请按你torchair版本替换
        return torchair.ge.Const(vals, dtype=DataType.DT_INT64, node_name=name)
    if isinstance(actual_seq_qlen, list):
        actual_seq_qlen = to_ge_int_tensor(actual_seq_qlen, "actual_seq_qlen_const")
    if isinstance(actual_seq_kvlen, list):
        actual_seq_kvlen = to_ge_int_tensor(actual_seq_kvlen, "actual_seq_kvlen_const")

    # dropped params
    quant_scale1 = None
    dequant_scale2 = None
    dequant_scale1 = None
    antiquant_scale = None
    antiquant_offset = None
    query_padding_size = None
    kv_padding_size = None
    key_shared_prefix = None
    value_shared_prefix = None
    actual_shared_prefix_len = None
    q_start_idx=None
    kv_start_idx=None
    # 1) Tensor inputs only
    inputs = {
        "query": query,
        "key": [key],
        "value": [value],
        "pse_shift": pse_shift,
        "atten_mask": atten_mask,
        "actual_seq_lengths": actual_seq_qlen,
        "actual_seq_lengths_kv": actual_seq_kvlen,
        "dequant_scale1": dequant_scale1,
        "quant_scale1": quant_scale1,
        "dequant_scale2": dequant_scale2,
        "quant_scale2": quant_scale_out,
        "quant_offset2": quant_offset_out,
        "antiquant_scale": antiquant_scale,
        "antiquant_offset": antiquant_offset,
        "block_table": block_table,
        "query_padding_size": query_padding_size,
        "kv_padding_size": kv_padding_size,
        "key_antiquant_scale": dequant_scale_key,
        "key_antiquant_offset": dequant_offset_key,
        "value_antiquant_scale": dequant_scale_value,
        "value_antiquant_offset": dequant_offset_value,
        "key_shared_prefix": key_shared_prefix,
        "value_shared_prefix": value_shared_prefix,
        "actual_shared_prefix_len": actual_shared_prefix_len,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "key_rope_antiquant_scale": dequant_scale_key_rope,
        "dequant_scale_query": dequant_scale_query,
        "learnable_sink": learnable_sink,
        "q_start_idx": q_start_idx,
        "kv_start_idx": kv_start_idx,      
    }

    # inputs = {k: v for k, v in inputs.items() if v is not None}
    # 2) Required attrs
    attrs = {
        "num_heads": attr.Int(num_query_heads),
        "scale": attr.Float(softmax_scale),
        "pre_tokens": attr.Int(pre_tokens),
        "next_tokens": attr.Int(next_tokens),
        "input_layout": attr.Str(input_layout),
        "num_key_value_heads": attr.Int(num_key_value_heads),
        "sparse_mode": attr.Int(sparse_mode),
        "inner_precise": attr.Int(inner_precise),
        "block_size": attr.Int(block_size),
        "antiquant_mode": attr.Int(0),
        "softmax_lse_flag": attr.Bool(return_softmax_lse),
        "query_quant_mode": attr.Int(query_quant_mode),
        "key_quant_mode": attr.Int(key_quant_mode),
        "value_quant_mode": attr.Int(value_quant_mode),
        "key_antiquant_mode": attr.Int(0),
        "value_antiquant_mode": attr.Int(0),
        "query_quant_mode": attr.Int(0),
        "pse_type": attr.Int(0),
        "out_dtype": attr.Int(0),
    }
    # # 3) Optional int attrs (only when not None)
    # def add_opt_int(name, v):
    #     if v is not None:
    #         attrs[name] = attr.Int(int(v))
    # add_opt_int("query_dtype", query_dtype)
    # add_opt_int("key_dtype", key_dtype)
    # add_opt_int("value_dtype", value_dtype)
    # add_opt_int("query_rope_dtype", query_rope_dtype)
    # add_opt_int("key_rope_dtype", key_rope_dtype)
    # add_opt_int("key_shared_prefix_dtype", key_shared_prefix_dtype)
    # add_opt_int("value_shared_prefix_dtype", value_shared_prefix_dtype)
    # add_opt_int("dequant_scale_query_dtype", dequant_scale_query_dtype)
    # add_opt_int("dequant_scale_key_dtype", dequant_scale_key_dtype)
    # add_opt_int("dequant_scale_value_dtype", dequant_scale_value_dtype)
    # add_opt_int("dequant_scale_key_rope_dtype", dequant_scale_key_rope_dtype)

    print("end convert_npu_fused_infer_attention_score")
    return torchair.ge.custom_op(
        "FusedInferAttentionScore",
        inputs=inputs,
        attrs=attrs,
        outputs=["attention_out", "softmax_lse"],
    )