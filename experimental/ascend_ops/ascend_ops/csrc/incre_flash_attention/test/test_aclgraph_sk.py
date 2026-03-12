#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch_npu
import torchair as tng
import torchair._contrib.custom_torch_ops
from torchair.configs.compiler_config import CompilerConfig

import ascend_ops


class ModelOrigin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        value,
        actual_seq_kvlen,
        block_table,
        dequant_scale_key,
        dequant_scale_value,
    ):
        return torch.ops.custom.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            actual_seq_kvlen=actual_seq_kvlen,
            input_layout="BNSD",
            softmax_scale=1.0 / (query.size(-1) ** 0.5),
            block_size=128,
            block_table=block_table,
            num_query_heads=64,
            num_key_value_heads=1,
            sparse_mode=0,
            inner_precise=1,
            dequant_scale_key=dequant_scale_key,
            dequant_scale_value=dequant_scale_value,
            key_quant_mode=0,
            value_quant_mode=0,
        )


if __name__ == "__main__":
    # -----------------------------
    # 环境设置
    # -----------------------------
    os.environ["ENABLE_ACLNN"] = "false"

    torch._dynamo.reset()

    torch_npu.npu.set_device("npu:0")   # 按需改成你的卡号，例如 npu:7
    torch_npu.npu.set_op_timeout_ms(1000)

    # -----------------------------
    # 编译配置
    # -----------------------------
    config = CompilerConfig()
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._super_kernel_optimize = True
    # 如有需要可指定 build 目录
    # config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "/your_path/aclgraph_sk/tmp"

    npu_backend = tng.get_npu_backend(compiler_config=config)

    # -----------------------------
    # 固定随机种子
    # -----------------------------
    seed = 1236
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------------
    # 构造输入
    # -----------------------------
    batch_size = 18
    q_head_num = 64
    kv_head_num = 1
    q_seq = 1
    head_dim = 128
    kv_seq_length = 8192
    block_size = 128

    # query: [B, N, S, D] -> BNSD
    q_tensor = torch.randn(
        batch_size, q_head_num, q_seq, head_dim,
        dtype=torch.bfloat16
    ).npu()

    max_block_num_prebatch = kv_seq_length // block_size + 1
    block_num = batch_size * max_block_num_prebatch

    block_table = torch.arange(
        batch_size * max_block_num_prebatch,
        dtype=torch.int32
    ).view(batch_size, max_block_num_prebatch).npu()

    # key/value 为 NZ 格式布局对应的 shape
    k_tensor = torch.randn(
        block_num, kv_head_num, head_dim // 32, block_size, 32,
        dtype=torch.int8
    ).npu()

    v_tensor = torch.randn(
        block_num, kv_head_num, head_dim // 32, block_size, 32,
        dtype=torch.int8
    ).npu()

    actual_seq_kvlen = [kv_seq_length] * batch_size

    dequant_scale_key = torch.randn(
        kv_head_num, 1, head_dim,
        dtype=torch.bfloat16
    ).npu()

    dequant_scale_value = torch.randn(
        kv_head_num, 1, head_dim,
        dtype=torch.bfloat16
    ).npu()

    # -----------------------------
    # 编译模型
    # -----------------------------
    print("----------------------- compile & run -----------------------------")

    model = ModelOrigin().npu()
    model = torch.compile(
        model,
        backend=npu_backend,
        fullgraph=True,
        dynamic=True   # 如果你的 shape 固定，也可以改成 False
    )

    # -----------------------------
    # 执行
    # -----------------------------
    output, softmaxlse = model(
        q_tensor,
        k_tensor,
        v_tensor,
        actual_seq_kvlen,
        block_table,
        dequant_scale_key,
        dequant_scale_value,
    )

    print("----------------------- result -----------------------------")
    print("output:")
    print(output)
    print("softmaxlse:")
    print(softmaxlse)
