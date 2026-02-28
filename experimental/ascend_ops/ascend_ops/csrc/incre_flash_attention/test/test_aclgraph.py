import torch
import torch_npu
import torch.nn as nn
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
import torchair as tng

import time
import numpy as np
import os
import ascend_ops
# import logging
# import torch
# torch._dynamo.reset()
# torch._dynamo.config.verbose = True
# torch._dynamo.config.suppress_errors = False
# # 新版日志接口
# torch._logging.set_logs(dynamo=logging.DEBUG)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, param: dict):
        # for k,v in param.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.device, v.dtype, tuple(v.shape), type(v))
        #     else:
        #         print(k, type(v), v)
        # return torch.ops.npu.npu_fused_infer_attention_score_v2(**param)
        return torch.ops.custom.npu_fused_infer_attention_score(**param)

os.environ["ENABLE_ACLNN"] = "false"  
torch._dynamo.reset()      
npu_mode = Network().npu()
config = CompilerConfig()   
config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
config.mode = "reduce-overhead"
# config.experimental_config.tiling_schedule_optimize = True                                                         
npu_backend = tng.get_npu_backend(compiler_config=config)       

npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)

batch_size = 18
q_head_num = 64
kv_head_num = 1
q_seq = 1
head_dim = 128
kv_seq_length = 8192
q_tensor = torch.randn(batch_size, q_head_num, q_seq, head_dim).to(dtype=torch.bfloat16).npu()
block_size = 128
block_num = batch_size * (kv_seq_length // block_size + 1)
max_block_num_prebatch = kv_seq_length // block_size + 1
blockTable = torch.arange(batch_size * max_block_num_prebatch, dtype=torch.int32).view(batch_size, max_block_num_prebatch).npu()
#kv NZ
k_tensor = torch.randn(block_num, kv_head_num, head_dim//32, block_size, 32).to(dtype=torch.int8).npu()
v_tensor = torch.randn(block_num, kv_head_num, head_dim//32, block_size, 32).to(dtype=torch.int8).npu()

actualSeqLengthqs = [q_seq] * batch_size  # [1, 1, 1, 1]
actualSeqLengthkvs = [kv_seq_length] * batch_size  # [1024, 1024, 1024, 1024]

key_antiquant_scale=torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()
value_antiquant_scale=torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()

scaleValue = 1 / (head_dim**0.5)
m_tensor = ~torch.tril(torch.ones(2048, 2048, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).npu()

# if q_tensor is not None:
#     torch._dynamo.mark_static(q_tensor)
# if k_tensor is not None:
#     torch._dynamo.mark_static(k_tensor)
# if v_tensor is not None:
#     torch._dynamo.mark_static(v_tensor)
# if m_tensor is not None:
#     torch._dynamo.mark_static(m_tensor)
# if blockTable is not None:
#     torch._dynamo.mark_static(blockTable)
# if key_antiquant_scale is not None:
#     torch._dynamo.mark_static(key_antiquant_scale)
# if value_antiquant_scale is not None:
#     torch._dynamo.mark_static(value_antiquant_scale)
# if actualSeqLengthqs is not None:
#     torch._dynamo.mark_static(actualSeqLengthqs)
# if actualSeqLengthkvs is not None:
#     torch._dynamo.mark_static(actualSeqLengthkvs)

param = dict(
        query = q_tensor,
        key = k_tensor,
        value = v_tensor,
        # atten_mask = mask_fa,
        # actual_seq_qlen = actualSeqLengthqs,
        actual_seq_kvlen = actualSeqLengthkvs,
        input_layout = "BNSD",
        softmax_scale = scaleValue,
        block_size = block_size,
        block_table = blockTable,
        num_query_heads  = q_head_num,
        num_key_value_heads = kv_head_num,
        sparse_mode=0,
        inner_precise=1,
        dequant_scale_key = key_antiquant_scale,
        dequant_scale_value = value_antiquant_scale, 
        key_quant_mode = 0,
        value_quant_mode = 0  
    )

output, softmaxlse = npu_mode(param)
print("res : []")
print(output)

