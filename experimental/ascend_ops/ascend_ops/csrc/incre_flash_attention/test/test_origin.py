import torch
import torch_npu
import time

# import get_profiler
# torch.ops.load_library('/data01/a3m9/chl/test/xpu_ops/src/build_out/lib/libxpu_ops.so')
batch_size = 18
q_head_num = 64
kv_head_num = 1
q_seq = 1
block_size = 128
head_dim = 128
kv_seq_length = 8192
block_num = batch_size * (kv_seq_length // block_size + 1)
max_block_num_prebatch = kv_seq_length // block_size + 1

qkv = torch.randn(batch_size, q_head_num, q_seq, head_dim).to(dtype=torch.bfloat16).npu()
kv_block_table = torch.arange(batch_size * max_block_num_prebatch, dtype=torch.int32).view(batch_size, max_block_num_prebatch).npu()

key_cache_npu = torch.randint(block_num, kv_head_num, head_dim//32, block_size, 32).to(dtype=torch.int8).npu()
value_cache_npu = torch.randint(block_num, kv_head_num, head_dim//32, block_size, 32).to(dtype=torch.int8).npu()

q_len = [q_seq] * batch_size
qkv_len = [kv_seq_length] * batch_size

key_antiquant_scale =  torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()
value_antiquant_scale = torch.randn(kv_head_num,1, head_dim).to(dtype=torch.bfloat16).npu()

scale_num = 1 / (head_dim**0.5)
infer_kwargs = dict(
    query = qkv,
    key = key_cache_npu,
    value = value_cache_npu,
    # atten_mask = mask_fa,
    actual_seq_qlen = q_len,
    actual_seq_kvlen = qkv_len,
    input_layout = "BNSD",
    softmax_scale = scale_num,
    block_size = block_size,
    block_table = kv_block_table,
    num_query_heads = q_head_num,
    num_key_value_heads = kv_head_num,
    sparse_mode=0,
    inner_precise=1,
    dequant_scale_key = key_antiquant_scale,
    dequant_scale_value = value_antiquant_scale,
    key_quant_mode = 0,
    value_quant_mode = 0
)

result, _ = torch_npu.npu_fused_infer_attention_score_v2(**infer_kwargs)
print("res : ")
print(result.cpu())