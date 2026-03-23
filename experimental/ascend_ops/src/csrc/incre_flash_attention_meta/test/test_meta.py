import torch
import torch_npu
import time
import ascend_ops

# -----------------------------
# 工具函数：打印张量 shape
# -----------------------------
def print_tensor_shape(tensor, name):
    """打印张量的 shape 和 dtype"""
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    device_str = str(tensor.device)
    print(f"{name:15} | shape: {shape_str:20} | dtype: {dtype_str:15} | device: {device_str}")

# -----------------------------
# 参数配置
# -----------------------------
batch_size = 4
q_seq = 1
q_head_num = 64
kv_seq_length = 8192
kv_head_num = 1
head_dim = 128
block_size = 128
max_block_num_per_batch = kv_seq_length // block_size + 1
actual_seq_lengths_kv = torch.tensor([kv_seq_length] * batch_size).to(torch.int32).npu()

infer_kwargs = dict(
    batch_size = batch_size,
    query_seq_size = q_seq,
    query_head_num = q_head_num,
    key_head_num = kv_head_num,
    head_dim = head_dim,
    block_size = 128,
    max_block_num_per_batch = max_block_num_per_batch,
    actual_seq_lengths_kv = actual_seq_lengths_kv,
    layout_query = "BNSD",
)

result = torch.ops.custom.npu_fused_infer_attention_score_metadata(**infer_kwargs)
# print(result.cpu())