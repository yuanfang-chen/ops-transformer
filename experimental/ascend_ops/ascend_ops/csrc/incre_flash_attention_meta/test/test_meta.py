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
batch_size = 18
q_head_num = 64
kv_head_num = 1
q_seq = 1
block_size = 128
head_dim = 128
kv_seq_length = 8192
block_num = batch_size * (kv_seq_length // block_size + 1)
max_block_num_per_batch = kv_seq_length // block_size + 1

q_len = [q_seq] * batch_size
qkv_len = [kv_seq_length] * batch_size

scale_num = 1 / (head_dim**0.5)

actual_seq_lengths_query = torch.tensor([q_seq] * batch_size).to(torch.int32).npu()
actual_seq_lengths_kv = torch.tensor([kv_seq_length] * batch_size).to(torch.int32).npu()

infer_kwargs = dict(
    batch_size = batch_size,
    query_seq_size = q_seq,
    query_head_num = q_head_num,
    key_seq_size = kv_seq_length,
    key_head_num = kv_head_num,
    block_size = 128,
    max_block_num_per_batch = max_block_num_per_batch,
    actual_seq_lengths_query = actual_seq_lengths_query,
    actual_seq_lengths_kv = None,
    layout_query = "BSND",
    layout_key = 'BSND'
)

result = torch.ops.custom.npu_fused_infer_attention_score_metadata(**infer_kwargs)
print(result.cpu())