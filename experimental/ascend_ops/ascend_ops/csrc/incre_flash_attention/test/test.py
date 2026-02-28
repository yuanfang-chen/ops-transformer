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
max_block_num_prebatch = kv_seq_length // block_size + 1

# -----------------------------
# 创建并打印各张量 shape
# -----------------------------

print("=== 张量初始化与 Shape 打印 ===\n")

# Query Key Value (QKV)
qkv = torch.randn(batch_size, q_head_num, q_seq, head_dim).to(dtype=torch.bfloat16).npu()
print_tensor_shape(qkv, "qkv")

# Block Table
kv_block_table = torch.arange(batch_size * max_block_num_prebatch, dtype=torch.int32).view(batch_size, max_block_num_prebatch).npu()
print_tensor_shape(kv_block_table, "kv_block_table")

# Key Cache (Int8 Quantized)
key_cache_npu = torch.randint(block_num, kv_head_num, head_dim // 32, block_size, 32).to(dtype=torch.int8).npu()
key_cache_npu = torch.randint(block_num, kv_head_num, head_dim // 32, block_size, 32).to(dtype=torch.int8).npu()
print_tensor_shape(key_cache_npu, "key_cache_npu")

# Value Cache (Int8 Quantized)
value_cache_npu = torch.randn(block_num, kv_head_num, head_dim // 32, block_size, 32).to(dtype=torch.int8).npu()
print_tensor_shape(value_cache_npu, "value_cache_npu")

# Sequence Lengths
q_len = [q_seq] * batch_size
qkv_len = [kv_seq_length] * batch_size

# ✅ 修复：打印前 5 个长度值
print(f"{'q_len':15} | length: {len(q_len):2d} | values: {q_len[:5]}...")
print(f"{'qkv_len':15} | length: {len(qkv_len):2d} | values: {qkv_len[:5]}...")

# Antiquantization Scales
key_antiquant_scale = torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()
value_antiquant_scale = torch.randn(kv_head_num, 1, head_dim).to(dtype=torch.bfloat16).npu()
print_tensor_shape(key_antiquant_scale, "key_antiquant_scale")
print_tensor_shape(value_antiquant_scale, "value_antiquant_scale")

# -----------------------------
# 推理参数配置
# -----------------------------
scale_num = 1 / (head_dim ** 0.5)

infer_kwargs = dict(
    query=qkv,
    key=key_cache_npu,
    value=value_cache_npu,
    actual_seq_kvlen=qkv_len,
    input_layout="BNSD",
    softmax_scale=scale_num,
    block_size=block_size,
    block_table=kv_block_table,
    num_query_heads=q_head_num,
    num_key_value_heads=kv_head_num,
    sparse_mode=0,
    inner_precise=1,
    dequant_scale_key=key_antiquant_scale,
    dequant_scale_value=value_antiquant_scale,
    key_quant_mode=0,
    value_quant_mode=0
)

# -----------------------------
# 调用自定义算子
# -----------------------------
print("\n=== 开始调用 npu_fused_infer_attention_score ===\n")
try:
    result, _ = torch.ops.custom.npu_fused_infer_attention_score(**infer_kwargs)
except Exception as e:
    print(f"❌ 调用算子失败: {e}")
    raise

# -----------------------------
# 打印结果 shape
# -----------------------------
print("\n=== 输出结果 Shape 打印 ===\n")
print_tensor_shape(result, "result")
print(result.cpu())
# 可选：打印部分结果（CPU）
# print("\n=== 输出结果部分值 (CPU) ===\n")
# print("result (first few values):")
# print(result.cpu().detach().numpy()[:2, :2, :2, :2])
