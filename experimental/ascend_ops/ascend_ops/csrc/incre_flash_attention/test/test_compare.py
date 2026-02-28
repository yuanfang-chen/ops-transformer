# -*- coding: utf-8 -*-
"""
🚀 完整版：NPU Attention 算子对比测试脚本
包含：
  - 张量初始化与 Shape 打印
  - 推理参数配置
  - 调用两个算子（内置 + 自定义）
  - 完整对比输出（相似度、误差、可视化）
  - 保存结果为 JSON 文件
"""

import torch
import torch_npu
import numpy as np
import json
from typing import Dict, Any
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
# 工具函数：对比两个输出
# -----------------------------
def compare_outputs(
    out1: torch.Tensor,
    out2: torch.Tensor,
    rtol: float = 5e-3,
    atol: float = 1e-4,
    print_detail: bool = True,
    max_print_elements: int = 10
) -> Dict[str, Any]:
    """
    比较两个 NPU Tensor 输出的相似性。

    Args:
        out1, out2: torch.Tensor, 输出张量（bfloat16）
        rtol: float, 相对容差
        atol: float, 绝对容差
        print_detail: bool, 是否打印详细对比信息
        max_print_elements: int, 打印前多少个值用于对比

    Returns:
        dict: 包含相似度、误差、结论等信息的字典
    """
    # Step 1: 检查 shape 是否一致
    if out1.shape != out2.shape:
        print(f"\n❌ Shape 不一致！")
        print(f"  out1 shape: {out1.shape}")
        print(f"  out2 shape: {out2.shape}")
        return {
            "success": False,
            "shape_match": False,
            "similarity_ratio": 0.0,
            "max_error": float('inf'),
            "avg_error": float('inf'),
            "conclusion": "Shape 不一致，无法比较"
        }
    else:
        print(f"\n✅ Shape 一致: {out1.shape}")

    # Step 2: 转为 float32 用于精确比较
    arr1 = out1.detach().to(torch.float32).cpu().numpy()
    arr2 = out2.detach().to(torch.float32).cpu().numpy()

    # Step 3: 使用 np.isclose 判断相似性
    is_close = np.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=False)
    similarity_ratio = is_close.mean()
    max_error = np.max(np.abs(arr1 - arr2))
    avg_error = np.mean(np.abs(arr1 - arr2))

    # Step 4: 打印详细信息
    if print_detail:
        print(f"\n🔍 数值相似度分析:")
        print(f"  相似比例: {similarity_ratio:.4f} ({similarity_ratio*100:.2f}%)")
        print(f"  最大误差: {max_error:.6f}")
        print(f"  平均误差: {avg_error:.6f}")

        # 打印前几个值对比
        print(f"\n📋 前 {max_print_elements} 个值对比:")
        print("  index | out1 (CPU)     | out2 (CPU)     | diff")
        print("  ------|------------------|------------------|--------")
        for i in range(min(max_print_elements, arr1.size)):
            val1 = arr1.flat[i]
            val2 = arr2.flat[i]
            diff = abs(val1 - val2)
            print(f"  {i:5d} | {val1:15.6f} | {val2:15.6f} | {diff:7.6f}")

    # Step 5: 判断结论
    if similarity_ratio > 0.995:
        conclusion = "🎉 两个算子输出完全一致，相似度 > 99.5%"
    elif similarity_ratio > 0.99:
        conclusion = "🟡 输出基本一致，存在微小差异（可能是精度或实现细节）"
    elif similarity_ratio > 0.95:
        conclusion = "🔶 输出存在轻微偏差，建议排查算子或输入"
    else:
        conclusion = "❌ 输出存在显著差异，可能存在逻辑错误或参数不一致"

    print(f"\n📌 {conclusion}")

    # Step 6: 返回结果
    result = {
        "success": True,
        "shape_match": True,
        "similarity_ratio": float(similarity_ratio),
        "max_error": float(max_error),
        "avg_error": float(avg_error),
        "conclusion": conclusion
    }

    return result


# -----------------------------
# 主函数：执行完整流程
# -----------------------------
def main():
    print("\n" + "="*60)
    print("🚀 开始对比两个 NPU Attention 算子输出")
    print("="*60)

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
    qkv = torch.randn(batch_size, q_head_num, q_seq, head_dim, dtype=torch.bfloat16).npu()
    print_tensor_shape(qkv, "qkv")

    # Block Table
    kv_block_table = torch.arange(batch_size * max_block_num_prebatch, dtype=torch.int32).view(batch_size, max_block_num_prebatch).npu()
    print_tensor_shape(kv_block_table, "kv_block_table")

    # Key Cache (Int8 Quantized)
    key_cache_npu = torch.randint(
        -128, 128,
        (block_num, kv_head_num, head_dim // 32, block_size, 32),
        dtype=torch.int8,
        device="npu"
    )
    # key_cache_npu = torch.randint(block_num, kv_head_num, head_dim // 32, block_size, 32, dtype=torch.int8).npu()
    print_tensor_shape(key_cache_npu, "key_cache_npu")

    # Value Cache (Int8 Quantized)
    value_cache_npu = torch.randint(
        -128, 128,
        (block_num, kv_head_num, head_dim // 32, block_size, 32),
        dtype=torch.int8,
        device="npu"
    )
    # value_cache_npu = torch.randint(block_num, kv_head_num, head_dim // 32, block_size, 32, dtype=torch.int8).npu()
    print_tensor_shape(value_cache_npu, "value_cache_npu")

    # Sequence Lengths
    q_len = [q_seq] * batch_size
    qkv_len = [kv_seq_length] * batch_size

    # ✅ 修复：打印前 5 个长度值
    print(f"{'q_len':15} | length: {len(q_len):2d} | values: {q_len[:5]}...")
    print(f"{'qkv_len':15} | length: {len(qkv_len):2d} | values: {qkv_len[:5]}...")

    # Antiquantization Scales
    key_antiquant_scale = torch.randn(kv_head_num, 1, head_dim, dtype=torch.bfloat16).npu()
    value_antiquant_scale = torch.randn(kv_head_num, 1, head_dim, dtype=torch.bfloat16).npu()
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
    # 调用两个算子
    # -----------------------------
    print("\n=== 开始调用两个算子 ===\n")

    try:
        print("➡️  调用 torch_npu.npu_fused_infer_attention_score_v2")
        result1, _ = torch_npu.npu_fused_infer_attention_score_v2(**infer_kwargs)

        print("➡️  调用 torch.ops.custom.npu_fused_infer_attention_score")
        result2, _ = torch.ops.custom.npu_fused_infer_attention_score(**infer_kwargs)

    except Exception as e:
        print(f"❌ 调用算子失败: {e}")
        raise

    # -----------------------------
    # 打印结果 Shape
    # -----------------------------
    print("\n=== 输出结果 Shape 打印 ===\n")
    print_tensor_shape(result1, "result1 (torch_npu)")
    print_tensor_shape(result2, "result2 (custom op)")

    # -----------------------------
    # 进行完整对比
    # -----------------------------
    print("\n" + "="*60)
    print("🚀 开始对比两个算子输出")
    print("="*60)

    compare_result = compare_outputs(
        out1=result1,
        out2=result2,
        rtol=5e-3,
        atol=1e-4,
        print_detail=True,
        max_print_elements=10
    )

    # -----------------------------
    # 保存结果到 JSON
    # -----------------------------
    json_file = "attention_compare_result.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(compare_result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 比较结果已保存至: {json_file}")

    # -----------------------------
    # 可选：打印部分结果（CPU）
    # -----------------------------
    print("\n=== 输出结果部分值 (CPU) ===\n")
    print("result1 (first few values):")
    print(result1.detach().to(torch.float32).cpu().numpy()[:2, :2, :2, :2]) 
    print("\nresult2 (first few values):")
    print(result2.detach().to(torch.float32).cpu().numpy()[:2, :2, :2, :2])


# -----------------------------
# 程序入口
# -----------------------------
if __name__ == "__main__":
    main()
