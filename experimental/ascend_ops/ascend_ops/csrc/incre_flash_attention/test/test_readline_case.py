# -*- coding: utf-8 -*-
"""
NPU Attention 算子批量对比测试脚本（重构版）

功能：
1. 从 Excel 读取测试用例
2. 构造输入张量
3. 分别调用内置算子与自定义算子
4. 对比输出结果
5. 所有 case 结果汇总保存到一个 JSON 文件
6. 支持打印开关 / 保存 JSON 开关
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_npu
import ascend_ops


# =========================================================
# 配置
# =========================================================
@dataclass
class Config:
    excel_path: str = "GQA伪量化.xlsx"
    output_dir: str = "results"
    output_json_name: str = "all_results.json"

    enable_print: bool = True
    enable_save_json: bool = True
    print_tensor_info: bool = True
    print_compare_detail: bool = True
    max_print_elements: int = 10

    rtol: float = 1e-4
    atol: float = 1e-4

    only_first_n_cases: Optional[int] = None  # None 表示全部执行


# =========================================================
# 通用工具
# =========================================================
def log(msg: str, config: Config):
    if config.enable_print:
        print(msg)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def convert_str2tuple(s: Any) -> Tuple[int, ...]:
    """
    将 '8,8' 或 '8, 8' 转成 tuple(int)
    """
    if isinstance(s, tuple):
        return s
    if isinstance(s, list):
        return tuple(map(int, s))
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return tuple()
        return tuple(map(int, s.split(',')))
    raise ValueError(f"无法转换为 tuple: {s}")


def tensor_brief(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    if tensor is None:
        return {
            "shape": None,
            "dtype": None,
            "device": None,
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def print_tensor_shape(tensor: Optional[torch.Tensor], name: str, config: Config):
    if not config.print_tensor_info:
        return
    if tensor is None:
        log(f"{name:20} | shape: {'None':20} | dtype: {'None':15} | device: None", config)
        return
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    device_str = str(tensor.device)
    log(f"{name:20} | shape: {shape_str:20} | dtype: {dtype_str:15} | device: {device_str}", config)


# =========================================================
# 数据结构
# =========================================================
@dataclass
class CaseResult:
    case_id: str
    case_name: str
    success: bool
    shape_match: bool
    similarity_ratio: float
    max_error: float
    avg_error: float
    conclusion: str
    error_message: str = ""
    inputs_summary: Optional[Dict[str, Any]] = None
    outputs_summary: Optional[Dict[str, Any]] = None


# =========================================================
# Excel 读取
# =========================================================
def load_test_cases(excel_path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(excel_path)
    return df.to_dict(orient="records")


# =================================================
# 输入构造
# =========================================================
def build_case_inputs(case: Dict[str, Any], config: Config) -> Dict[str, Any]:
    log("\n=== 张量初始化与 Shape 打印 ===\n", config)

    batch_size = case['B']
    q_seq = case['Q_S']
    kv_seq_length = case['KV_S']
    head_dim = case['Q_D']
    input_layout = case['inputLayout']
    q_head_num = case['numHeads']
    kv_head_num = case['numKeyValueHeads']
    block_size = case['blockSize']

    # Query
    q = torch.randn(convert_str2tuple(case['q_shape']), dtype=torch.bfloat16).npu()
    print_tensor_shape(q, "query", config)

    # Block Table
    block_table_shape = convert_str2tuple(case['blockTable_shape'])
    kv_block_table = (
        torch.arange(torch.prod(torch.tensor(block_table_shape)))
        .view(*block_table_shape)
        .to(dtype=torch.int32)
        .npu()
    )
    print_tensor_shape(kv_block_table, "block_table", config)

    # Key Cache
    k_range = convert_str2tuple(case['k_datarange'])
    k_shape = convert_str2tuple(case['k_cache_shape'])
    key_cache_npu = torch.randint(
        low=k_range[0],
        high=k_range[1] + 1,
        size=k_shape,
        dtype=torch.int8
    ).npu()
    print_tensor_shape(key_cache_npu, "key_cache", config)

    # Value Cache
    v_range = convert_str2tuple(case['v_datarange'])
    v_shape = convert_str2tuple(case['v_cache_shape'])
    value_cache_npu = torch.randint(
        low=v_range[0],
        high=v_range[1] + 1,
        size=v_shape,
        dtype=torch.int8
    ).npu()
    print_tensor_shape(value_cache_npu, "value_cache", config)

    # Sequence Lengths
    kv_len = list(convert_str2tuple(case['actual_seq_lengths_kv']))

    log(f"{'actual_seq_kvlen':20} | length: {len(kv_len):2} | values: {kv_len[:5]}", config)

    # Dequant Scale
    scale_dtype = torch.bfloat16 if case['v_antiquantScale_dtype'] == 'BF16' else torch.float32
    key_antiquant_scale = torch.randn(
        convert_str2tuple(case['k_antiquantScale_shape']),
        dtype=scale_dtype
    ).npu()
    value_antiquant_scale = torch.randn(
        convert_str2tuple(case['v_antiquantScale_shape']),
        dtype=scale_dtype
    ).npu()
    print_tensor_shape(key_antiquant_scale, "dequant_scale_key", config)
    print_tensor_shape(value_antiquant_scale, "dequant_scale_value", config)

    # Mask
    sparse_mode = int(case['sparse'])
    mask = None
    if sparse_mode == 3:
        mask = torch.triu(
            torch.ones(convert_str2tuple(case['m_shape']), dtype=torch.bool),
            diagonal=0
        )
        mask_type = case['m_dtype']
        if mask_type == 'BOOL':
            mask = mask.to(dtype=torch.bool).npu()
        elif mask_type == 'INT8':
            mask = mask.to(dtype=torch.int8).npu()
        elif mask_type == 'UINT8':
            mask = mask.to(dtype=torch.uint8).npu()
    print_tensor_shape(mask, "atten_mask", config)

    infer_kwargs = dict(
        query=q,
        key=key_cache_npu,
        value=value_cache_npu,
        actual_seq_kvlen=kv_len,
        atten_mask=mask,
        sparse_mode=sparse_mode,
        input_layout=input_layout,
        softmax_scale=1 / (head_dim ** 0.5),
        block_size=block_size,
        block_table=kv_block_table,
        num_query_heads=int(q_head_num),
        num_key_value_heads=int(kv_head_num),
        inner_precise=int(case['innerprecise']),
        dequant_scale_key=key_antiquant_scale,
        dequant_scale_value=value_antiquant_scale,
        key_quant_mode=int(case['k_antiquantMode']),
        value_quant_mode=int(case['v_antiquantMode']),
    )

    inputs_summary = {
        "batch_size": int(batch_size),
        "q_seq": int(q_seq),
        "kv_seq_length": int(kv_seq_length),
        "head_dim": int(head_dim),
        "input_layout": input_layout,
        "q_head_num": int(q_head_num),
        "kv_head_num": int(kv_head_num),
        "block_size": int(block_size),
        "query": tensor_brief(q),
        "key": tensor_brief(key_cache_npu),
        "value": tensor_brief(value_cache_npu),
        # "actual_seq_kvlen": tensor_brief(kv_len),
        "block_table": tensor_brief(kv_block_table),
        "dequant_scale_key": tensor_brief(key_antiquant_scale),
        "dequant_scale_value": tensor_brief(value_antiquant_scale),
        "atten_mask": tensor_brief(mask),
    }

    return {
        "infer_kwargs": infer_kwargs,
        "inputs_summary": inputs_summary,
    }


# =========================================================
# 算子执行
# =========================================================
def run_builtin_op(infer_kwargs: Dict[str, Any], config: Config):
    log("➡️  调用 torch_npu.npu_fused_infer_attention_score_v2", config)
    return torch_npu.npu_fused_infer_attention_score_v2(**infer_kwargs)


def run_custom_op(infer_kwargs: Dict[str, Any], config: Config):
    actual_seq_kvlen = infer_kwargs.get("actual_seq_kvlen")
    actual_seq_kvlen = torch.tensor(actual_seq_kvlen, dtype=torch.int64).npu()


    log("➡️  调用 torch.ops.custom.npu_fused_infer_attention_score", config)

    # return torch.ops.custom.npu_fused_infer_attention_score(**infer_kwargs)

    return torch.ops.custom.npu_fused_infer_attention_score(
        **{k: v for k, v in infer_kwargs.items() if k != "actual_seq_kvlen"},
        actual_seq_kvlen=actual_seq_kvlen
    )


# =========================================================
# 输出对比
# =========================================================
def compare_outputs(
    out1: torch.Tensor,
    out2: torch.Tensor,
    config: Config
) -> Dict[str, Any]:
    if out1.shape != out2.shape:
        log(f"\n❌ Shape 不一致！", config)
        log(f"  out1 shape: {out1.shape}", config)
        log(f"  out2 shape: {out2.shape}", config)
        return {
            "success": False,
            "shape_match": False,
            "similarity_ratio": 0.0,
            "max_error": float("inf"),
            "avg_error": float("inf"),
            "conclusion": "Shape 不一致，无法比较"
        }

    log(f"\n✅ Shape 一致: {out1.shape}", config)

    arr1 = out1.detach().to(torch.float32).cpu().numpy()
    arr2 = out2.detach().to(torch.float32).cpu().numpy()

    is_close = np.isclose(arr1, arr2, rtol=config.rtol, atol=config.atol, equal_nan=False)
    similarity_ratio = float(is_close.mean())
    max_error = float(np.max(np.abs(arr1 - arr2)))
    avg_error = float(np.mean(np.abs(arr1 - arr2)))

    if config.print_compare_detail:
        log(f"\n🔍值相似度分析:", config)
        log(f"  相似比例: {similarity_ratio:.4f} ({similarity_ratio*100:.2f}%)", config)
        log(f"  最大误差: {max_error:.6f}", config)
        log(f"  平均误差: {avg_error:.6f}", config)

        log(f"\n📋 前 {config.max_print_elements} 个值对比:", config)
        log("  index | out1            | out2            | diff", config)
        log("  ------|-----------------|-----------------|--------", config)
        for i in range(min(config.max_print_elements, arr1.size)):
            val1 = arr1.flat[i]
            val2 = arr2.flat[i]
            diff = abs(val1 - val2)
            log(f"  {i:5d} | {val1:15.6f} | {val2:15.6f} | {diff:7.6f}", config)

    if similarity_ratio > 0.995:
        conclusion = "两个算子输出完全一致，相似度 > 99.5%"
    elif similarity_ratio > 0.99:
        conclusion = "输出基本一致，存在微小差异"
    elif similarity_ratio > 0.95:
        conclusion = "输出存在轻微偏差，建议排查"
    else:
        conclusion = "输出存在显著差异，可能存在逻辑错误"

    log(f"\n📌 {conclusion}", config)

    return {
        "success": True,
        "shape_match": True,
        "similarity_ratio": similarity_ratio,
        "max_error": max_error,
        "avg_error": avg_error,
        "conclusion": conclusion
    }


# =========================================================
# 单个 case 执行
# =========================================================
def run_single_case(case: Dict[str, Any], case_id: str, config: Config) -> CaseResult:
    case_name = case.get("Testcase_Name", "Unnamed Case")
    log(f"\n{'='*60}", config)
    log(f"🧪 正在处理 {case_id}: {case_name}", config)

    try:
        built = build_case_inputs(case, config)
        infer_kwargs = built["infer_kwargs"]
        inputs_summary = built["inputs_summary"]

        log("\n=== 开始调用两个算子 ===\n", config)
        result1, lse1 = run_builtin_op(infer_kwargs, config)

        result2, lse2 = run_custom_op(infer_kwargs, config)
        # result2, lse2 = run_builtin_op(infer_kwargs, config)
        
        # result1 = result2
        # lse1 = lse2
        log("\n=== 输出结果 Shape 打印 ===\n", config)
        print_tensor_shape(result1, "result_builtin", config)
        print_tensor_shape(result2, "result_custom", config)
        print_tensor_shape(lse1, "softmax_lse_builtin", config)
        print_tensor_shape(lse2, "softmax_lse_custom", config)

        log(f"\n{'='*60}", config)
        log("🚀 开始对比两个算子输出", config)
        log(f"{'='*60}", config)

        compare_result = compare_outputs(result1, result2, config)

        outputs_summary = {
            "result_builtin": tensor_brief(result1),
            "result_custom": tensor_brief(result2),
            "softmax_lse_builtin": tensor_brief(lse1),
            "softmax_lse_custom": tensor_brief(lse2),
        }

        return CaseResult(
            case_id=case_id,
            case_name=case_name,
            success=compare_result["success"],
            shape_match=compare_result["shape_match"],
            similarity_ratio=compare_result["similarity_ratio"],
            max_error=compare_result["max_error"],
            avg_error=compare_result["avg_error"],
            conclusion=compare_result["conclusion"],
            inputs_summary=inputs_summary,
            outputs_summary=outputs_summary
        )

    except Exception as e:
        log(f"❌ 用例执行失败: {e}", config)
        return CaseResult(
            case_id=case_id,
            case_name=case_name,
            success=False,
            shape_match=False,
            similarity_ratio=0.0,
            max_error=float("inf"),
            avg_error=float("inf"),
            conclusion="算子调用或执行失败",
            error_message=str(e)
        )


# =========================================================
# 汇总结果
# =========================================================
def build_summary(results: List[CaseResult]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed

    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": 0.0 if total == 0 else passed / total,
    }


def save_all_results(results: List[CaseResult], summary: Dict[str, Any], config: Config):
    if not config.enable_save_json:
        return

    ensure_dir(config.output_dir)
    output_path = os.path.join(config.output_dir, config.output_json_name)

    data = {
        "summary": summary,
        "config": asdict(config),
        "results": [asdict(r) for r in results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log(f"\n✅ 所有结果已保存至: {output_path}", config)


# =========================================================
# 主流程
# =========================================================
def main():
    config = Config(
        excel_path="GQA伪量化.xlsx",
        output_dir="results",
        output_json_name="all_results.json",

        enable_print=True,          # 是否打印日志
        enable_save_json=True,      # 是否保存 JSON
        print_tensor_info=True,     # 是否打印张量 shape 信息
        print_compare_detail=True,  # 是否打印逐元素对比
        max_print_elements=10,

        rtol=1e-5,
        atol=1e-5,

        only_first_n_cases=2,    # 例如设成 1 就只跑第一条
    )

    log("\n" + "=" * 60, config)
    log("🚀 开始批量对比 NPU Attention 算子输出", config)
    log("=" * 60, config)

    try:
        test_cases = load_test_cases(config.excel_path)
    except Exception as e:
        log(f"❌ 无法读取 Excel 文件: {e}", config)
        return

    if config.only_first_n_cases is not None:
        test_cases = test_cases[:config.only_first_n_cases]

    for idx, case in enumerate(test_cases):
        log(f"✅ 已加载用例 {idx+1}: {case.get('Testcase_Name', 'Unnamed Case')}", config)

    results: List[CaseResult] = []

    for idx, case in enumerate(test_cases):
        case_id = f"case_{idx + 1:03d}"
        result = run_single_case(case, case_id, config)
        results.append(result)

    summary = build_summary(results)

    log(f"\n{'=' * 60}", config)
    log("📊 批量测试完成！", config)
    log(f"   总用例数: {summary['total_cases']}", config)
    log(f"   通过用例: {summary['passed_cases']}", config)
    log(f"   失败用例: {summary['failed_cases']}", config)
    log(f"   通过率: {summary['pass_rate']:.2%}", config)
    log(f"{'=' * 60}", config)

    save_all_results(results, summary, config)


if __name__ == "__main__":
    main()
