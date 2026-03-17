#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# 最小化成对覆盖测试用例生成器。
#
# 对 18 种合法量化组合 × 6 种 cache_mode × 2 种 t_flag × 2 种 query_norm_flag
# 进行成对覆盖（pairwise coverage），生成约 25-30 个测试用例，覆盖所有
# 内核模板分支而无需穷举 ~336 种合法配置。
#
# 用法:
#   python gen_coverage_testcases.py          # 独立运行，打印生成的用例
#   from gen_coverage_testcases import generate_coverage_params  # 作为模块导入

import itertools
import torch

# wqm=3 requires torch.float8_e8m0 (PyTorch 2.6+)
HAS_FLOAT8_E8M0 = hasattr(torch, 'float8_e8m0')

# ---------------------------------------------------------------------------
# 18 种合法量化组合 (weight_quant_mode, kv_cache_quant_mode, query_quant_mode)
# ---------------------------------------------------------------------------
# 约束:
#   wqm=0 → qqm=0, kqm=0
#   wqm=1 → qqm=0, kqm∈{0,1,2}
#   wqm=2 → qqm∈{0,1}, kqm∈{0,1,2,3}
#   wqm=3 → qqm∈{0,1}, kqm∈{0,1,2} (not 3)
VALID_QUANT_COMBOS = [
    (0, 0, 0),
    (1, 0, 0), (1, 2, 0),
    (3, 0, 0), (3, 1, 1), (3, 3, 0)
]

CACHE_MODES = ["PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"]


def is_valid(wqm, kqm, qqm, cache_mode, t_flag, qnorm):
    """检查参数组合是否满足所有约束。"""
    # wqm=3 需要 torch.float8_e8m0 支持
    if wqm == 3 and not HAS_FLOAT8_E8M0:
        return False
    # kqm=3 只支持 PA_BSND, BSND, TND
    if kqm == 3 and cache_mode not in ("PA_BSND", "BSND", "TND"):
        return False
    # BSND 必须 t_flag=False; TND 必须 t_flag=True
    if cache_mode == "BSND" and t_flag:
        return False
    if cache_mode == "TND" and not t_flag:
        return False
    return True


def generate_coverage_params():
    """生成最小化成对覆盖测试参数列表。

    使用 6 维度 (wqm, kqm, qqm, cache_mode, t_flag, qnorm) 进行成对覆盖，
    通过贪心集合覆盖算法选取最少配置覆盖所有合法维度对，并确保 18 种量化
    组合至少各出现一次。

    Returns:
        list[dict]: 约 25-30 个参数字典
    """
    # 枚举所有合法配置
    valid_configs = []
    for wqm, kqm, qqm in VALID_QUANT_COMBOS:
        for cm in CACHE_MODES:
            for tf in [True, False]:
                for qn in [True, False]:
                    if is_valid(wqm, kqm, qqm, cm, tf, qn):
                        valid_configs.append((wqm, kqm, qqm, cm, tf, qn))

    # 6 维度: wqm, kqm, qqm, cache_mode, t_flag, qnorm
    NUM_DIMS = 6

    def get_dims(cfg):
        return cfg  # (wqm, kqm, qqm, cm, tf, qn)

    dim_pairs = list(itertools.combinations(range(NUM_DIMS), 2))

    # 收集所有合法的成对目标
    all_pairs = set()
    for cfg in valid_configs:
        dims = get_dims(cfg)
        for i, j in dim_pairs:
            all_pairs.add((i, j, dims[i], dims[j]))

    # 贪心集合覆盖
    uncovered = set(all_pairs)
    selected = []

    while uncovered:
        best_cfg = None
        best_count = 0
        best_pairs = set()
        for cfg in valid_configs:
            dims = get_dims(cfg)
            pairs = set()
            for i, j in dim_pairs:
                pair = (i, j, dims[i], dims[j])
                if pair in uncovered:
                    pairs.add(pair)
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_pairs = pairs
                best_cfg = cfg
        if best_cfg is None:
            break
        selected.append(best_cfg)
        uncovered -= best_pairs

    # 确保所有 18 种量化组合至少出现一次
    covered_quants = set()
    for cfg in selected:
        covered_quants.add((cfg[0], cfg[1], cfg[2]))
    for wqm, kqm, qqm in VALID_QUANT_COMBOS:
        if (wqm, kqm, qqm) not in covered_quants:
            for cfg in valid_configs:
                if cfg[0] == wqm and cfg[1] == kqm and cfg[2] == qqm:
                    selected.append(cfg)
                    covered_quants.add((wqm, kqm, qqm))
                    break

    # 转换为参数字典
    result = []
    for wqm, kqm, qqm, cm, tf, qn in selected:
        result.append({
            "batch_size": 2,
            "seq_len": 4,
            "head_num": 8,
            "He": 7168,
            "dtype": torch.bfloat16,
            "cache_mode": cm,
            "block_size": 16,
            "weight_quant_mode": wqm,
            "kv_cache_quant_mode": kqm,
            "query_quant_mode": qqm,
            "ckvkr_repo_mode": 1 if kqm == 3 else 0,
            "quant_scale_repo_mode": 1 if kqm == 3 else 0,
            "query_norm_flag": qn,
            "t_flag": tf,
        })

    return result


if __name__ == "__main__":
    params = generate_coverage_params()
    print(f"Generated {len(params)} coverage test cases:\n")

    # 统计覆盖情况
    quant_combos_seen = set()
    cache_modes_seen = set()
    t_flags_seen = set()
    qnorms_seen = set()

    for i, p in enumerate(params):
        qc = (p["weight_quant_mode"], p["kv_cache_quant_mode"], p["query_quant_mode"])
        quant_combos_seen.add(qc)
        cache_modes_seen.add(p["cache_mode"])
        t_flags_seen.add(p["t_flag"])
        qnorms_seen.add(p["query_norm_flag"])

        print(f"  [{i:2d}] wqm={qc[0]} kqm={qc[1]} qqm={qc[2]} "
              f"cache={p['cache_mode']:<12s} t_flag={str(p['t_flag']):<5s} "
              f"qnorm={str(p['query_norm_flag']):<5s}")

    print(f"\nCoverage summary:")
    print(f"  Quant combos: {len(quant_combos_seen)}/18")
    print(f"  Cache modes:  {len(cache_modes_seen)}/6  {sorted(cache_modes_seen)}")
    print(f"  t_flag:       {len(t_flags_seen)}/2")
    print(f"  qnorm:        {len(qnorms_seen)}/2")
