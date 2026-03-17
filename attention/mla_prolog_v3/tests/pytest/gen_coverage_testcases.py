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
# 对量化组合 × cache_mode × t_flag × query_norm_flag × head_num × T类别 × He
# 进行成对覆盖（pairwise coverage），通过贪心集合覆盖算法生成最少测试用例。
#
# 用法:
#   python gen_coverage_testcases.py          # 独立运行，打印生成的用例
#   from gen_coverage_testcases import generate_coverage_params  # 作为模块导入

import itertools
import torch

# wqm=3 requires torch.float8_e8m0 (PyTorch 2.6+)
HAS_FLOAT8_E8M0 = hasattr(torch, 'float8_e8m0fnu')

# ---------------------------------------------------------------------------
# 量化组合 (weight_quant_mode, kv_cache_quant_mode, query_quant_mode)
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

# ---------------------------------------------------------------------------
# 形状维度
# ---------------------------------------------------------------------------
HEAD_NUMS = [1, 2, 4, 8, 32, 64, 128]

# T类别: "small" = T=1 (单token decode), "large" = T>128 (prefill)
T_CATEGORIES = ["small", "large"]
T_CATEGORY_MAP = {
    "small": (1, 1),      # (batch_size, seq_len) → T=1
    "large": (2, 128),    # (batch_size, seq_len) → T=256
}

HE_VALUES = [1024, 3072, 7168, 7680]


def is_valid(wqm, kqm, qqm, cache_mode, t_flag, qnorm, head_num, t_cat, he):
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

    9 维度: wqm, kqm, qqm, cache_mode, t_flag, qnorm, head_num, t_cat, He

    Returns:
        list[dict]: 参数字典列表
    """
    # 枚举所有合法配置
    valid_configs = []
    for wqm, kqm, qqm in VALID_QUANT_COMBOS:
        for cm in CACHE_MODES:
            for tf in [True, False]:
                for qn in [True, False]:
                    for hn in HEAD_NUMS:
                        for tc in T_CATEGORIES:
                            for he in HE_VALUES:
                                if is_valid(wqm, kqm, qqm, cm, tf, qn, hn, tc, he):
                                    valid_configs.append((wqm, kqm, qqm, cm, tf, qn, hn, tc, he))

    NUM_DIMS = 9
    dim_pairs = list(itertools.combinations(range(NUM_DIMS), 2))

    # 收集所有合法的成对目标
    all_pairs = set()
    for cfg in valid_configs:
        for i, j in dim_pairs:
            all_pairs.add((i, j, cfg[i], cfg[j]))

    # 贪心集合覆盖
    uncovered = set(all_pairs)
    selected = []

    while uncovered:
        best_cfg = None
        best_count = 0
        best_pairs = set()
        for cfg in valid_configs:
            pairs = set()
            for i, j in dim_pairs:
                pair = (i, j, cfg[i], cfg[j])
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

    # 确保所有量化组合至少出现一次
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
    for wqm, kqm, qqm, cm, tf, qn, hn, tc, he in selected:
        bs, sl = T_CATEGORY_MAP[tc]
        result.append({
            "batch_size": bs,
            "seq_len": sl,
            "head_num": hn,
            "He": he,
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

    quant_combos_seen = set()
    cache_modes_seen = set()
    t_flags_seen = set()
    qnorms_seen = set()
    head_nums_seen = set()
    t_cats_seen = set()
    hes_seen = set()

    for i, p in enumerate(params):
        qc = (p["weight_quant_mode"], p["kv_cache_quant_mode"], p["query_quant_mode"])
        quant_combos_seen.add(qc)
        cache_modes_seen.add(p["cache_mode"])
        t_flags_seen.add(p["t_flag"])
        qnorms_seen.add(p["query_norm_flag"])
        head_nums_seen.add(p["head_num"])
        t = p["batch_size"] * p["seq_len"]
        t_cats_seen.add("small" if t == 1 else "large")
        hes_seen.add(p["He"])

        print(f"  [{i:2d}] wqm={qc[0]} kqm={qc[1]} qqm={qc[2]} "
              f"cache={p['cache_mode']:<12s} t_flag={str(p['t_flag']):<5s} "
              f"qnorm={str(p['query_norm_flag']):<5s} "
              f"N={p['head_num']:<3d} T={t:<3d} He={p['He']}")

    print(f"\nCoverage summary:")
    print(f"  Quant combos: {len(quant_combos_seen)}/{len(VALID_QUANT_COMBOS)}")
    print(f"  Cache modes:  {len(cache_modes_seen)}/6  {sorted(cache_modes_seen)}")
    print(f"  t_flag:       {len(t_flags_seen)}/2")
    print(f"  qnorm:        {len(qnorms_seen)}/2")
    print(f"  head_num:     {len(head_nums_seen)}/{len(HEAD_NUMS)}  {sorted(head_nums_seen)}")
    print(f"  T_category:   {len(t_cats_seen)}/{len(T_CATEGORIES)}  {sorted(t_cats_seen)}")
    print(f"  He:           {len(hes_seen)}/{len(HE_VALUES)}  {sorted(hes_seen)}")
