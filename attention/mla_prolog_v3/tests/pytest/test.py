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

import itertools
import random
import pytest
from testcases import ENABLED_PARAMS, FUZZ_PARAMS, FUZZ_ENABLED, FUZZ_CASES, FUZZ_SEED, COVERAGE_PARAMS
import check_valid_param
import mla_prolog_v3_cpu_ref


# ---------- 参数组合生成 ----------

param_names = [
    "batch_size", "seq_len", "head_num", "He", "dtype",
    "cache_mode", "block_size",
    "weight_quant_mode", "kv_cache_quant_mode", "query_quant_mode",
    "ckvkr_repo_mode", "quant_scale_repo_mode",
]

param_combinations = []
for params in ENABLED_PARAMS:
    param_values = [params[name] for name in param_names]
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_combinations.append(param_dict)


# ---------- CI 测试 ----------

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_mla_prolog_v3(param_combinations):
    """MLA Prolog V3 单算子精度测试。"""
    try:
        check_valid_param.validate_config(param_combinations)
    except ValueError as e:
        pytest.skip(f"参数校验失败: {e}")

    expect_list, result_list = mla_prolog_v3_cpu_ref.test_mla_prolog_v3(param_combinations)
    check_valid_param.check_result(expect_list, result_list)


# ---------- 成对覆盖测试 ----------

@pytest.mark.ci
@pytest.mark.coverage
@pytest.mark.parametrize("params", COVERAGE_PARAMS,
                          ids=[f"cov_{i}" for i in range(len(COVERAGE_PARAMS))])
def test_mla_prolog_v3_coverage(params):
    """Coverage test: minimal pairwise set of dtype/mode combinations."""
    try:
        check_valid_param.validate_config(params)
    except ValueError as e:
        pytest.skip(f"参数校验失败: {e}")
    expect_list, result_list = mla_prolog_v3_cpu_ref.test_mla_prolog_v3(params)
    check_valid_param.check_result(expect_list, result_list)


# ---------- Fuzz 测试 ----------

@pytest.mark.fuzz
def test_mla_prolog_v3_fuzz():
    """MLA Prolog V3 随机化 Fuzz 测试。

    通过环境变量控制:
        MLA_PROLOG_V3_ENABLE_FUZZ=1  开启 fuzz 测试
        MLA_PROLOG_V3_FUZZ_CASES=50  fuzz 测试用例数
        MLA_PROLOG_V3_FUZZ_SEED=42   随机种子
    """
    if not FUZZ_ENABLED:
        pytest.skip("Fuzz testing disabled. Set MLA_PROLOG_V3_ENABLE_FUZZ=1 to enable.")

    rng = random.Random(FUZZ_SEED)

    for case_idx in range(FUZZ_CASES):
        # 从 fuzz 参数范围中随机选择
        params = {}
        for name in param_names:
            params[name] = rng.choice(FUZZ_PARAMS[name])

        # 校验参数合法性，跳过非法组合
        try:
            check_valid_param.validate_config(params)
        except ValueError:
            continue

        print(f"\n[Fuzz case {case_idx + 1}/{FUZZ_CASES}] params: {params}")
        expect_list, result_list = mla_prolog_v3_cpu_ref.test_mla_prolog_v3(params)
        check_valid_param.check_result(expect_list, result_list)
