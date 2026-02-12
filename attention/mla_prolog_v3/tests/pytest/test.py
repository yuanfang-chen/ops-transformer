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
import os
import random
import pytest
import torch_npu

import check_valid_param
import prologv3_generalized
from testcases import ENABLED_PARAMS, FUZZ_PARAM_SPACE

PARAM_NAMES = [
    "batch_size", "He", "Hcq", "Hckv", "q_head_num",
    "kv_head_num", "head_dim", "rope_head_dim", "q_seq",
    "block_size", "input_layout", "cache_mode", "cq_epsilon", "ckv_epsilon", "dtype",
    "weight_quant_mode", "kv_quant_mode", "query_quant_mode",
    "ckvkr_repo_mode", "quant_scale_repo_mode", "tile_size",
    "qc_qr_scale", "kc_scale"
]


def _build_param_combinations():
    combinations = []
    seen = set()
    for params in ENABLED_PARAMS:
        values = [params[name] for name in PARAM_NAMES]
        for combo in itertools.product(*values):
            param_dict = dict(zip(PARAM_NAMES, combo))
            valid, _ = prologv3_generalized.validate_quant_cache_combo(
                param_dict["cache_mode"],
                param_dict["weight_quant_mode"],
                param_dict["kv_quant_mode"],
                param_dict["query_quant_mode"],
                param_dict["ckvkr_repo_mode"],
                param_dict["quant_scale_repo_mode"],
            )
            if not valid:
                continue
            key = tuple(param_dict[name] for name in PARAM_NAMES)
            if key in seen:
                continue
            seen.add(key)
            combinations.append(param_dict)
    return combinations


PARAM_COMBINATIONS = _build_param_combinations()


def _to_test_data(param_combinations):
    return (
        param_combinations["batch_size"],
        param_combinations["He"],
        param_combinations["Hcq"],
        param_combinations["Hckv"],
        param_combinations["q_head_num"],
        param_combinations["kv_head_num"],
        param_combinations["head_dim"],
        param_combinations["rope_head_dim"],
        param_combinations["q_seq"],
        param_combinations["block_size"],
        param_combinations["input_layout"],
        param_combinations["cache_mode"],
        param_combinations["cq_epsilon"],
        param_combinations["ckv_epsilon"],
        param_combinations["dtype"],
        param_combinations["weight_quant_mode"],
        param_combinations["kv_quant_mode"],
        param_combinations["query_quant_mode"],
        param_combinations["ckvkr_repo_mode"],
        param_combinations["quant_scale_repo_mode"],
        param_combinations["tile_size"],
        param_combinations["qc_qr_scale"],
        param_combinations["kc_scale"]
    )


def _run_single_case(param_combinations):
    torch_npu.npu.set_device(0)
    check_valid_param.log_discontinuous_error_mode_once()
    test_data = _to_test_data(param_combinations)
    check_valid_param.validate_config(test_data)
    expect, result = prologv3_generalized.test_prologv3_generalized(test_data)
    check_valid_param.check_result(expect, result)


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def build_fuzz_param_cases(case_count=20, seed=3):
    rng = random.Random(seed)
    cases = []
    seen = set()
    max_attempts = max(100, case_count * 200)

    for _ in range(max_attempts):
        if len(cases) >= case_count:
            break
        param_dict = {name: rng.choice(FUZZ_PARAM_SPACE[name]) for name in PARAM_NAMES}
        valid, _ = prologv3_generalized.validate_quant_cache_combo(
            param_dict["cache_mode"],
            param_dict["weight_quant_mode"],
            param_dict["kv_quant_mode"],
            param_dict["query_quant_mode"],
            param_dict["ckvkr_repo_mode"],
            param_dict["quant_scale_repo_mode"],
        )
        if not valid:
            continue

        try:
            check_valid_param.validate_config(_to_test_data(param_dict))
        except ValueError:
            continue

        key = tuple(param_dict[name] for name in PARAM_NAMES)
        if key in seen:
            continue
        seen.add(key)
        cases.append(param_dict)

    return cases


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", PARAM_COMBINATIONS)
def test_mla_prolog_v3(param_combinations):
    try:
        _run_single_case(param_combinations)
    except ValueError as err:
        pytest.skip(f"参数校验失败: {err}")


@pytest.mark.fuzz
def test_mla_prolog_v3_fuzz():
    # Opt-in fuzzing to avoid affecting regular CI runtime.
    if os.getenv("MLA_PROLOG_V3_ENABLE_FUZZ", "0") != "1":
        pytest.skip("set MLA_PROLOG_V3_ENABLE_FUZZ=1 to run fuzz test")

    case_count = _env_int("MLA_PROLOG_V3_FUZZ_CASES", 20)
    seed = _env_int("MLA_PROLOG_V3_FUZZ_SEED", 3)
    if case_count <= 0:
        pytest.skip("MLA_PROLOG_V3_FUZZ_CASES <= 0")

    fuzz_cases = build_fuzz_param_cases(case_count=case_count, seed=seed)
    if not fuzz_cases:
        pytest.skip("no valid fuzz cases generated from FUZZ_PARAM_SPACE")

    for fuzz_case in fuzz_cases:
        _run_single_case(fuzz_case)
