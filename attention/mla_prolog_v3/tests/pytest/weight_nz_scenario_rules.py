#!/usr/bin/env python3
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

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple


WEIGHT_NZ_NPU_SUPPORTED_WEIGHT_MODES = (0, 1, 2, 3)
WEIGHT_NZ_NPU_SUPPORTED_CACHE_MODES = ("PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND")
PERTILE_UNSUPPORTED_CACHE_MODES = {"PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ"}
PA_BLK_CACHE_MODES = {"PA_BLK_BSND", "PA_BLK_NZ"}

SCENARIO_NO_QUANT = "no_quant"
SCENARIO_HALF_NO_KV = "half_quant_kv_no_quant"
SCENARIO_HALF_PER_CHANNEL = "half_quant_kv_per_channel"
SCENARIO_HALF_PER_TILE = "half_quant_kv_per_tile"
SCENARIO_INT8_FULL_NO_KV = "int8_full_kv_no_quant"
SCENARIO_INT8_FULL_PER_TENSOR = "int8_full_kv_per_tensor"
SCENARIO_INT8_FULL_PER_TILE = "int8_full_kv_per_tile"
SCENARIO_MXFP8_FULL_NO_KV = "mxfp8_full_kv_no_quant"
SCENARIO_MXFP8_FULL_PER_TENSOR = "mxfp8_full_kv_per_tensor"
SCENARIO_MXFP8_FULL_PER_TILE = "mxfp8_full_kv_per_tile"

_SCENARIO_MAP = {
    (0, 0): SCENARIO_NO_QUANT,
    (1, 0): SCENARIO_HALF_NO_KV,
    (1, 2): SCENARIO_HALF_PER_CHANNEL,
    (1, 3): SCENARIO_HALF_PER_TILE,
    (2, 0): SCENARIO_INT8_FULL_NO_KV,
    (2, 1): SCENARIO_INT8_FULL_PER_TENSOR,
    (2, 3): SCENARIO_INT8_FULL_PER_TILE,
    (3, 0): SCENARIO_MXFP8_FULL_NO_KV,
    (3, 1): SCENARIO_MXFP8_FULL_PER_TENSOR,
    (3, 3): SCENARIO_MXFP8_FULL_PER_TILE,
}


def classify_weight_nz_npu_quant_scenario(weight_quant_mode: int, kv_quant_mode: int) -> Optional[str]:
    return _SCENARIO_MAP.get((int(weight_quant_mode), int(kv_quant_mode)))


def derive_weight_nz_query_repo_modes(weight_quant_mode: int, kv_quant_mode: int) -> Optional[Tuple[int, int, int]]:
    scenario = classify_weight_nz_npu_quant_scenario(weight_quant_mode, kv_quant_mode)
    if scenario is None:
        return None

    query_quant_mode = int(scenario in (SCENARIO_INT8_FULL_PER_TENSOR, SCENARIO_MXFP8_FULL_PER_TENSOR))
    if int(kv_quant_mode) == 3:
        return query_quant_mode, 1, 1
    return query_quant_mode, 0, 0


def validate_weight_nz_npu_combo(cache_mode,
                                 weight_quant_mode,
                                 kv_quant_mode,
                                 query_quant_mode,
                                 ckvkr_repo_mode,
                                 quant_scale_repo_mode) -> Tuple[bool, str]:
    cache_mode = str(cache_mode)
    weight_quant_mode = int(weight_quant_mode)
    kv_quant_mode = int(kv_quant_mode)
    query_quant_mode = int(query_quant_mode)
    ckvkr_repo_mode = int(ckvkr_repo_mode)
    quant_scale_repo_mode = int(quant_scale_repo_mode)

    if cache_mode not in WEIGHT_NZ_NPU_SUPPORTED_CACHE_MODES:
        return False, f"unsupported cache_mode={cache_mode}"
    if weight_quant_mode not in WEIGHT_NZ_NPU_SUPPORTED_WEIGHT_MODES:
        return False, "WeightNz NPU only supports weight_quant_mode in {0,1,2,3}"

    scenario = classify_weight_nz_npu_quant_scenario(weight_quant_mode, kv_quant_mode)
    if scenario is None:
        return False, "unsupported WeightNz NPU weight/kv quant scenario"

    if kv_quant_mode == 3:
        if cache_mode in PERTILE_UNSUPPORTED_CACHE_MODES:
            return False, f"cache_mode={cache_mode} does not support per-tile kv quant"
        if ckvkr_repo_mode != 1 or quant_scale_repo_mode != 1:
            return False, "per-tile kv quant requires ckvkr_repo_mode=1 and quant_scale_repo_mode=1"
    else:
        if ckvkr_repo_mode != 0 or quant_scale_repo_mode != 0:
            return False, "non per-tile kv quant requires ckvkr_repo_mode=0 and quant_scale_repo_mode=0"

    expected_modes = derive_weight_nz_query_repo_modes(weight_quant_mode, kv_quant_mode)
    assert expected_modes is not None
    expected_query_quant_mode, _, _ = expected_modes
    if query_quant_mode != expected_query_quant_mode:
        return False, f"query_quant_mode should be {expected_query_quant_mode} for this WeightNz scenario"

    return True, ""


def get_weight_nz_optional_input_policy(named_params: Mapping[str, object]) -> Dict[str, bool]:
    weight_quant_mode = int(named_params["weight_quant_mode"])
    kv_quant_mode = int(named_params["kv_quant_mode"])
    cache_mode = str(named_params["cache_mode"])
    bs_fused_flag = int(named_params.get("bs_fused_flag", 0))
    smooth_scales_cq_flag = int(named_params.get("smooth_scales_cq_flag", 0))

    scenario = classify_weight_nz_npu_quant_scenario(weight_quant_mode, kv_quant_mode)
    if scenario is None:
        raise ValueError("unsupported WeightNz NPU scenario")

    int8_full_scenarios = {
        SCENARIO_INT8_FULL_NO_KV,
        SCENARIO_INT8_FULL_PER_TENSOR,
        SCENARIO_INT8_FULL_PER_TILE,
    }
    mxfp8_full_scenarios = {
        SCENARIO_MXFP8_FULL_NO_KV,
        SCENARIO_MXFP8_FULL_PER_TENSOR,
        SCENARIO_MXFP8_FULL_PER_TILE,
    }
    half_quant_scenarios = {
        SCENARIO_HALF_NO_KV,
        SCENARIO_HALF_PER_CHANNEL,
        SCENARIO_HALF_PER_TILE,
    }

    has_full_dequant = scenario in int8_full_scenarios or scenario in mxfp8_full_scenarios
    has_half_dequant_uqqr = scenario in half_quant_scenarios
    has_smooth_scales = bool(smooth_scales_cq_flag) and (scenario in half_quant_scenarios or scenario in int8_full_scenarios)

    return {
        "dequant_scale_x": has_full_dequant,
        "dequant_scale_w_dq": has_full_dequant,
        "dequant_scale_w_uq_qr": has_half_dequant_uqqr or has_full_dequant,
        "dequant_scale_w_dkv_kr": has_full_dequant,
        "quant_scale_ckv": scenario in {
            SCENARIO_HALF_PER_CHANNEL,
            SCENARIO_INT8_FULL_PER_TENSOR,
            SCENARIO_MXFP8_FULL_PER_TENSOR,
        },
        "quant_scale_ckr": scenario == SCENARIO_HALF_PER_CHANNEL,
        "smooth_scales_cq": has_smooth_scales,
        "actual_seq_len": bool(bs_fused_flag) and cache_mode in PA_BLK_CACHE_MODES,
        "k_nope_clip_alpha": scenario in {SCENARIO_HALF_PER_TILE, SCENARIO_INT8_FULL_PER_TILE},
    }
