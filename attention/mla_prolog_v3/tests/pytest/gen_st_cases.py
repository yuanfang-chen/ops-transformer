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

"""
Coverage-driven testcase generator for MlaPrologV3 pytest.

This script models key tiling/kernel decision predicates, fuzzes a factor pool,
builds a minimal positive set by deterministic set cover, emits negative runtime
cases, and writes:
1) testcases.py
2) coverage markdown report
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


PARAM_NAMES: List[str] = [
    "batch_size",
    "He",
    "Hcq",
    "Hckv",
    "q_head_num",
    "kv_head_num",
    "head_dim",
    "rope_head_dim",
    "q_seq",
    "block_size",
    "input_layout",
    "cache_mode",
    "bs_fused_flag",
    "cq_epsilon",
    "ckv_epsilon",
    "dtype",
    "weight_quant_mode",
    "kv_quant_mode",
    "query_quant_mode",
    "ckvkr_repo_mode",
    "quant_scale_repo_mode",
    "query_norm_flag",
    "tile_size",
    "qc_qr_scale",
    "kc_scale",
]


SUPPORTED_HE: Set[int] = {1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192}
SUPPORTED_N: Set[int] = {1, 2, 4, 8, 16, 32, 64, 128}
SUPPORTED_CACHE_MODE: Tuple[str, ...] = (
    "PA_BSND",
    "PA_NZ",
    "PA_BLK_BSND",
    "PA_BLK_NZ",
    "BSND",
    "TND",
)


DEFAULT_FACTOR_SPACE: Dict[str, List[object]] = {
    "batch_size": [1, 2, 4, 6, 8, 9, 12, 16],
    "He": [1024, 2048, 7168, 7680],
    "q_head_num": [1, 2, 4, 8, 16, 32, 64, 128],
    "q_seq": [1, 2, 4, 8, 16],
    "block_size": [16, 128],
    "cache_mode": ["PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"],
    "bs_fused_flag": [0, 1],
    "weight_quant_mode": [0, 1, 2, 3],
    "kv_quant_mode": [0, 1, 2, 3],
    "query_norm_flag": [0, 1],
}


FIXED_FIELDS: Dict[str, object] = {
    "Hcq": 1536,
    "Hckv": 512,
    "kv_head_num": 1,
    "head_dim": 128,
    "rope_head_dim": 64,
    "input_layout": "BSH",
    "cq_epsilon": 0.0005,
    "ckv_epsilon": 0.0005,
    "dtype": "torch.bfloat16",
    "tile_size": 128,
    "qc_qr_scale": 1.0,
    "kc_scale": 1.0,
    "query_norm_flag": 0,
}


FACTOR_SUMMARY: Tuple[Tuple[str, str], ...] = (
    ("weight_quant_mode / kv_quant_mode / query_quant_mode", "QUANT_MODE routing and dynamic-quant/dequant paths"),
    ("cache_mode", "ND/PA scatter path family and per-tile legality"),
    ("bs_fused_flag", "tokenX 2D fused path; gates actualSeqMode for PA_BLK cache modes"),
    ("ckvkr_repo_mode / quant_scale_repo_mode / tile_size", "per-tile storage path and dtile constraints"),
    ("batch_size / q_seq", "token count, vector split tail, and step batch tail"),
    ("q_head_num", "mm3/mm4 split and tail behavior, dequant-opt gating"),
    ("He", "mm1/mm2 stepK (3/4), kL1 loops, and baseK path"),
    ("block_size", "PA block behavior and legality"),
    ("kv_head_num", "fixed positive at 1; invalid values used for negative runtime cases"),
    ("query_norm_flag", "query_norm output path"),
    ("hardware profile (aic_num/aiv_num)", "cube/vector partition and tail reachability"),
)


MODEL_FACTOR_NAMES: Tuple[str, ...] = (
    "batch_size",
    "He",
    "q_head_num",
    "q_seq",
    "block_size",
    "cache_mode",
    "bs_fused_flag",
    "weight_quant_mode",
    "kv_quant_mode",
    "query_norm_flag",
)


NEGATIVE_RUNTIME_CASES_DEFAULT: List[Dict[str, object]] = [
    {
        "name": "invalid_n_size_not_supported",
        "params": {
            "batch_size": 1,
            "He": 7168,
            "Hcq": 1536,
            "Hckv": 512,
            "q_head_num": 3,
            "kv_head_num": 1,
            "head_dim": 128,
            "rope_head_dim": 64,
            "q_seq": 1,
            "block_size": 128,
            "input_layout": "BSH",
            "cache_mode": "PA_BSND",
            "bs_fused_flag": 0,
            "cq_epsilon": 0.0005,
            "ckv_epsilon": 0.0005,
            "dtype": "torch.bfloat16",
            "weight_quant_mode": 0,
            "kv_quant_mode": 0,
            "query_quant_mode": 0,
            "ckvkr_repo_mode": 0,
            "quant_scale_repo_mode": 0,
            "query_norm_flag": 0,
            "tile_size": 128,
            "qc_qr_scale": 1.0,
            "kc_scale": 1.0,
        },
        "expected_error_substrings": ["n", "head", "unsupported", "invalid", "error"],
    },
    {
        "name": "invalid_tile_size_for_pertile",
        "params": {
            "batch_size": 1,
            "He": 7168,
            "Hcq": 1536,
            "Hckv": 512,
            "q_head_num": 32,
            "kv_head_num": 1,
            "head_dim": 128,
            "rope_head_dim": 64,
            "q_seq": 1,
            "block_size": 128,
            "input_layout": "BSH",
            "cache_mode": "PA_BSND",
            "bs_fused_flag": 0,
            "cq_epsilon": 0.0005,
            "ckv_epsilon": 0.0005,
            "dtype": "torch.bfloat16",
            "weight_quant_mode": 1,
            "kv_quant_mode": 3,
            "query_quant_mode": 0,
            "ckvkr_repo_mode": 1,
            "quant_scale_repo_mode": 1,
            "query_norm_flag": 0,
            "tile_size": 64,
            "qc_qr_scale": 1.0,
            "kc_scale": 1.0,
        },
        "expected_error_substrings": ["tile", "unsupported", "invalid", "error"],
    },
    {
        "name": "invalid_kv_head_num",
        "params": {
            "batch_size": 1,
            "He": 7168,
            "Hcq": 1536,
            "Hckv": 512,
            "q_head_num": 32,
            "kv_head_num": 2,
            "head_dim": 128,
            "rope_head_dim": 64,
            "q_seq": 1,
            "block_size": 128,
            "input_layout": "BSH",
            "cache_mode": "PA_BSND",
            "bs_fused_flag": 0,
            "cq_epsilon": 0.0005,
            "ckv_epsilon": 0.0005,
            "dtype": "torch.bfloat16",
            "weight_quant_mode": 0,
            "kv_quant_mode": 0,
            "query_quant_mode": 0,
            "ckvkr_repo_mode": 0,
            "quant_scale_repo_mode": 0,
            "query_norm_flag": 0,
            "tile_size": 128,
            "qc_qr_scale": 1.0,
            "kc_scale": 1.0,
        },
        "expected_error_substrings": ["kv", "head", "unsupported", "invalid", "error"],
    },
    {
        "name": "invalid_he_not_supported",
        "params": {
            "batch_size": 1,
            "He": 7000,
            "Hcq": 1536,
            "Hckv": 512,
            "q_head_num": 32,
            "kv_head_num": 1,
            "head_dim": 128,
            "rope_head_dim": 64,
            "q_seq": 1,
            "block_size": 128,
            "input_layout": "BSH",
            "cache_mode": "PA_BSND",
            "bs_fused_flag": 0,
            "cq_epsilon": 0.0005,
            "ckv_epsilon": 0.0005,
            "dtype": "torch.bfloat16",
            "weight_quant_mode": 0,
            "kv_quant_mode": 0,
            "query_quant_mode": 0,
            "ckvkr_repo_mode": 0,
            "quant_scale_repo_mode": 0,
            "query_norm_flag": 0,
            "tile_size": 128,
            "qc_qr_scale": 1.0,
            "kc_scale": 1.0,
        },
        "expected_error_substrings": ["he", "unsupported", "invalid", "error"],
    },
]


@dataclass(frozen=True)
class HardwareProfile:
    aic_num: int = 24
    aiv_num: int = 48

    @property
    def cv_ratio(self) -> int:
        if self.aic_num <= 0:
            return 1
        return max(1, self.aiv_num // self.aic_num)

    @property
    def cv_mode(self) -> str:
        if self.aic_num == self.aiv_num:
            return "1:1"
        return "1:2"


@dataclass
class CaseWithCoverage:
    name: str
    params: Dict[str, object]
    tags: Set[str]

    @property
    def sort_key(self) -> Tuple:
        p = self.params
        return (
            int(p["weight_quant_mode"]),
            int(p["kv_quant_mode"]),
            str(p["cache_mode"]),
            int(p["bs_fused_flag"]),
            int(p["He"]),
            int(p["q_head_num"]),
            int(p["batch_size"]),
            int(p["q_seq"]),
            int(p["block_size"]),
        )

    @property
    def runtime_cost(self) -> int:
        p = self.params
        return int(p["batch_size"]) * int(p["q_seq"]) * int(p["q_head_num"])


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def calc_single_core_n(n: int, core_num: int, align_num: int) -> int:
    return ceil_div(n, align_num * core_num) * align_num


def quant_mode(weight_quant_mode: int, kv_quant_mode: int) -> Optional[int]:
    if weight_quant_mode == 0:
        return 0 if kv_quant_mode == 0 else None
    if weight_quant_mode == 1:
        return {0: 1, 2: 2, 3: 5}.get(kv_quant_mode)
    if weight_quant_mode == 2:
        return {0: 3, 1: 4, 3: 6}.get(kv_quant_mode)
    if weight_quant_mode == 3:
        return {0: 7, 1: 8, 3: 9}.get(kv_quant_mode)
    return None


def derive_query_repo_modes(weight_quant_mode: int, kv_quant_mode: int) -> Tuple[int, int, int]:
    if kv_quant_mode == 3:
        ckvkr_repo_mode = 1
        quant_scale_repo_mode = 1
    else:
        ckvkr_repo_mode = 0
        quant_scale_repo_mode = 0
    query_quant_mode = 1 if (weight_quant_mode in (2, 3) and kv_quant_mode == 1) else 0
    return query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode


def validate_positive_case(case: Dict[str, object]) -> bool:
    weight_quant_mode = int(case["weight_quant_mode"])
    kv_quant_mode = int(case["kv_quant_mode"])
    cache_mode = str(case["cache_mode"])
    bs_fused_flag = int(case["bs_fused_flag"])
    query_norm_flag = int(case.get("query_norm_flag", 0))
    query_norm_flag = int(case.get("query_norm_flag", 0))
    query_norm_flag = int(case.get("query_norm_flag", 0))
    query_quant_mode = int(case["query_quant_mode"])
    ckvkr_repo_mode = int(case["ckvkr_repo_mode"])
    quant_scale_repo_mode = int(case["quant_scale_repo_mode"])
    tile_size = int(case["tile_size"])

    qm = quant_mode(weight_quant_mode, kv_quant_mode)
    if qm is None:
        return False
    if cache_mode not in SUPPORTED_CACHE_MODE:
        return False
    if bs_fused_flag not in (0, 1):
        return False
    if int(case["He"]) not in SUPPORTED_HE:
        return False
    if int(case["q_head_num"]) not in SUPPORTED_N:
        return False
    if int(case["kv_head_num"]) != 1:
        return False
    if int(case["Hcq"]) != 1536 or int(case["Hckv"]) != 512:
        return False
    if int(case["head_dim"]) != 128 or int(case["rope_head_dim"]) != 64:
        return False
    if cache_mode == "TND" and bs_fused_flag != 1:
        return False
    if cache_mode == "BSND" and bs_fused_flag != 0:
        return False
    if int(case["block_size"]) < 16 or int(case["block_size"]) > 1024 or int(case["block_size"]) % 16 != 0:
        return False

    if kv_quant_mode == 3:
        if cache_mode in {"PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ"}:
            return False
        if ckvkr_repo_mode != 1 or quant_scale_repo_mode != 1:
            return False
        if tile_size != 128:
            return False
    else:
        if ckvkr_repo_mode != 0 or quant_scale_repo_mode != 0:
            return False

    expect_query_quant_mode = 1 if (weight_quant_mode in (2, 3) and kv_quant_mode == 1) else 0
    if query_quant_mode != expect_query_quant_mode:
        return False
    return True


def build_case_tags(case: Dict[str, object], hw: HardwareProfile) -> Set[str]:
    tags: Set[str] = set()
    b = int(case["batch_size"])
    s = int(case["q_seq"])
    n = int(case["q_head_num"])
    he = int(case["He"])
    weight_quant_mode = int(case["weight_quant_mode"])
    kv_quant_mode = int(case["kv_quant_mode"])
    cache_mode = str(case["cache_mode"])
    bs_fused_flag = int(case["bs_fused_flag"])
    query_norm_flag = int(case.get("query_norm_flag", 0))

    qm = quant_mode(weight_quant_mode, kv_quant_mode)
    assert qm is not None

    t_size = b * s
    step_batch_size = min(128, t_size)
    vector_block_num = min(step_batch_size, hw.aiv_num)

    is_pertile = kv_quant_mode == 3
    is_mm_input_one_byte = weight_quant_mode in (2, 3)
    is_mm_uqqr_one_byte = weight_quant_mode in (1, 2, 3)

    enable_dequant_opt = False
    if weight_quant_mode == 3:
        enable_dequant_opt = True
    elif weight_quant_mode in (1, 2) and n >= 8:
        enable_dequant_opt = True

    enable_group_compute_opt = False  # unreachable under current V3 pytest constraints (Nkv fixed to 1)
    need_qn_dynamic_quant = weight_quant_mode in (2, 3) and kv_quant_mode == 1 and not is_pertile

    tags.add(f"quant_mode:qm{qm}")
    tags.add(f"cache_mode:{cache_mode}")
    tags.add(f"tiling:bs_fused_flag:{bs_fused_flag}")
    tags.add(f"tiling:enable_dequant_opt:{int(enable_dequant_opt)}")
    tags.add(f"tiling:enable_group_compute_opt:{int(enable_group_compute_opt)}")
    if bs_fused_flag == 1 and cache_mode in {"PA_BLK_BSND", "PA_BLK_NZ"}:
        tags.add("tiling:actual_seq_mode:en_q_len")
    else:
        tags.add("tiling:actual_seq_mode:disabled")
    tags.add("tiling:split_m_mode:0")
    tags.add("tiling:empty_tensor_mode:non_empty")
    tags.add(f"tiling:cv_mode:{hw.cv_mode}")

    tags.add(f"post:need_qn_dynamic_quant:{int(need_qn_dynamic_quant)}")
    tags.add(f"post:is_pertile:{int(is_pertile)}")
    tags.add(f"post:query_norm_flag:{query_norm_flag}")

    # mm1
    mm1_align = 32 if is_mm_input_one_byte else 16
    mm1_single = calc_single_core_n(1536, hw.aic_num, mm1_align)
    mm1_single = max(mm1_single, 64)
    mm1_block_num = ceil_div(1536, mm1_single)
    mm1_last_n = 1536 - (mm1_block_num - 1) * mm1_single
    mm1_cube_tail = int(mm1_last_n != mm1_single)
    mm1_base_n = 64 if weight_quant_mode == 3 else 128
    mm1_nl1_loops = ceil_div(mm1_last_n, mm1_base_n)
    mm1_nl1_tail = int(mm1_last_n % mm1_base_n != 0)
    tags.add(f"cube:mm1_tail:{mm1_cube_tail}")
    tags.add(f"l1:mm1_nl1_loops:{'multi' if mm1_nl1_loops > 1 else 'single'}")
    tags.add(f"l1:mm1_nl1_tail:{mm1_nl1_tail}")

    # mm2
    if hw.aic_num >= 9:
        mm2_single = 64
        mm2_block_num = (512 + 64) // 64
    else:
        mm2_align = 32 if is_mm_input_one_byte else 16
        mm2_single = calc_single_core_n(576, hw.aic_num, mm2_align)
        mm2_block_num = ceil_div(576, mm2_single)
    mm2_last_n = 576 - (mm2_block_num - 1) * mm2_single
    mm2_cube_tail = int(mm2_last_n != mm2_single)
    mm2_base_n = 128
    mm2_nl1_loops = ceil_div(mm2_last_n, mm2_base_n)
    mm2_nl1_tail = int(mm2_last_n % mm2_base_n != 0)
    tags.add(f"cube:mm2_tail:{mm2_cube_tail}")
    tags.add(f"l1:mm2_nl1_loops:{'multi' if mm2_nl1_loops > 1 else 'single'}")
    tags.add(f"l1:mm2_nl1_tail:{mm2_nl1_tail}")

    # mm3
    mm3_ori_m = n * (128 + 64)
    if enable_group_compute_opt:
        mm3_single = calc_single_core_n(n * 128, 8, 128)
    elif enable_dequant_opt:
        mm3_single = calc_single_core_n(mm3_ori_m, hw.aic_num, 192)
    else:
        mm3_align = 32 if is_mm_uqqr_one_byte else 16
        mm3_single = calc_single_core_n(mm3_ori_m, hw.aic_num, mm3_align)
    mm3_block_num = ceil_div(mm3_ori_m, mm3_single)
    mm3_last_n = mm3_ori_m - (mm3_block_num - 1) * mm3_single
    mm3_cube_tail = int(mm3_last_n != mm3_single)
    if weight_quant_mode == 3:
        mm3_base_n = 128
    elif step_batch_size <= 64:
        mm3_base_n = 256
    else:
        mm3_base_n = 128
    mm3_nl1_loops = ceil_div(mm3_last_n, mm3_base_n)
    mm3_nl1_tail = int(mm3_last_n % mm3_base_n != 0)
    tags.add(f"cube:mm3_tail:{mm3_cube_tail}")
    tags.add(f"l1:mm3_nl1_loops:{'multi' if mm3_nl1_loops > 1 else 'single'}")
    tags.add(f"l1:mm3_nl1_tail:{mm3_nl1_tail}")

    # mm4
    mm4_single = ceil_div(n, hw.aic_num)
    mm4_block_num = ceil_div(n, mm4_single)
    mm4_last = n - (mm4_block_num - 1) * mm4_single
    mm4_tail = int(mm4_last != mm4_single)
    tags.add(f"cube:mm4_tail:{mm4_tail}")

    # kL1 / stepK for mm1/mm2
    mm_base_k = 256 if is_mm_input_one_byte else 128
    step_k_12 = 4
    if ((he // mm_base_k) % step_k_12) != 0:
        step_k_12 = 3
    kl1_step_12 = mm_base_k * step_k_12
    kl1_loops_12 = ceil_div(he, kl1_step_12)
    tags.add(f"l1:mm12_stepk:{step_k_12}")
    tags.add(f"l1:mm12_kl1_loops:{'multi' if kl1_loops_12 > 1 else 'single'}")
    tags.add(f"l0:mm12_basek_stepk:{step_k_12}")

    # kL1 for mm3 (k fixed Hcq=1536)
    mm3_base_k = 128 if is_mm_uqqr_one_byte else 64
    mm3_kl1_loops = ceil_div(1536, mm3_base_k * 4)
    tags.add(f"l1:mm3_kl1_loops:{'multi' if mm3_kl1_loops > 1 else 'single'}")

    # vector
    has_batch_tail = int(t_size > step_batch_size and (t_size % step_batch_size != 0))
    has_vector_tail = int(vector_block_num > 0 and (step_batch_size % vector_block_num != 0))
    has_vector_inactive_lanes = int(step_batch_size < hw.aiv_num)
    tags.add(f"vector:step_batch_tail:{has_batch_tail}")
    tags.add(f"vector:token_split_tail:{has_vector_tail}")
    tags.add(f"vector:inactive_lanes:{has_vector_inactive_lanes}")

    return tags


def enumerate_positive_candidates(factor_space: Dict[str, Sequence[object]], hw: HardwareProfile) -> List[CaseWithCoverage]:
    candidates: List[CaseWithCoverage] = []
    keys = [
        "batch_size",
        "He",
        "q_head_num",
        "q_seq",
        "block_size",
        "cache_mode",
        "bs_fused_flag",
        "weight_quant_mode",
        "kv_quant_mode",
        "query_norm_flag",
    ]
    index = 0
    for values in itertools.product(*(factor_space[k] for k in keys)):
        base = dict(FIXED_FIELDS)
        base.update(dict(zip(keys, values)))

        query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode = derive_query_repo_modes(
            int(base["weight_quant_mode"]), int(base["kv_quant_mode"])
        )
        base["query_quant_mode"] = query_quant_mode
        base["ckvkr_repo_mode"] = ckvkr_repo_mode
        base["quant_scale_repo_mode"] = quant_scale_repo_mode
        if int(base.get("query_norm_flag", 0)) == 1:
            base["kc_scale"] = 1.1

        if not validate_positive_case(base):
            continue
        tags = build_case_tags(base, hw)
        candidates.append(CaseWithCoverage(name=f"candidate_{index:04d}", params=base, tags=tags))
        index += 1

    candidates.sort(key=lambda c: c.sort_key)
    return candidates


def deterministic_set_cover(candidates: Sequence[CaseWithCoverage], universe: Set[str]) -> List[CaseWithCoverage]:
    uncovered = set(universe)
    selected: List[CaseWithCoverage] = []
    available = list(candidates)

    while uncovered:
        best: Optional[CaseWithCoverage] = None
        best_cover_count = 0
        for candidate in available:
            cover_count = len(candidate.tags & uncovered)
            if cover_count == 0:
                continue
            if best is None or cover_count > best_cover_count:
                best = candidate
                best_cover_count = cover_count
                continue
            if best is None:
                continue
            if cover_count == best_cover_count:
                if candidate.runtime_cost < best.runtime_cost:
                    best = candidate
                elif candidate.runtime_cost == best.runtime_cost and candidate.sort_key < best.sort_key:
                    best = candidate
        if best is None:
            break
        selected.append(best)
        uncovered -= best.tags
        available = [c for c in available if c is not best]

    # Redundancy prune while preserving order.
    changed = True
    while changed:
        changed = False
        for idx, case in list(enumerate(selected)):
            remainder = selected[:idx] + selected[idx + 1 :]
            covered: Set[str] = set()
            for c in remainder:
                covered |= c.tags
            if universe.issubset(covered):
                selected.pop(idx)
                changed = True
                break
    return selected


def _python_literal(value: object) -> str:
    if isinstance(value, str):
        if value.startswith("torch."):
            return value
        return repr(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _format_case_dict(case_name: str, params: Dict[str, object]) -> str:
    lines: List[str] = [f'    "{case_name}": {{']
    for key in PARAM_NAMES:
        value = params[key]
        lines.append(f"        {repr(key)}: [{_python_literal(value)}],")
    lines.append("    }")
    return "\n".join(lines)


def _format_enabled_params(case_names: Sequence[str]) -> str:
    names = ", ".join(f'TEST_PARAMS["{name}"]' for name in case_names)
    return f"ENABLED_PARAMS = [{names}]"


def render_testcases_py(
    selected_cases: Sequence[CaseWithCoverage],
    fuzz_space: Dict[str, Sequence[object]],
    negative_runtime_cases: Sequence[Dict[str, object]],
) -> str:
    header = """#!/usr/bin/python
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

import torch

# Auto-generated by gen_st_cases.py. Do not hand-edit.
"""
    case_names = [f"coverage_case_{idx:03d}" for idx, _ in enumerate(selected_cases)]
    cases_block = ",\n".join(
        _format_case_dict(case_names[idx], case.params) for idx, case in enumerate(selected_cases)
    )
    test_params = f"TEST_PARAMS = {{\n{cases_block}\n}}\n\n"
    enabled_params = _format_enabled_params(case_names) + "\n\n"

    fuzz_lines = ["FUZZ_PARAM_SPACE = {"]
    for key in PARAM_NAMES:
        values = list(fuzz_space.get(key, []))
        rendered = ", ".join(_python_literal(v) for v in values)
        fuzz_lines.append(f"    {repr(key)}: [{rendered}],")
    fuzz_lines.append("}\n")

    neg_lines = ["NEGATIVE_RUNTIME_CASES = ["]
    for case in negative_runtime_cases:
        neg_lines.append("    {")
        neg_lines.append(f"        'name': {repr(case['name'])},")
        neg_lines.append("        'params': {")
        params = case["params"]
        for key in PARAM_NAMES:
            neg_lines.append(f"            {repr(key)}: {_python_literal(params[key])},")
        expected_subs = case.get("expected_error_substrings", [])
        rendered_subs = ", ".join(repr(s) for s in expected_subs)
        neg_lines.append("        },")
        neg_lines.append(f"        'expected_error_substrings': [{rendered_subs}],")
        neg_lines.append("    },")
    neg_lines.append("]")
    neg_lines.append("")

    return header + test_params + enabled_params + "\n".join(fuzz_lines) + "\n" + "\n".join(neg_lines)


def build_tree_map(universe: Set[str], hw: HardwareProfile) -> str:
    quant_modes = sorted(tag.split(":")[1] for tag in universe if tag.startswith("quant_mode:"))
    cache_modes = sorted(tag.split(":")[1] for tag in universe if tag.startswith("cache_mode:"))

    def yn(tag: str) -> str:
        return "reachable" if tag in universe else "unreachable"

    lines = [
        "MlaPrologV3 Condition Coverage",
        "├─ Quant Scenario",
    ]
    for qm in quant_modes:
        lines.append(f"│  ├─ {qm} ({yn(f'quant_mode:{qm}')})")
    lines.extend(
        [
            "├─ Cache Path",
        ]
    )
    for cm in cache_modes:
        lines.append(f"│  ├─ {cm} ({yn(f'cache_mode:{cm}')})")
    lines.extend(
        [
            "├─ Tiling Flags",
            f"│  ├─ enableDequantOpt = {{{0},{1}}} ({yn('tiling:enable_dequant_opt:0')}, {yn('tiling:enable_dequant_opt:1')})",
            "│  ├─ enableGroupComputeOpt = {0} [unreachable=1 under current constraints]",
            f"│  ├─ bsFusedFlag = {{{0},{1}}} ({yn('tiling:bs_fused_flag:0')}, {yn('tiling:bs_fused_flag:1')})",
            f"│  ├─ actualSeqMode = {{DISABLED, EN_Q_LEN}} ({yn('tiling:actual_seq_mode:disabled')}, {yn('tiling:actual_seq_mode:en_q_len')})",
            "│  ├─ splitMFlag = 0 (fixed by tiling)",
            "│  ├─ emptyTensorMode = NON_EMPTY (positive set), EMPTY_* via negative/runtime-fail set",
            f"│  └─ cvMode = {hw.cv_mode}",
            "├─ Cube Core Tails",
            f"│  ├─ mm1 tail yes/no ({yn('cube:mm1_tail:1')}, {yn('cube:mm1_tail:0')})",
            f"│  ├─ mm2 tail yes/no ({yn('cube:mm2_tail:1')}, {yn('cube:mm2_tail:0')})",
            f"│  ├─ mm3 tail yes/no ({yn('cube:mm3_tail:1')}, {yn('cube:mm3_tail:0')})",
            f"│  └─ mm4 tail yes/no ({yn('cube:mm4_tail:1')}, {yn('cube:mm4_tail:0')})",
            "├─ L1 Loops",
            f"│  ├─ nL1 tail mm1 yes/no ({yn('l1:mm1_nl1_tail:1')}, {yn('l1:mm1_nl1_tail:0')})",
            f"│  ├─ nL1 tail mm2 yes/no ({yn('l1:mm2_nl1_tail:1')}, {yn('l1:mm2_nl1_tail:0')})",
            f"│  ├─ nL1 tail mm3 yes/no ({yn('l1:mm3_nl1_tail:1')}, {yn('l1:mm3_nl1_tail:0')})",
            f"│  ├─ kL1 loops mm12 single/multi ({yn('l1:mm12_kl1_loops:single')}, {yn('l1:mm12_kl1_loops:multi')})",
            f"│  └─ stepK mode {{3,4}} ({yn('l1:mm12_stepk:3')}, {yn('l1:mm12_stepk:4')})",
            "├─ L0 Loops",
            f"│  └─ baseK inner loops stepK {{3,4}} ({yn('l0:mm12_basek_stepk:3')}, {yn('l0:mm12_basek_stepk:4')})",
            "├─ Vector Core",
            f"│  ├─ step batch tail yes/no ({yn('vector:step_batch_tail:1')}, {yn('vector:step_batch_tail:0')})",
            f"│  ├─ vector token split tail yes/no ({yn('vector:token_split_tail:1')}, {yn('vector:token_split_tail:0')})",
            f"│  └─ active/inactive vector lanes ({yn('vector:inactive_lanes:0')}, {yn('vector:inactive_lanes:1')})",
            "└─ Postprocess",
            f"   ├─ needQnDynamicQuant yes/no ({yn('post:need_qn_dynamic_quant:1')}, {yn('post:need_qn_dynamic_quant:0')})",
            f"   ├─ isPertile yes/no ({yn('post:is_pertile:1')}, {yn('post:is_pertile:0')})",
            f"   └─ query_norm_flag yes/no ({yn('post:query_norm_flag:1')}, {yn('post:query_norm_flag:0')})",
        ]
    )
    return "\n".join(lines)


def render_coverage_report(
    selected_cases: Sequence[CaseWithCoverage],
    candidates: Sequence[CaseWithCoverage],
    universe: Set[str],
    hw: HardwareProfile,
    focus_factors: Optional[Sequence[str]] = None,
) -> str:
    selected_union: Set[str] = set()
    for case in selected_cases:
        selected_union |= case.tags
    uncovered = sorted(universe - selected_union)

    lines: List[str] = [
        "# MlaPrologV3 ST Coverage Report",
        "",
        f"- Hardware profile: AIC={hw.aic_num}, AIV={hw.aiv_num}, CV={hw.cv_mode}",
        f"- Candidate count: {len(candidates)}",
        f"- Selected positive cases: {len(selected_cases)}",
        f"- Reachable tag count: {len(universe)}",
        f"- Uncovered reachable tags: {len(uncovered)}",
    ]
    if focus_factors:
        lines.append(f"- Focus factors: {', '.join(focus_factors)}")
    lines.extend(
        [
            "",
            "## Factor Summary",
        ]
    )
    for factor, reason in FACTOR_SUMMARY:
        lines.append(f"- `{factor}`: {reason}")

    lines.extend(
        [
            "",
            "## Tree Map",
            "```text",
            build_tree_map(universe, hw),
            "```",
            "",
            "## Selected Cases",
        ]
    )

    for idx, case in enumerate(selected_cases):
        lines.append(f"### coverage_case_{idx:03d}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(case.params, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        lines.append("- Covered tags:")
        for tag in sorted(case.tags):
            lines.append(f"  - `{tag}`")
        lines.append("")

    lines.append("## Case-Condition Matrix")
    case_headers = [f"coverage_case_{idx:03d}" for idx, _ in enumerate(selected_cases)]
    lines.append("| condition tag | " + " | ".join(case_headers) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in case_headers) + " |")
    for tag in sorted(universe):
        marks = ["Y" if tag in case.tags else "" for case in selected_cases]
        lines.append("| `" + tag + "` | " + " | ".join(marks) + " |")
    lines.append("")

    lines.append("## Uncovered Reachable Tags")
    if uncovered:
        for tag in uncovered:
            lines.append(f"- `{tag}`")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Reachable Tags")
    for tag in sorted(universe):
        lines.append(f"- `{tag}`")
    lines.append("")
    return "\n".join(lines)


def _merge_fuzz_space(factor_space: Dict[str, Sequence[object]]) -> Dict[str, List[object]]:
    fuzz: Dict[str, List[object]] = {k: [] for k in PARAM_NAMES}
    for key in PARAM_NAMES:
        if key in factor_space:
            fuzz[key] = list(factor_space[key])
        elif key in FIXED_FIELDS:
            fuzz[key] = [FIXED_FIELDS[key]]

    # Expand with valid derived dimensions and defaults.
    fuzz["Hcq"] = [1536]
    fuzz["Hckv"] = [512]
    fuzz["kv_head_num"] = [1]
    fuzz["head_dim"] = [128]
    fuzz["rope_head_dim"] = [64]
    fuzz["input_layout"] = ["BSH"]
    fuzz["bs_fused_flag"] = list(factor_space.get("bs_fused_flag", [0, 1]))
    fuzz["dtype"] = ["torch.bfloat16"]
    fuzz["cq_epsilon"] = [0.0005, 0.001]
    fuzz["ckv_epsilon"] = [0.0005, 0.001]
    fuzz["tile_size"] = [128]
    fuzz["qc_qr_scale"] = [1.0]
    fuzz["kc_scale"] = [1.0, 1.1]
    fuzz["query_norm_flag"] = [0, 1]
    fuzz["query_quant_mode"] = [0, 1]
    fuzz["ckvkr_repo_mode"] = [0, 1]
    fuzz["quant_scale_repo_mode"] = [0, 1]

    derived_modes: Set[int] = set()
    derived_ckvkr_modes: Set[int] = set()
    derived_quant_scale_modes: Set[int] = set()
    for weight_mode in fuzz["weight_quant_mode"]:
        for kv_mode in fuzz["kv_quant_mode"]:
            qm = quant_mode(int(weight_mode), int(kv_mode))
            if qm is None:
                continue
            query_mode, ckvkr_mode, quant_scale_mode = derive_query_repo_modes(int(weight_mode), int(kv_mode))
            derived_modes.add(query_mode)
            derived_ckvkr_modes.add(ckvkr_mode)
            derived_quant_scale_modes.add(quant_scale_mode)

    if derived_modes:
        fuzz["query_quant_mode"] = sorted(derived_modes)
    if derived_ckvkr_modes:
        fuzz["ckvkr_repo_mode"] = sorted(derived_ckvkr_modes)
    if derived_quant_scale_modes:
        fuzz["quant_scale_repo_mode"] = sorted(derived_quant_scale_modes)

    # Ensure stable order and dedup.
    for key in PARAM_NAMES:
        seen: List[object] = []
        for val in fuzz[key]:
            if val not in seen:
                seen.append(val)
        fuzz[key] = seen
    return fuzz


def _normalize_factor_space(space: Dict[str, Sequence[object]]) -> Dict[str, List[object]]:
    normalized: Dict[str, List[object]] = {}
    for key, value in space.items():
        normalized[key] = list(value)
    return normalized


def _load_factor_space(path: Optional[Path]) -> Dict[str, List[object]]:
    if path is None:
        return _normalize_factor_space(DEFAULT_FACTOR_SPACE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("factor space json must be an object")
    merged = _normalize_factor_space(DEFAULT_FACTOR_SPACE)
    for key, value in payload.items():
        if key not in merged:
            continue
        if not isinstance(value, list):
            raise ValueError(f"factor {key} must be list")
        merged[key] = value
    return merged


def _normalize_focus_factors(focus_factors: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for factor in focus_factors:
        item = factor.strip()
        if not item:
            continue
        if item not in MODEL_FACTOR_NAMES:
            raise ValueError(f"unknown focus factor: {item}")
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("focus_factors is empty after normalization")
    return normalized


def _select_baseline_factors(factor_space: Dict[str, Sequence[object]]) -> Dict[str, object]:
    baseline: Dict[str, object] = {}
    for name in MODEL_FACTOR_NAMES:
        values = list(factor_space.get(name, []))
        if not values:
            raise ValueError(f"factor_space[{name}] should not be empty")
        baseline[name] = values[0]
    return baseline


def enumerate_feature_relative_candidates(
    factor_space: Dict[str, Sequence[object]],
    hw: HardwareProfile,
    focus_factors: Sequence[str],
) -> List[CaseWithCoverage]:
    normalized_focus = _normalize_focus_factors(focus_factors)
    baseline = _select_baseline_factors(factor_space)
    varying_keys = [k for k in MODEL_FACTOR_NAMES if k in normalized_focus]

    candidates: List[CaseWithCoverage] = []
    index = 0
    for values in itertools.product(*(factor_space[k] for k in varying_keys)):
        base = dict(FIXED_FIELDS)
        for name in MODEL_FACTOR_NAMES:
            base[name] = baseline[name]
        base.update(dict(zip(varying_keys, values)))

        query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode = derive_query_repo_modes(
            int(base["weight_quant_mode"]), int(base["kv_quant_mode"])
        )
        base["query_quant_mode"] = query_quant_mode
        base["ckvkr_repo_mode"] = ckvkr_repo_mode
        base["quant_scale_repo_mode"] = quant_scale_repo_mode
        if int(base.get("query_norm_flag", 0)) == 1:
            base["kc_scale"] = 1.1

        if not validate_positive_case(base):
            continue
        tags = build_case_tags(base, hw)
        candidates.append(CaseWithCoverage(name=f"feature_candidate_{index:04d}", params=base, tags=tags))
        index += 1

    candidates.sort(key=lambda c: c.sort_key)
    return candidates


def generate_feature_relative_cases(
    factor_space: Dict[str, Sequence[object]],
    hw: HardwareProfile,
    focus_factors: Sequence[str],
) -> Tuple[List[CaseWithCoverage], List[CaseWithCoverage], Set[str]]:
    candidates = enumerate_feature_relative_candidates(factor_space, hw, focus_factors)
    if not candidates:
        raise RuntimeError("no valid feature-relative candidates generated")

    universe: Set[str] = set()
    for case in candidates:
        universe |= case.tags

    selected = deterministic_set_cover(candidates, universe)
    if not selected:
        raise RuntimeError("failed to build feature-relative selected cases")

    selected_covered: Set[str] = set()
    for case in selected:
        selected_covered |= case.tags
    missing = universe - selected_covered
    if missing:
        raise RuntimeError(f"feature-relative selected cases missed tags: {sorted(missing)}")
    return selected, candidates, universe


def generate(
    output_testcases: Path,
    output_report: Path,
    factor_space: Dict[str, Sequence[object]],
    hw: HardwareProfile,
    focus_factors: Optional[Sequence[str]] = None,
) -> None:
    if focus_factors:
        selected, candidates, universe = generate_feature_relative_cases(dict(factor_space), hw, focus_factors)
    else:
        candidates = enumerate_positive_candidates(dict(factor_space), hw)
        if not candidates:
            raise RuntimeError("no valid positive candidates generated")

        universe = set()
        for c in candidates:
            universe |= c.tags

        selected = deterministic_set_cover(candidates, universe)
        if not selected:
            raise RuntimeError("failed to build selected positive cases")

        selected_covered: Set[str] = set()
        for c in selected:
            selected_covered |= c.tags
        missing = universe - selected_covered
        if missing:
            raise RuntimeError(f"selected cases did not cover all reachable tags: {sorted(missing)}")

    fuzz_space = _merge_fuzz_space(dict(factor_space))

    testcases_text = render_testcases_py(selected, fuzz_space, NEGATIVE_RUNTIME_CASES_DEFAULT)
    output_testcases.write_text(testcases_text, encoding="utf-8")

    report_text = render_coverage_report(selected, candidates, universe, hw, focus_factors=focus_factors)
    output_report.write_text(report_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate coverage-driven ST cases for mla_prolog_v3 pytest.")
    parser.add_argument(
        "--factor-space-json",
        type=Path,
        default=None,
        help="Optional json file to override default factor pool.",
    )
    parser.add_argument(
        "--aic-num",
        type=int,
        default=24,
        help="AIC core count of current hardware profile.",
    )
    parser.add_argument(
        "--aiv-num",
        type=int,
        default=48,
        help="AIV core count of current hardware profile.",
    )
    parser.add_argument(
        "--output-testcases",
        type=Path,
        default=Path(__file__).resolve().parent / "testcases.py",
        help="Output path for generated testcase module.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).resolve().parent / "st_case_coverage_report.md",
        help="Output path for generated coverage report.",
    )
    parser.add_argument(
        "--focus-factors",
        type=str,
        default="",
        help=(
            "Comma-separated factor names for feature-relative generation only. "
            "Supported: " + ",".join(MODEL_FACTOR_NAMES)
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor_space = _load_factor_space(args.factor_space_json)
    hw = HardwareProfile(aic_num=args.aic_num, aiv_num=args.aiv_num)
    focus_factors = None
    if args.focus_factors.strip():
        focus_factors = _normalize_focus_factors(args.focus_factors.split(","))
    generate(
        output_testcases=args.output_testcases,
        output_report=args.output_report,
        factor_space=factor_space,
        hw=hw,
        focus_factors=focus_factors,
    )
    print(f"generated: {args.output_testcases}")
    print(f"generated: {args.output_report}")


if __name__ == "__main__":
    main()
