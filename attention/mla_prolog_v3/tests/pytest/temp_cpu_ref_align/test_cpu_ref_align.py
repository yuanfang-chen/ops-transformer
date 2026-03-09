import os
import sys
from pathlib import Path

import pytest
import torch


PYTEST_DIR = Path(__file__).resolve().parents[1]
if str(PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTEST_DIR))

import check_valid_param
import prologv3_generalized

from .runner import assert_cpu_refs_aligned, skip_if_case_unsupported


BASE_CASE = {
    "batch_size": 2,
    "He": 1024,
    "Hcq": 1536,
    "Hckv": 512,
    "q_head_num": 8,
    "kv_head_num": 1,
    "head_dim": 128,
    "rope_head_dim": 64,
    "q_seq": 2,
    "block_size": 16,
    "input_layout": "BSH",
    "cache_mode": "BSND",
    "bs_fused_flag": 0,
    "cq_epsilon": 0.0005,
    "ckv_epsilon": 0.0005,
    "dtype": torch.bfloat16,
    "weight_quant_mode": 0,
    "kv_quant_mode": 0,
    "query_quant_mode": 0,
    "ckvkr_repo_mode": 0,
    "quant_scale_repo_mode": 0,
    "smooth_scales_cq_flag": 0,
    "query_norm_flag": 0,
    "tile_size": 128,
    "qc_qr_scale": 1.0,
    "kc_scale": 1.0,
}


def _make_case(**overrides):
    case = dict(BASE_CASE)
    case.update(overrides)
    return case


CPU_REF_ALIGN_CASES = [
    _make_case(weight_quant_mode=0, kv_quant_mode=0, cache_mode="BSND", bs_fused_flag=0,
               query_quant_mode=0, smooth_scales_cq_flag=0, query_norm_flag=0),
    _make_case(weight_quant_mode=1, kv_quant_mode=0, cache_mode="PA_BSND", bs_fused_flag=0,
               query_quant_mode=0, smooth_scales_cq_flag=1, query_norm_flag=1),
    _make_case(weight_quant_mode=1, kv_quant_mode=3, cache_mode="TND", bs_fused_flag=1,
               query_quant_mode=0, ckvkr_repo_mode=1, quant_scale_repo_mode=1,
               smooth_scales_cq_flag=1, query_norm_flag=0),
    _make_case(weight_quant_mode=2, kv_quant_mode=1, cache_mode="BSND", bs_fused_flag=0,
               query_quant_mode=1, smooth_scales_cq_flag=1, query_norm_flag=1),
    _make_case(weight_quant_mode=2, kv_quant_mode=3, cache_mode="PA_BSND", bs_fused_flag=0,
               query_quant_mode=0, ckvkr_repo_mode=1, quant_scale_repo_mode=1,
               smooth_scales_cq_flag=1, query_norm_flag=0),
    _make_case(weight_quant_mode=3, kv_quant_mode=1, cache_mode="PA_BSND", bs_fused_flag=0,
               query_quant_mode=1, smooth_scales_cq_flag=0, query_norm_flag=0),
    _make_case(weight_quant_mode=3, kv_quant_mode=3, cache_mode="BSND", bs_fused_flag=0,
               query_quant_mode=0, ckvkr_repo_mode=1, quant_scale_repo_mode=1,
               smooth_scales_cq_flag=0, query_norm_flag=0),
    _make_case(weight_quant_mode=4, kv_quant_mode=0, cache_mode="PA_BSND", bs_fused_flag=0,
               query_quant_mode=0, smooth_scales_cq_flag=0, query_norm_flag=0),
    _make_case(weight_quant_mode=4, kv_quant_mode=1, cache_mode="BSND", bs_fused_flag=0,
               query_quant_mode=1, smooth_scales_cq_flag=1, query_norm_flag=1),
    _make_case(weight_quant_mode=4, kv_quant_mode=3, cache_mode="TND", bs_fused_flag=1,
               query_quant_mode=0, ckvkr_repo_mode=1, quant_scale_repo_mode=1,
               smooth_scales_cq_flag=1, query_norm_flag=1),
    _make_case(weight_quant_mode=5, kv_quant_mode=1, cache_mode="PA_BSND", bs_fused_flag=0,
               query_quant_mode=1, smooth_scales_cq_flag=1, query_norm_flag=1),
    _make_case(weight_quant_mode=5, kv_quant_mode=3, cache_mode="TND", bs_fused_flag=1,
               query_quant_mode=0, ckvkr_repo_mode=1, quant_scale_repo_mode=1,
               smooth_scales_cq_flag=1, query_norm_flag=0),
]


def _case_to_param_tuple(case_dict):
    return tuple(case_dict[name] for name in prologv3_generalized.PROLOGV3_PARAM_NAMES)


@pytest.mark.cpu_ref_align
@pytest.mark.parametrize(
    "case_dict",
    CPU_REF_ALIGN_CASES,
    ids=[
        f"w{case['weight_quant_mode']}_kv{case['kv_quant_mode']}_{case['cache_mode']}_"
        f"qnorm{case['query_norm_flag']}_smooth{case['smooth_scales_cq_flag']}"
        for case in CPU_REF_ALIGN_CASES
    ],
)
def test_cpu_ref_alignment(case_dict):
    if os.getenv("MLA_PROLOG_V3_ENABLE_CPU_REF_ALIGN", "1") != "1":
        pytest.skip("set MLA_PROLOG_V3_ENABLE_CPU_REF_ALIGN=1 to run temporary CPU alignment checks")

    skip_if_case_unsupported(case_dict)
    params = _case_to_param_tuple(case_dict)
    check_valid_param.validate_config(params)
    assert_cpu_refs_aligned(params)
