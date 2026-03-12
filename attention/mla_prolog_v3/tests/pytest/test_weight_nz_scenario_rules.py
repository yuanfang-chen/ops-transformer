#!/usr/bin/env python3

import pytest

import gen_st_cases
import prologv3_generalized
import weight_nz_scenario_rules as rules


def _make_case(**overrides):
    case = dict(gen_st_cases.FIXED_FIELDS)
    case.update(
        {
            "batch_size": 1,
            "He": 1024,
            "q_head_num": 1,
            "q_seq": 1,
            "block_size": 16,
            "cache_mode": "BSND",
            "bs_fused_flag": 0,
            "weight_quant_mode": 0,
            "kv_quant_mode": 0,
            "smooth_scales_cq_flag": 0,
            "query_norm_flag": 0,
            "qc_qr_scale": 1.0,
            "kc_scale": 1.0,
        }
    )
    case.update(overrides)
    query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode = gen_st_cases.derive_query_repo_modes(
        int(case["weight_quant_mode"]),
        int(case["kv_quant_mode"]),
    )
    case["query_quant_mode"] = query_quant_mode
    case["ckvkr_repo_mode"] = ckvkr_repo_mode
    case["quant_scale_repo_mode"] = quant_scale_repo_mode
    return case


def _make_params(**overrides):
    case = _make_case(**overrides)
    return tuple(case[name] for name in prologv3_generalized.PROLOGV3_PARAM_NAMES)


def test_k_nope_clip_alpha_policy_only_for_half_and_int8_full_pertile():
    half_policy = rules.get_weight_nz_optional_input_policy(
        _make_case(weight_quant_mode=1, kv_quant_mode=3, cache_mode="TND", bs_fused_flag=1)
    )
    int8_policy = rules.get_weight_nz_optional_input_policy(
        _make_case(weight_quant_mode=2, kv_quant_mode=3, cache_mode="TND", bs_fused_flag=1)
    )
    mxfp8_policy = rules.get_weight_nz_optional_input_policy(
        _make_case(weight_quant_mode=3, kv_quant_mode=3, cache_mode="TND", bs_fused_flag=1)
    )

    assert half_policy["k_nope_clip_alpha"]
    assert int8_policy["k_nope_clip_alpha"]
    assert not mxfp8_policy["k_nope_clip_alpha"]


@pytest.mark.parametrize("weight_quant_mode", [4, 5])
def test_validate_weight_nz_npu_combo_rejects_cpu_only_modes(weight_quant_mode):
    is_valid, reason = rules.validate_weight_nz_npu_combo("PA_BSND", weight_quant_mode, 0, 0, 0, 0)
    assert not is_valid
    assert "0,1,2,3" in reason


def test_reduced_dimension_scenario_audit():
    legal_count = 0
    for weight_quant_mode in range(6):
        for kv_quant_mode in range(4):
            derived_modes = rules.derive_weight_nz_query_repo_modes(weight_quant_mode, kv_quant_mode)
            if derived_modes is None:
                continue
            query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode = derived_modes
            for cache_mode in rules.WEIGHT_NZ_NPU_SUPPORTED_CACHE_MODES:
                is_valid, _ = rules.validate_weight_nz_npu_combo(
                    cache_mode,
                    weight_quant_mode,
                    kv_quant_mode,
                    query_quant_mode,
                    ckvkr_repo_mode,
                    quant_scale_repo_mode,
                )
                if not is_valid:
                    continue
                legal_count += 1
                scenario = rules.classify_weight_nz_npu_quant_scenario(weight_quant_mode, kv_quant_mode)
                case = _make_case(
                    cache_mode=cache_mode,
                    bs_fused_flag=int(cache_mode in rules.PA_BLK_CACHE_MODES),
                    weight_quant_mode=weight_quant_mode,
                    kv_quant_mode=kv_quant_mode,
                )
                policy = rules.get_weight_nz_optional_input_policy(case)
                assert policy["k_nope_clip_alpha"] == (
                    scenario in {rules.SCENARIO_HALF_PER_TILE, rules.SCENARIO_INT8_FULL_PER_TILE}
                )
                assert policy["actual_seq_len"] == (cache_mode in rules.PA_BLK_CACHE_MODES)
    assert legal_count > 0


@pytest.mark.parametrize(
    "weight_quant_mode, expect_present",
    [
        (1, True),
        (2, True),
        (3, False),
    ],
)
def test_default_case_payload_follows_k_nope_clip_alpha_policy(weight_quant_mode, expect_present):
    params = _make_params(
        cache_mode="TND",
        bs_fused_flag=1,
        weight_quant_mode=weight_quant_mode,
        kv_quant_mode=3,
    )
    try:
        payload = prologv3_generalized._build_default_case_payload(
            params,
            validate_quant_combo=False,
            runtime_device="cpu",
        )
    except pytest.skip.Exception as exc:
        pytest.skip(str(exc))

    assert (payload["runtime_inputs"]["k_nope_clip_alpha"] is not None) is expect_present
