#!/usr/bin/env python3

from pathlib import Path

import gen_st_cases


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

    if int(case.get("query_norm_flag", 0)) == 1:
        case["qc_qr_scale"] = 1.1
        case["kc_scale"] = 1.1
    else:
        case["qc_qr_scale"] = 1.0
        case["kc_scale"] = 1.0
    return case


def test_build_case_tags_marks_multi_step_loop_and_tail_for_t_gt_128():
    case = _make_case(batch_size=9, q_seq=16, cache_mode="PA_BLK_BSND", bs_fused_flag=1)
    hw = gen_st_cases.HardwareProfile()

    tags = gen_st_cases.build_case_tags(case, hw)

    assert "vector:multi_step_loop:1" in tags
    assert "vector:step_batch_tail:1" in tags


def test_build_case_tags_marks_no_multi_step_loop_for_t_le_128():
    case = _make_case(batch_size=8, q_seq=16)
    hw = gen_st_cases.HardwareProfile()

    tags = gen_st_cases.build_case_tags(case, hw)

    assert "vector:multi_step_loop:0" in tags


def test_actual_seq_mode_only_for_fused_pa_blk():
    fused_pa_blk = _make_case(cache_mode="PA_BLK_BSND", bs_fused_flag=1)
    fused_non_pa_blk = _make_case(cache_mode="TND", bs_fused_flag=1)
    non_fused_pa_blk = _make_case(cache_mode="PA_BLK_NZ", bs_fused_flag=0)
    hw = gen_st_cases.HardwareProfile()

    assert "tiling:actual_seq_mode:en_q_len" in gen_st_cases.build_case_tags(fused_pa_blk, hw)
    assert "tiling:actual_seq_mode:disabled" in gen_st_cases.build_case_tags(fused_non_pa_blk, hw)
    assert "tiling:actual_seq_mode:disabled" in gen_st_cases.build_case_tags(non_fused_pa_blk, hw)


def test_validate_positive_case_mirrors_pertile_repo_and_cache_rules():
    valid_pertile = _make_case(weight_quant_mode=1, kv_quant_mode=3, cache_mode="PA_BSND")
    invalid_pertile_cache = _make_case(weight_quant_mode=1, kv_quant_mode=3, cache_mode="PA_NZ")
    invalid_non_pertile_repo = _make_case(weight_quant_mode=0, kv_quant_mode=0)
    invalid_non_pertile_repo["ckvkr_repo_mode"] = 1
    invalid_non_pertile_repo["quant_scale_repo_mode"] = 1

    assert gen_st_cases.validate_positive_case(valid_pertile)
    assert not gen_st_cases.validate_positive_case(invalid_pertile_cache)
    assert not gen_st_cases.validate_positive_case(invalid_non_pertile_repo)


def test_default_generation_retains_multi_step_loop_coverage(tmp_path: Path):
    factor_space = gen_st_cases._normalize_factor_space(gen_st_cases.DEFAULT_FACTOR_SPACE)
    hw = gen_st_cases.HardwareProfile()
    candidates = gen_st_cases.enumerate_positive_candidates(factor_space, hw)
    universe = set()
    for candidate in candidates:
        universe |= candidate.tags
    selected = gen_st_cases.deterministic_set_cover(candidates, universe)

    assert any("vector:multi_step_loop:1" in case.tags for case in selected)

    output_testcases = tmp_path / "testcases.py"
    output_report = tmp_path / "st_case_coverage_report.md"
    gen_st_cases.generate(output_testcases, output_report, factor_space, hw)

    report_text = output_report.read_text(encoding="utf-8")
    assert "multi-step outer loop yes/no" in report_text
