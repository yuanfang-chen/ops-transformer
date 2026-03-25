#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split-KN performance evaluation test suite.

Minimal test cases covering the key scenarios where Split-KN (2D N×K split)
impacts MM1 (MatmulCq) and MM2 (MatmulCkvKr) performance on Ascend 950.

Test dimensions:
  1. Token count (T): decode (1,4,8,32) vs prefill (128,512,1024)
  2. Quantization: BF16, INT8 full quant, MXFP8
  3. Head count (N): 128 (DeepSeek-V3), 16 (small model)

Usage:
    python test_split_kn_perf.py                     # Run all, table output
    python test_split_kn_perf.py --csv                # CSV output for spreadsheet
    python test_split_kn_perf.py --filter decode      # Only decode cases
    python test_split_kn_perf.py --filter prefill     # Only prefill cases
    python test_split_kn_perf.py --verbose             # Per-MM breakdown
"""

import sys
import os
import argparse
import math

sys.path.insert(0, os.path.dirname(__file__))

from perf_model import (
    ASCEND_950, OperatorParams, estimate_all_stages,
    SplitKNSpec, VectorOpTiming,
)
from tiling_sim import (
    compute_tiling, search_split_kn, SplitKNResult,
)
from pipeline_model import build_pipeline_dag


# ============================================================================
# Test case definitions
# ============================================================================

PERF_CASES = [
    # --- Decode scenarios (T small, Split-KN benefits MM1/MM2) ---
    # These are the PRIMARY beneficiaries of Split-KN
    {
        "name": "decode_bs1_bf16",
        "tag": "decode",
        "desc": "Decode BS=1, BF16 (baseline decode)",
        "batch_size": 1, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "decode_bs4_bf16",
        "tag": "decode",
        "desc": "Decode BS=4, BF16 (multi-request)",
        "batch_size": 1, "seq_len": 4, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "decode_bs8_bf16",
        "tag": "decode",
        "desc": "Decode BS=8, BF16 (batch decode)",
        "batch_size": 8, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "decode_bs32_bf16",
        "tag": "decode",
        "desc": "Decode BS=32, BF16 (large batch)",
        "batch_size": 32, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    # INT8 quantized decode
    {
        "name": "decode_bs1_int8",
        "tag": "decode",
        "desc": "Decode BS=1, INT8 full quant",
        "batch_size": 1, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 2, "kv_cache_quant_mode": 1,
    },
    {
        "name": "decode_bs8_int8",
        "tag": "decode",
        "desc": "Decode BS=8, INT8 full quant",
        "batch_size": 8, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 2, "kv_cache_quant_mode": 1,
    },
    # MXFP8 quantized decode
    {
        "name": "decode_bs1_fp8",
        "tag": "decode",
        "desc": "Decode BS=1, MXFP8",
        "batch_size": 1, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 3, "kv_cache_quant_mode": 0,
    },
    {
        "name": "decode_bs8_fp8",
        "tag": "decode",
        "desc": "Decode BS=8, MXFP8",
        "batch_size": 8, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 3, "kv_cache_quant_mode": 0,
    },

    # --- Prefill scenarios (T large, Split-KN may or may not help) ---
    {
        "name": "prefill_t128_bf16",
        "tag": "prefill",
        "desc": "Prefill T=128, BF16",
        "batch_size": 1, "seq_len": 128, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "prefill_t512_bf16",
        "tag": "prefill",
        "desc": "Prefill T=512, BF16",
        "batch_size": 1, "seq_len": 512, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "prefill_t1024_bf16",
        "tag": "prefill",
        "desc": "Prefill T=1024, BF16",
        "batch_size": 1, "seq_len": 1024, "head_num": 128,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "prefill_t128_int8",
        "tag": "prefill",
        "desc": "Prefill T=128, INT8 full quant",
        "batch_size": 1, "seq_len": 128, "head_num": 128,
        "weight_quant_mode": 2, "kv_cache_quant_mode": 1,
    },

    # --- Small model scenarios (N=16, smaller He) ---
    {
        "name": "small_decode_bs1",
        "tag": "small",
        "desc": "Small model decode BS=1 (N=16, He=2048)",
        "batch_size": 1, "seq_len": 1, "head_num": 16, "He": 2048,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "small_decode_bs8",
        "tag": "small",
        "desc": "Small model decode BS=8 (N=16, He=2048)",
        "batch_size": 8, "seq_len": 1, "head_num": 16, "He": 2048,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },

    # --- Edge cases ---
    {
        "name": "decode_bs1_bf16_n8",
        "tag": "edge",
        "desc": "Decode BS=1, N=8 heads (GQA-like)",
        "batch_size": 1, "seq_len": 1, "head_num": 8,
        "weight_quant_mode": 0, "kv_cache_quant_mode": 0,
    },
    {
        "name": "decode_bs1_int8_kvpertile",
        "tag": "edge",
        "desc": "Decode BS=1, INT8 + KV per-tile quant",
        "batch_size": 1, "seq_len": 1, "head_num": 128,
        "weight_quant_mode": 2, "kv_cache_quant_mode": 3,
    },
]


# ============================================================================
# Evaluation logic
# ============================================================================

def build_params(case: dict) -> OperatorParams:
    """Build OperatorParams from test case dict."""
    return OperatorParams(
        batch_size=case["batch_size"],
        seq_len=case["seq_len"],
        head_num=case["head_num"],
        He=case.get("He", 7168),
        weight_quant_mode=case["weight_quant_mode"],
        kv_cache_quant_mode=case["kv_cache_quant_mode"],
        query_quant_mode=case.get("query_quant_mode", 0),
    )


def evaluate_case(case: dict, hw=ASCEND_950):
    """Evaluate Split-N baseline vs best Split-KN for a single case.

    Returns dict with:
      - baseline_us: Split-N pipeline total
      - best_kn_us: Best Split-KN pipeline total
      - best_kn_label: Config label of best Split-KN
      - speedup: baseline / best_kn
      - mm1_baseline, mm2_baseline, mm3_baseline: per-MM baseline times
      - mm1_best, mm2_best: per-MM best Split-KN times
      - mm1_speedup, mm2_speedup: per-MM speedups
    """
    params = build_params(case)
    tiling = compute_tiling(params, hw)

    # Baseline: Split-N
    baseline_stages = estimate_all_stages(params, tiling, hw)
    baseline_pipeline = build_pipeline_dag(params, tiling, hw, baseline_stages)
    baseline_us = baseline_pipeline.total_us

    # Split-KN search
    kn_results = search_split_kn(params, tiling, hw)

    if not kn_results:
        return {
            "baseline_us": baseline_us,
            "best_kn_us": baseline_us,
            "best_kn_label": "N/A",
            "speedup": 1.0,
            "mm1_baseline": baseline_stages["MM1_Cq"].total_us,
            "mm2_baseline": baseline_stages["MM2_CkvKr"].total_us,
            "mm3_baseline": baseline_stages["MM3_QcQr"].total_us,
            "mm1_best": baseline_stages["MM1_Cq"].total_us,
            "mm2_best": baseline_stages["MM2_CkvKr"].total_us,
            "mm1_speedup": 1.0,
            "mm2_speedup": 1.0,
        }

    best = kn_results[0]
    baseline_kn = next((r for r in kn_results if r.label == "baseline"), None)

    mm1_base = baseline_stages["MM1_Cq"].total_us
    mm2_base = baseline_stages["MM2_CkvKr"].total_us
    mm3_base = baseline_stages["MM3_QcQr"].total_us

    return {
        "baseline_us": baseline_us,
        "best_kn_us": best.pipeline_us,
        "best_kn_label": best.label,
        "speedup": baseline_us / best.pipeline_us if best.pipeline_us > 0 else 0,
        "mm1_baseline": mm1_base,
        "mm2_baseline": mm2_base,
        "mm3_baseline": mm3_base,
        "mm1_best": best.mm1_cube_us,
        "mm2_best": best.mm2_cube_us,
        "mm1_speedup": mm1_base / best.mm1_cube_us if best.mm1_cube_us > 0 else 0,
        "mm2_speedup": mm2_base / best.mm2_cube_us if best.mm2_cube_us > 0 else 0,
    }


# ============================================================================
# Output formatters
# ============================================================================

def print_table(results, verbose=False):
    """Print results as a formatted table."""
    print("=" * 110)
    print("  Split-KN Performance Evaluation — Ascend 950")
    print("=" * 110)
    print()

    header = (f"  {'Case':<30} {'T':>5} {'Quant':<6} | {'Baseline':>8} {'SplitKN':>8} "
              f"{'Speedup':>7} | {'Best Config':<22}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for case, result in results:
        T = case["batch_size"] * case["seq_len"]
        qm = case["weight_quant_mode"]
        quant = "BF16" if qm == 0 else ("INT8" if qm <= 2 else "FP8")
        speedup = result["speedup"]
        marker = " ★" if speedup > 1.2 else (" ▲" if speedup > 1.05 else "")

        print(f"  {case['name']:<30} {T:>5} {quant:<6} | "
              f"{result['baseline_us']:>7.2f} {result['best_kn_us']:>7.2f} "
              f"{speedup:>6.2f}x | {result['best_kn_label']:<22}{marker}")

    print()

    if verbose:
        print("  Per-MM Breakdown:")
        print(f"  {'Case':<30} | {'MM1 base':>8} {'MM1 KN':>8} {'MM1 spd':>7} | "
              f"{'MM2 base':>8} {'MM2 KN':>8} {'MM2 spd':>7} | {'MM3':>8}")
        print("  " + "-" * 105)
        for case, result in results:
            print(f"  {case['name']:<30} | "
                  f"{result['mm1_baseline']:>7.2f} {result['mm1_best']:>7.2f} "
                  f"{result['mm1_speedup']:>6.1f}x | "
                  f"{result['mm2_baseline']:>7.2f} {result['mm2_best']:>7.2f} "
                  f"{result['mm2_speedup']:>6.1f}x | "
                  f"{result['mm3_baseline']:>7.2f}")
        print()

    # Summary statistics
    speedups = [r["speedup"] for _, r in results]
    decode_speedups = [r["speedup"] for c, r in results if c["tag"] == "decode"]
    prefill_speedups = [r["speedup"] for c, r in results if c["tag"] == "prefill"]

    print("  Summary:")
    print(f"    Overall:  avg {sum(speedups)/len(speedups):.2f}x, "
          f"max {max(speedups):.2f}x, min {min(speedups):.2f}x")
    if decode_speedups:
        print(f"    Decode:   avg {sum(decode_speedups)/len(decode_speedups):.2f}x, "
              f"max {max(decode_speedups):.2f}x")
    if prefill_speedups:
        print(f"    Prefill:  avg {sum(prefill_speedups)/len(prefill_speedups):.2f}x, "
              f"max {max(prefill_speedups):.2f}x")
    print()
    print("  Legend: ★ = >1.2x speedup, ▲ = >1.05x speedup")


def print_csv(results):
    """Print results as CSV for spreadsheet import."""
    print("case,tag,T,quant,head_num,He,baseline_us,split_kn_us,speedup,"
          "best_config,mm1_base,mm1_kn,mm1_spd,mm2_base,mm2_kn,mm2_spd,mm3")
    for case, result in results:
        T = case["batch_size"] * case["seq_len"]
        qm = case["weight_quant_mode"]
        quant = "BF16" if qm == 0 else ("INT8" if qm <= 2 else "FP8")
        print(f"{case['name']},{case['tag']},{T},{quant},{case['head_num']},"
              f"{case.get('He', 7168)},"
              f"{result['baseline_us']:.2f},{result['best_kn_us']:.2f},"
              f"{result['speedup']:.3f},{result['best_kn_label']},"
              f"{result['mm1_baseline']:.2f},{result['mm1_best']:.2f},"
              f"{result['mm1_speedup']:.2f},"
              f"{result['mm2_baseline']:.2f},{result['mm2_best']:.2f},"
              f"{result['mm2_speedup']:.2f},{result['mm3_baseline']:.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split-KN performance evaluation for MLA Prolog V3")
    parser.add_argument("--csv", action="store_true",
                        help="Output as CSV")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-MM breakdown")
    parser.add_argument("--filter", type=str, default=None,
                        choices=["decode", "prefill", "small", "edge"],
                        help="Only run cases matching this tag")
    parser.add_argument("--case", type=str, default=None,
                        help="Run a single named case")
    args = parser.parse_args()

    cases = PERF_CASES
    if args.filter:
        cases = [c for c in cases if c["tag"] == args.filter]
    if args.case:
        cases = [c for c in cases if c["name"] == args.case]

    if not cases:
        print("No matching test cases found.")
        return

    results = []
    for case in cases:
        result = evaluate_case(case)
        results.append((case, result))

    if args.csv:
        print_csv(results)
    else:
        print_table(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
