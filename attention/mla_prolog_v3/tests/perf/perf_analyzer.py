#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for MLA Prolog V3 performance analysis.

Usage:
    python perf_analyzer.py --batch-size 2 --seq-len 4 --head-num 32 --He 7168 --mode full
    python perf_analyzer.py --from-testcase base_default --mode bound
    python perf_analyzer.py --from-testcase base_default --mode pipeline
    python perf_analyzer.py --from-testcase base_default --mode estimate
    python perf_analyzer.py --from-testcase base_default --mode advice
    python perf_analyzer.py --from-testcase base_default --mode search
"""

import argparse
import sys
import os

# Add parent pytest dir to path for testcase import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pytest"))

from perf_model import (
    AscendHWSpec, ASCEND_910B, OperatorParams, MatMulTiming, VectorOpTiming,
    estimate_all_stages, is_full_quant, is_int8_quant, is_mxfp8,
)
from tiling_sim import TilingConfig, compute_tiling, search_best_tiling
from pipeline_model import (
    build_pipeline_dag, estimate_kernel_time, format_timeline, PipelineResult,
)


def params_from_testcase(name: str) -> OperatorParams:
    """Load OperatorParams from a named test case in testcases.py."""
    try:
        from testcases import TEST_PARAMS
    except ImportError:
        print(f"Error: cannot import testcases.py from ../pytest/", file=sys.stderr)
        sys.exit(1)

    if name not in TEST_PARAMS:
        print(f"Error: unknown test case '{name}'. Available: {list(TEST_PARAMS.keys())}", file=sys.stderr)
        sys.exit(1)

    tc = TEST_PARAMS[name]
    return OperatorParams(
        batch_size=tc["batch_size"][0],
        seq_len=tc["seq_len"][0],
        head_num=tc["head_num"][0],
        He=tc["He"][0],
        weight_quant_mode=tc["weight_quant_mode"][0],
        kv_cache_quant_mode=tc["kv_cache_quant_mode"][0],
        query_quant_mode=tc["query_quant_mode"][0],
    )


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def mode_bound(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 1: Per-stage resource analysis table."""
    print_header("BOUND ANALYSIS — Per-Stage Resource Breakdown")

    stages = estimate_all_stages(params, tiling, hw)

    print(f"Parameters: T={params.T}, He={params.He}, N={params.N}, Hcq={params.Hcq}, Hckv={params.Hckv}, D={params.D}, Dr={params.Dr}")
    print(f"Quant: weight={params.weight_quant_mode}, kv_cache={params.kv_cache_quant_mode}, query={params.query_quant_mode}")
    print(f"Tiling: stepBatch={tiling.step_batch_size}")
    print()

    # MatMul stages
    mm_header = f"{'Stage':<16} {'M':>5} {'K':>6} {'N':>7} {'Cores':>5} | {'MTE1':>8} {'Cube':>8} {'FixPipe':>8} {'Total':>8} | {'Bound':<8} {'A src':<5}"
    print(mm_header)
    print("-" * len(mm_header))

    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        t = stages[name]
        print(f"{t.name:<16} {t.M:>5} {t.K:>6} {t.N:>7} {t.active_cores:>5} | "
              f"{t.mte1_us:>7.2f} {t.cube_us:>7.2f} {t.fixpipe_us:>7.2f} {t.total_us:>7.2f} | "
              f"{t.bound:<8} {t.a_source:<5}")

    print()

    # Vector stages
    vec_header = f"{'Stage':<16} | {'MTE2':>8} {'Vector':>8} {'MTE3':>8} {'Total':>8} | {'Bound':<8}"
    print(vec_header)
    print("-" * len(vec_header))

    vec_names = ["RmsNormCq", "RmsNormCkvKr", "RopeKr", "ScatterCkvKr",
                 "DequantQc", "RopeQr", "DynamicQuantQn", "MulQr"]
    for name in vec_names:
        if name not in stages:
            continue
        t = stages[name]
        print(f"{t.name:<16} | "
              f"{t.mte2_load_us:>7.2f} {t.vec_compute_us:>7.2f} {t.mte3_store_us:>7.2f} {t.total_us:>7.2f} | "
              f"{t.bound:<8}")


def mode_pipeline(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 2: Timeline visualization."""
    print_header("PIPELINE TIMELINE — AIC/AIV Execution Overlap")

    result = build_pipeline_dag(params, tiling, hw)
    print(format_timeline(result))


def mode_estimate(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 3: Kernel duration estimate."""
    print_header("KERNEL DURATION ESTIMATE")

    total_us = estimate_kernel_time(params, tiling, hw)
    T = params.T
    step = tiling.step_batch_size
    num_steps = (T + step - 1) // step

    print(f"Parameters: T={T}, He={params.He}, N={params.N}")
    print(f"stepBatchSize={step}, num_steps={num_steps}")
    print()
    print(f"Estimated kernel time: {total_us:.2f} us ({total_us / 1000:.3f} ms)")

    if num_steps > 1:
        result = build_pipeline_dag(params, tiling, hw)
        print(f"  Single step: {result.total_us:.2f} us")
        print(f"  Total ({num_steps} steps): {total_us:.2f} us")


def mode_advice(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 4: Improvement suggestions."""
    print_header("PERFORMANCE ADVICE")

    stages = estimate_all_stages(params, tiling, hw)
    result = build_pipeline_dag(params, tiling, hw, stages)
    advice = []

    # Check each matmul for bottleneck
    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        t = stages[name]
        if t.bound in ("HBM", "L2"):
            if params.weight_quant_mode == 0:
                advice.append(
                    f"[{name}] MTE1-bound ({t.bound}): consider INT8/FP8 quantization to halve weight load "
                    f"({t.mte1_us:.2f} us MTE1 vs {t.cube_us:.2f} us Cube)")
            else:
                advice.append(
                    f"[{name}] MTE1-bound ({t.bound}): already quantized. Consider restructuring tiling to "
                    f"improve L2 reuse.")

    # Check core utilization
    for name, block_num_attr in [("MM1", "mm1_block_num"), ("MM2", "mm2_block_num"),
                                  ("MM3", "mm3_block_num"), ("MM4", "mm4_block_num")]:
        cores = getattr(tiling, block_num_attr)
        if cores < hw.aic_num * 0.5:
            advice.append(
                f"[{name}] Low core utilization: {cores}/{hw.aic_num} AIC cores active. "
                f"Consider rebalancing N-split to use more cores.")

    # Check AIC/AIV balance
    if result.aiv_busy_us > 0 and result.aic_busy_us > 0:
        ratio = result.aiv_busy_us / result.aic_busy_us
        if ratio < 0.3:
            advice.append(
                f"AIV largely idle ({result.aiv_busy_us:.2f} vs AIC {result.aic_busy_us:.2f} us). "
                f"Vector pipeline is underutilized — consider fusing more post-MM vector ops.")
        elif ratio > 3.0:
            advice.append(
                f"AIC largely idle ({result.aic_busy_us:.2f} vs AIV {result.aiv_busy_us:.2f} us). "
                f"Cube pipeline is underutilized — potential for more matmul overlap.")

    # Check sync overhead
    total_us = result.total_us
    if total_us > 0 and result.sync_overhead_us / total_us > 0.15:
        advice.append(
            f"Sync overhead is {result.sync_overhead_us:.2f} us ({result.sync_overhead_us/total_us*100:.1f}% of total). "
            f"Consider reducing cross-core synchronization events.")

    # Check fixpipe bound
    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        t = stages[name]
        if t.bound == "FIXPIPE":
            advice.append(
                f"[{name}] FixPipe-bound: store bandwidth is the bottleneck. "
                f"Consider fusing with downstream vector ops to avoid GM round-trip.")

    # Check scatter efficiency
    scatter = stages.get("ScatterCkvKr")
    if scatter and scatter.bound == "MTE3":
        advice.append(
            f"[ScatterCkvKr] Scatter store is the bottleneck ({scatter.mte3_store_us:.2f} us). "
            f"Indexed writes have {hw.scatter_efficiency:.0%} efficiency. "
            f"Consider contiguous cache layout if possible.")

    if not advice:
        advice.append("No obvious bottlenecks detected. Performance appears well-balanced.")

    for i, a in enumerate(advice, 1):
        print(f"  {i}. {a}")
        print()


def mode_search(params: OperatorParams, hw: AscendHWSpec):
    """Mode 5: Best tiling search."""
    print_header("TILING SEARCH — Best Configuration")

    candidates = [8, 16, 32, 64, 128]
    results = search_best_tiling(params, hw, step_candidates=candidates)

    print(f"Parameters: T={params.T}, He={params.He}, N={params.N}")
    print(f"Searching stepBatchSize in {candidates}...")
    print()

    header = f"{'Rank':>4} {'Step':>6} {'MM1c':>5} {'MM2c':>5} {'MM3c':>5} {'MM4c':>5} {'Est(us)':>10} {'GroupOpt':>8} {'DeqOpt':>7}"
    print(header)
    print("-" * len(header))

    for i, (cfg, est_us) in enumerate(results, 1):
        marker = " <-- best" if i == 1 else ""
        print(f"{i:>4} {cfg.step_batch_size:>6} {cfg.mm1_block_num:>5} {cfg.mm2_block_num:>5} "
              f"{cfg.mm3_block_num:>5} {cfg.mm4_block_num:>5} {est_us:>10.2f} "
              f"{'Y' if cfg.enable_group_compute_opt else 'N':>8} "
              f"{'Y' if cfg.enable_dequant_opt else 'N':>7}{marker}")

    if results:
        print()
        best_cfg, best_us = results[0]
        print(f"Best: stepBatchSize={best_cfg.step_batch_size}, estimated {best_us:.2f} us")
        print()
        print("Tiling details:")
        print(best_cfg.summary())


def main():
    parser = argparse.ArgumentParser(
        description="MLA Prolog V3 Performance Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Parameter sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-testcase", type=str, metavar="NAME",
                     help="Load parameters from a named test case in testcases.py")
    src.add_argument("--batch-size", type=int,
                     help="Batch size (requires other params too)")

    # Manual parameters
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--He", type=int, default=7168)
    parser.add_argument("--Hcq", type=int, default=1536)
    parser.add_argument("--Hckv", type=int, default=512)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--Dr", type=int, default=64)
    parser.add_argument("--weight-quant-mode", type=int, default=0)
    parser.add_argument("--kv-cache-quant-mode", type=int, default=0)
    parser.add_argument("--query-quant-mode", type=int, default=0)

    # Analysis mode
    parser.add_argument("--mode", type=str, default="full",
                        choices=["bound", "pipeline", "estimate", "advice", "search", "full"],
                        help="Analysis mode (default: full = all modes)")

    # Hardware override
    parser.add_argument("--aic-num", type=int, default=24)
    parser.add_argument("--aiv-num", type=int, default=48)

    args = parser.parse_args()

    # Build params
    if args.from_testcase:
        params = params_from_testcase(args.from_testcase)
    else:
        params = OperatorParams(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            head_num=args.head_num,
            He=args.He,
            Hcq=args.Hcq,
            Hckv=args.Hckv,
            D=args.D,
            Dr=args.Dr,
            weight_quant_mode=args.weight_quant_mode,
            kv_cache_quant_mode=args.kv_cache_quant_mode,
            query_quant_mode=args.query_quant_mode,
        )

    hw = AscendHWSpec(aic_num=args.aic_num, aiv_num=args.aiv_num)

    # Print params summary
    print(f"MLA Prolog V3 Performance Analysis")
    print(f"  B={params.batch_size}, S={params.seq_len}, T={params.T}, N={params.N}, He={params.He}")
    print(f"  Hcq={params.Hcq}, Hckv={params.Hckv}, D={params.D}, Dr={params.Dr}")
    print(f"  weight_quant={params.weight_quant_mode}, kv_quant={params.kv_cache_quant_mode}, "
          f"query_quant={params.query_quant_mode}")
    print(f"  Resolved quant_mode: {params.quant_mode.name}")
    print(f"  HW: {hw.name}, AIC={hw.aic_num}, AIV={hw.aiv_num}")

    if args.mode == "search":
        mode_search(params, hw)
        return

    tiling = compute_tiling(params, hw)
    print(f"\nTiling:")
    print(f"  {tiling.summary()}")

    modes = {
        "bound": mode_bound,
        "pipeline": mode_pipeline,
        "estimate": mode_estimate,
        "advice": mode_advice,
    }

    if args.mode == "full":
        for mode_fn in [mode_bound, mode_pipeline, mode_estimate, mode_advice]:
            mode_fn(params, tiling, hw)
        mode_search(params, hw)
    else:
        modes[args.mode](params, tiling, hw)


if __name__ == "__main__":
    main()
