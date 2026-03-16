#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline DAG model with critical path analysis for MLA Prolog V3."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .perf_model import (
        AscendHWSpec, ASCEND_910B, OperatorParams, MatMulTiming, VectorOpTiming,
        estimate_all_stages, is_full_quant, is_int8_quant, is_mxfp8,
        MatmulBlockSpec, get_default_block_specs, estimate_matmul_detailed,
    )
    from .tiling_sim import TilingConfig, compute_tiling, ceil_div
except ImportError:
    from perf_model import (
        AscendHWSpec, ASCEND_910B, OperatorParams, MatMulTiming, VectorOpTiming,
        estimate_all_stages, is_full_quant, is_int8_quant, is_mxfp8,
        MatmulBlockSpec, get_default_block_specs, estimate_matmul_detailed,
    )
    from tiling_sim import TilingConfig, compute_tiling, ceil_div


@dataclass
class PipelineStage:
    name: str
    pipe: str  # "AIC" or "AIV"
    duration_us: float
    depends_on: List[str] = field(default_factory=list)
    start_us: float = 0.0
    end_us: float = 0.0
    on_critical_path: bool = False


@dataclass
class PipelineResult:
    stages: Dict[str, PipelineStage]
    critical_path: List[str]
    total_us: float
    aic_busy_us: float
    aiv_busy_us: float
    sync_overhead_us: float


def build_pipeline_dag(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
    stage_timings: Optional[Dict] = None,
) -> PipelineResult:
    """Build and solve the AIC/AIV pipeline DAG.

    Pipeline from kernel_mla_prolog_split_n.h lines 786-910:

    AIC: MM1 -> signal(CQ) -> MM2 -> signal(CKVKR) -> wait(RMSNORM_CQ) -> MM3 -> signal(QCQR) -> wait(DEQUANT_QC) -> MM4
    AIV: CopyInSinCos -> wait(CQ) -> RmsNormCq -> signal(RMSNORM_CQ) -> wait(CKVKR) -> RmsNormCkvKr+RopeKr+Scatter
         -> wait(QCQR) -> [DequantQc] -> signal(DEQUANT_QC) -> RopeQr -> [DynamicQuantQn+MulQr]
    """
    if stage_timings is None:
        stage_timings = estimate_all_stages(params, tiling, hw)

    qm = params.quant_mode
    has_dequant = is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm)
    has_dynamic_quant = params.query_quant_mode == 1

    sync_us = hw.sync_overhead_us

    def get_us(name: str) -> float:
        t = stage_timings.get(name)
        if t is None:
            return 0.0
        return t.total_us

    stages: Dict[str, PipelineStage] = {}

    # --- AIC pipeline ---
    stages["MM1_Cq"] = PipelineStage(
        "MM1_Cq", "AIC", get_us("MM1_Cq"))

    stages["sync_signal_CQ"] = PipelineStage(
        "sync_signal_CQ", "AIC", sync_us, depends_on=["MM1_Cq"])

    stages["MM2_CkvKr"] = PipelineStage(
        "MM2_CkvKr", "AIC", get_us("MM2_CkvKr"), depends_on=["sync_signal_CQ"])

    stages["sync_signal_CKVKR"] = PipelineStage(
        "sync_signal_CKVKR", "AIC", sync_us, depends_on=["MM2_CkvKr"])

    # AIC waits for vector RmsNormCq before MM3
    stages["aic_wait_RMSNORM_CQ"] = PipelineStage(
        "aic_wait_RMSNORM_CQ", "AIC", sync_us,
        depends_on=["sync_signal_CKVKR", "sync_signal_RMSNORM_CQ"])

    stages["MM3_QcQr"] = PipelineStage(
        "MM3_QcQr", "AIC", get_us("MM3_QcQr"),
        depends_on=["aic_wait_RMSNORM_CQ"])

    stages["sync_signal_QCQR"] = PipelineStage(
        "sync_signal_QCQR", "AIC", sync_us, depends_on=["MM3_QcQr"])

    # AIC waits for vector DequantQc before MM4
    if has_dequant:
        stages["aic_wait_DEQUANT_QC"] = PipelineStage(
            "aic_wait_DEQUANT_QC", "AIC", sync_us,
            depends_on=["sync_signal_QCQR", "sync_signal_DEQUANT_QC"])
        stages["MM4_Qn"] = PipelineStage(
            "MM4_Qn", "AIC", get_us("MM4_Qn"),
            depends_on=["aic_wait_DEQUANT_QC"])
    else:
        stages["aic_wait_QCQR_for_rope"] = PipelineStage(
            "aic_wait_QCQR_for_rope", "AIC", sync_us,
            depends_on=["sync_signal_QCQR"])
        stages["MM4_Qn"] = PipelineStage(
            "MM4_Qn", "AIC", get_us("MM4_Qn"),
            depends_on=["aic_wait_QCQR_for_rope"])

    # --- AIV pipeline ---
    stages["CopyInSinCos"] = PipelineStage(
        "CopyInSinCos", "AIV", 0.5)  # Small constant

    stages["aiv_wait_CQ"] = PipelineStage(
        "aiv_wait_CQ", "AIV", sync_us,
        depends_on=["CopyInSinCos", "sync_signal_CQ"])

    stages["RmsNormCq"] = PipelineStage(
        "RmsNormCq", "AIV", get_us("RmsNormCq"),
        depends_on=["aiv_wait_CQ"])

    stages["sync_signal_RMSNORM_CQ"] = PipelineStage(
        "sync_signal_RMSNORM_CQ", "AIV", sync_us,
        depends_on=["RmsNormCq"])

    stages["aiv_wait_CKVKR"] = PipelineStage(
        "aiv_wait_CKVKR", "AIV", sync_us,
        depends_on=["sync_signal_RMSNORM_CQ", "sync_signal_CKVKR"])

    stages["RmsNormCkvKr"] = PipelineStage(
        "RmsNormCkvKr", "AIV", get_us("RmsNormCkvKr"),
        depends_on=["aiv_wait_CKVKR"])

    stages["RopeKr"] = PipelineStage(
        "RopeKr", "AIV", get_us("RopeKr"),
        depends_on=["RmsNormCkvKr"])

    stages["ScatterCkvKr"] = PipelineStage(
        "ScatterCkvKr", "AIV", get_us("ScatterCkvKr"),
        depends_on=["RopeKr"])

    # After MM3 completes: Dequant/Cast Qc, then RoPE Qr
    if has_dequant:
        stages["aiv_wait_QCQR"] = PipelineStage(
            "aiv_wait_QCQR", "AIV", sync_us,
            depends_on=["ScatterCkvKr", "sync_signal_QCQR"])

        stages["DequantQc"] = PipelineStage(
            "DequantQc", "AIV", get_us("DequantQc"),
            depends_on=["aiv_wait_QCQR"])

        stages["sync_signal_DEQUANT_QC"] = PipelineStage(
            "sync_signal_DEQUANT_QC", "AIV", sync_us,
            depends_on=["DequantQc"])

        stages["RopeQr"] = PipelineStage(
            "RopeQr", "AIV", get_us("RopeQr"),
            depends_on=["sync_signal_DEQUANT_QC"])
    else:
        stages["aiv_wait_QCQR_vec"] = PipelineStage(
            "aiv_wait_QCQR_vec", "AIV", sync_us,
            depends_on=["ScatterCkvKr", "sync_signal_QCQR"])

        stages["RopeQr"] = PipelineStage(
            "RopeQr", "AIV", get_us("RopeQr"),
            depends_on=["aiv_wait_QCQR_vec"])

    if has_dynamic_quant:
        stages["DynamicQuantQn"] = PipelineStage(
            "DynamicQuantQn", "AIV", get_us("DynamicQuantQn"),
            depends_on=["RopeQr", "MM4_Qn"])

        stages["MulQr"] = PipelineStage(
            "MulQr", "AIV", get_us("MulQr"),
            depends_on=["DynamicQuantQn"])

    # --- Solve DAG: forward pass longest-path ---
    # Topological sort via Kahn's algorithm
    in_degree = {name: 0 for name in stages}
    for s in stages.values():
        for dep in s.depends_on:
            if dep in stages:
                in_degree[s.name] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        # Process in insertion order for determinism
        node = queue.pop(0)
        topo_order.append(node)
        for s in stages.values():
            if node in s.depends_on:
                in_degree[s.name] -= 1
                if in_degree[s.name] == 0:
                    queue.append(s.name)

    # Forward pass: compute start/end times
    for name in topo_order:
        s = stages[name]
        if s.depends_on:
            s.start_us = max(stages[dep].end_us for dep in s.depends_on if dep in stages)
        else:
            s.start_us = 0.0
        s.end_us = s.start_us + s.duration_us

    # Find total time
    total_us = max(s.end_us for s in stages.values())

    # Backtrack critical path
    critical_path = []
    # Find the stage that ends at total_us
    current = max(stages.values(), key=lambda s: s.end_us)
    while current:
        current.on_critical_path = True
        critical_path.append(current.name)
        if not current.depends_on:
            break
        # Follow the dependency that determined start time
        pred = max(
            (stages[dep] for dep in current.depends_on if dep in stages),
            key=lambda s: s.end_us,
        )
        current = pred

    critical_path.reverse()

    # Compute busy times
    aic_busy = sum(s.duration_us for s in stages.values() if s.pipe == "AIC" and not s.name.startswith("sync_") and not s.name.startswith("aic_wait"))
    aiv_busy = sum(s.duration_us for s in stages.values() if s.pipe == "AIV" and not s.name.startswith("sync_") and not s.name.startswith("aiv_wait"))
    sync_total = sum(s.duration_us for s in stages.values() if s.name.startswith("sync_") or "wait" in s.name)

    return PipelineResult(
        stages=stages,
        critical_path=critical_path,
        total_us=total_us,
        aic_busy_us=aic_busy,
        aiv_busy_us=aiv_busy,
        sync_overhead_us=sync_total,
    )


def estimate_kernel_time(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec = ASCEND_910B,
) -> float:
    """Estimate total kernel execution time in microseconds.

    Accounts for multi-step execution when T > stepBatchSize.
    """
    T = params.T
    step = tiling.step_batch_size
    num_steps = ceil_div(T, step)

    result = build_pipeline_dag(params, tiling, hw)
    single_step_us = result.total_us

    if num_steps <= 1:
        return single_step_us

    # Multi-step: first step + (num_steps-1) * steady_state
    # Steady state overlaps some init overhead
    steady_state_us = single_step_us * 0.95  # ~5% overlap from pipelining
    return single_step_us + (num_steps - 1) * steady_state_us


def format_timeline(result: PipelineResult, width: int = 100) -> str:
    """Generate ASCII art timeline showing AIC/AIV stages."""
    total = result.total_us
    if total == 0:
        return "(no stages)"

    scale = (width - 20) / total  # chars per microsecond

    lines = []
    lines.append(f"Total: {total:.2f} us")
    lines.append(f"AIC busy: {result.aic_busy_us:.2f} us | AIV busy: {result.aiv_busy_us:.2f} us | Sync: {result.sync_overhead_us:.2f} us")
    lines.append("")

    # Time ruler
    ruler_marks = 5
    ruler = " " * 18 + "|"
    for i in range(1, ruler_marks + 1):
        t = total * i / ruler_marks
        pos = int(t * scale)
        ruler = ruler.ljust(18 + pos + 1)
        ruler = ruler[:18 + pos] + "|"
    lines.append(ruler)

    time_labels = " " * 18 + "0"
    for i in range(1, ruler_marks + 1):
        t = total * i / ruler_marks
        pos = int(t * scale)
        label = f"{t:.1f}"
        time_labels = time_labels.ljust(18 + pos + 1)
        time_labels = time_labels[:18 + pos - len(label) + 1] + label
    lines.append(time_labels + " us")
    lines.append("")

    # Group stages by pipe
    for pipe in ["AIC", "AIV"]:
        lines.append(f"  {pipe}:")
        pipe_stages = [
            s for s in result.stages.values()
            if s.pipe == pipe
            and not s.name.startswith("sync_")
            and not s.name.startswith("aic_wait")
            and not s.name.startswith("aiv_wait")
        ]
        pipe_stages.sort(key=lambda s: s.start_us)

        for s in pipe_stages:
            start_col = int(s.start_us * scale)
            end_col = int(s.end_us * scale)
            bar_len = max(end_col - start_col, 1)

            label = s.name[:14]
            cp_marker = "*" if s.on_critical_path else " "

            bar = " " * start_col + "[" + "=" * max(bar_len - 2, 0) + "]"
            line = f"  {cp_marker} {label:14s} {bar} {s.duration_us:.2f}us"
            lines.append(line)

        lines.append("")

    # Critical path
    lines.append("Critical path:")
    cp_stages = [result.stages[n] for n in result.critical_path
                 if not n.startswith("sync_") and "wait" not in n]
    cp_names = [f"{s.name}({s.duration_us:.1f})" for s in cp_stages]
    lines.append("  " + " -> ".join(cp_names))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Roofline model analysis
# ---------------------------------------------------------------------------

@dataclass
class RooflineResult:
    """Roofline analysis result for a single matmul."""
    name: str
    block: MatmulBlockSpec
    M: int
    K: int
    N: int
    active_cores: int
    # Time components (us)
    cube_us: float = 0.0
    mte2_l0_us: float = 0.0      # max(L0A, L0B) load time
    fixpipe_us: float = 0.0
    hbm_l1_us: float = 0.0       # HBM/L2 -> L1 time
    total_us: float = 0.0
    # Derived metrics
    arithmetic_intensity: float = 0.0  # FLOPs / bytes_loaded
    peak_gflops: float = 0.0
    achieved_gflops: float = 0.0
    efficiency: float = 0.0       # achieved / peak
    bound: str = ""


def roofline_analysis(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
    block_specs: Optional[Dict[str, MatmulBlockSpec]] = None,
) -> Dict[str, RooflineResult]:
    """Compute roofline analysis for each matmul.

    Requires per-cycle HW spec (hw.has_cycle_spec == True).
    """
    if not hw.has_cycle_spec:
        return {}

    if block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    T = tiling.step_batch_size
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N

    mm_configs = {
        "MM1_Cq": (T, He, Hcq, tiling.mm1_block_num, False),
        "MM2_CkvKr": (T, He, Hckv + Dr, tiling.mm2_block_num, True),
        "MM3_QcQr": (T, Hcq, N * (D + Dr), tiling.mm3_block_num, False),
        "MM4_Qn": (T, D, Hckv, tiling.mm4_block_num, False),
    }

    results = {}
    for mm_name, (M, K, N_total, cores, a_reused) in mm_configs.items():
        block = block_specs[mm_name]
        C = max(cores, 1)
        per_core_N = math.ceil(N_total / C)

        # Get detailed timing
        timing = estimate_matmul_detailed(
            mm_name, M, K, N_total, cores, hw, block, a_reused)

        # Arithmetic intensity: FLOPs / bytes loaded from HBM
        total_flops = 2.0 * M * per_core_N * K
        # Bytes loaded per core from HBM/L2 to L1
        dtype_a = block.dtype_a_size
        dtype_b = block.dtype_b_size
        dtype_c = block.dtype_c_size
        bytes_A = M * K * dtype_a
        bytes_B = K * per_core_N * dtype_b
        bytes_C = M * per_core_N * dtype_c
        total_bytes = bytes_A + bytes_B + bytes_C
        ai = total_flops / total_bytes if total_bytes > 0 else 0

        # Peak compute (per core)
        if dtype_a == 1:
            peak_ops_per_sec = hw.mmad_ops_per_cycle_fp8 * hw.freq_hz * 2.0
        else:
            peak_ops_per_sec = hw.mmad_ops_per_cycle_fp16 * hw.freq_hz * 2.0
        peak_gflops = peak_ops_per_sec / 1e9

        # Achieved
        actual_time_s = timing.total_us / 1e6
        achieved_gflops = (total_flops / actual_time_s / 1e9) if actual_time_s > 0 else 0
        efficiency = achieved_gflops / peak_gflops if peak_gflops > 0 else 0

        results[mm_name] = RooflineResult(
            name=mm_name,
            block=block,
            M=M, K=K, N=N_total,
            active_cores=C,
            cube_us=timing.cube_us,
            mte2_l0_us=max(timing.l0a_us, timing.l0b_us) * block.stepK * timing.kL1_loops
                if block.mode == "split_k" else max(timing.l0a_us, timing.l0b_us) * timing.kL1_loops,
            fixpipe_us=timing.fixpipe_us,
            hbm_l1_us=timing.mte1_outer_us * timing.kL1_loops
                if block.mode == "split_k" else timing.mte1_outer_us,
            total_us=timing.total_us,
            arithmetic_intensity=ai,
            peak_gflops=peak_gflops,
            achieved_gflops=achieved_gflops,
            efficiency=efficiency,
            bound=timing.bound,
        )

    return results


def format_roofline_report(
    results: Dict[str, RooflineResult],
    hw: AscendHWSpec,
) -> str:
    """Format roofline analysis as a table."""
    lines = []
    lines.append(f"Roofline Analysis ({hw.name}, freq={hw.freq_ghz}GHz)")
    lines.append("")

    header = (
        f"{'Matmul':<12} {'Block(M*N*K)':<16} {'stepK':>5} {'Cores':>5} | "
        f"{'Cube':>8} {'MTE2_L0':>8} {'FixPipe':>8} {'HBM>L1':>8} {'Total':>8} | "
        f"{'AI':>6} {'Peak':>8} {'Achv':>8} {'Eff':>6} {'Bound':<8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for mm_name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        if mm_name not in results:
            continue
        r = results[mm_name]
        b = r.block
        block_str = f"{b.baseM}x{b.baseN}x{b.baseK}"
        lines.append(
            f"{r.name:<12} {block_str:<16} {b.stepK:>5} {r.active_cores:>5} | "
            f"{r.cube_us:>7.2f} {r.mte2_l0_us:>7.2f} {r.fixpipe_us:>7.2f} {r.hbm_l1_us:>7.2f} {r.total_us:>7.2f} | "
            f"{r.arithmetic_intensity:>5.1f} {r.peak_gflops:>7.1f} {r.achieved_gflops:>7.1f} {r.efficiency:>5.1%} {r.bound:<8}"
        )

    lines.append("")

    # Summary: ridge point
    if hw.has_cycle_spec:
        hbm_bw_per_core = hw.per_core_bw("hbm", hw.aic_num)
        fp16_peak = hw.mmad_ops_per_cycle_fp16 * hw.freq_hz * 2.0
        fp8_peak = hw.mmad_ops_per_cycle_fp8 * hw.freq_hz * 2.0
        ridge_fp16 = fp16_peak / hbm_bw_per_core if hbm_bw_per_core > 0 else 0
        ridge_fp8 = fp8_peak / hbm_bw_per_core if hbm_bw_per_core > 0 else 0
        lines.append(f"Ridge point (HBM, {hw.aic_num} cores): "
                     f"BF16={ridge_fp16:.1f} FLOPs/B, FP8={ridge_fp8:.1f} FLOPs/B")
        lines.append(f"Per-core HBM BW: {hbm_bw_per_core/1e9:.1f} GB/s, "
                     f"L2 BW: {hw.per_core_bw('l2', hw.aic_num)/1e9:.1f} GB/s")

    return "\n".join(lines)
