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
    )
    from .tiling_sim import TilingConfig, compute_tiling, ceil_div
except ImportError:
    from perf_model import (
        AscendHWSpec, ASCEND_910B, OperatorParams, MatMulTiming, VectorOpTiming,
        estimate_all_stages, is_full_quant, is_int8_quant, is_mxfp8,
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
