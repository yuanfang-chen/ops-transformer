#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python replica of mla_prolog_tiling.cpp tiling logic."""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    from .perf_model import (
        AscendHWSpec, ASCEND_910B, OperatorParams, QuantMode,
        is_full_quant, is_int8_quant, is_mxfp8, resolve_quant_mode,
        MatmulBlockSpec, MatMulTiming, VectorOpTiming,
        check_cache_fit, derive_stepK,
        estimate_matmul_detailed, get_default_block_specs,
        estimate_mm2_split_k_cross_core,
    )
except ImportError:
    from perf_model import (
        AscendHWSpec, ASCEND_910B, OperatorParams, QuantMode,
        is_full_quant, is_int8_quant, is_mxfp8, resolve_quant_mode,
        MatmulBlockSpec, MatMulTiming, VectorOpTiming,
        check_cache_fit, derive_stepK,
        estimate_matmul_detailed, get_default_block_specs,
        estimate_mm2_split_k_cross_core,
    )

# Constants from mla_prolog_tiling.h / mla_prolog_comm.h
BLOCK_SIZE = 32
GROUP_COMPUTE_CUBE_NUM_PER_GROUP = 8
HIGH_THROUGHPUT_D_SIZE = 128
GROUP_COMPUTE_T_SIZE = 1
GROUP_COMPUTE_N_SIZE = 8
GROUP_COMPUTE_MIN_AIC_NUM = 16
GROUP_COMPUTE_MIN_AIV_NUM = 32
NUM_BYTES_BF16 = 2
NUM_BYTES_INT32 = 4


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def calc_single_core_n(n: int, core_num: int, align_num: int) -> int:
    """Replicate CalcSingleCoreN from tiling.cpp line 316-319."""
    return ceil_div(n, align_num * core_num) * align_num


@dataclass
class TilingConfig:
    """Output of the tiling computation, mirrors MlaPrologBaseParams."""
    step_batch_size: int = 32
    m_sub_size: int = 0
    m_sub_core_num: int = 0
    step_num_head_dequant: int = 16
    vector_block_num: int = 32

    mm1_block_num: int = 24
    mm1_single_core_n: int = 64
    mm2_block_num: int = 9
    mm2_single_core_n: int = 64
    mm3_block_num: int = 24
    mm3_single_core_n: int = 256
    mm4_block_num: int = 24
    mm4_single_core_batch: int = 1

    enable_group_compute_opt: bool = False
    enable_dequant_opt: bool = False
    workspace_size: int = 0

    # Derived info for analysis
    aic_num: int = 24
    aiv_num: int = 48

    def summary(self) -> str:
        lines = [
            f"stepBatchSize={self.step_batch_size}, vectorBlocks={self.vector_block_num}",
            f"MM1: cores={self.mm1_block_num}, singleN={self.mm1_single_core_n}",
            f"MM2: cores={self.mm2_block_num}, singleN={self.mm2_single_core_n}",
            f"MM3: cores={self.mm3_block_num}, singleN={self.mm3_single_core_n}",
            f"MM4: cores={self.mm4_block_num}, singleBatch={self.mm4_single_core_batch}",
            f"groupComputeOpt={self.enable_group_compute_opt}, dequantOpt={self.enable_dequant_opt}",
            f"workspace={self.workspace_size} bytes",
        ]
        return "\n".join(lines)


def compute_tiling(
    params: OperatorParams,
    hw: AscendHWSpec = ASCEND_910B,
    override_step_batch_size: Optional[int] = None,
) -> TilingConfig:
    """Replicate the C++ tiling logic from mla_prolog_tiling.cpp."""
    T = params.T
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    Nkv = params.Nkv

    qm = params.quant_mode
    aic_num = hw.aic_num
    aiv_num = hw.aiv_num

    # Weight dtype size for alignment calculation
    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        wt_dtype_size = 1  # INT8 or FP8
    else:
        wt_dtype_size = 2  # BF16

    # UqQr weight dtype: may differ from main weight dtype
    if is_full_quant(qm):
        uqqr_dtype_size = 1
    elif is_mxfp8(qm):
        uqqr_dtype_size = 1
    else:
        uqqr_dtype_size = 2

    cfg = TilingConfig(aic_num=aic_num, aiv_num=aiv_num)

    # === ProcessBaseInputs (lines 415-452) ===
    split_m_flag = False  # Default: split-N mode

    step_batch_size = min(128, T)
    if override_step_batch_size is not None:
        step_batch_size = min(override_step_batch_size, T)
    cfg.step_batch_size = step_batch_size

    if split_m_flag and step_batch_size > 0 and aic_num > 0 and T > 0:
        cfg.m_sub_size = ceil_div(T, aic_num)
        cfg.m_sub_core_num = T - (cfg.m_sub_size - 1) * aic_num

    if D == HIGH_THROUGHPUT_D_SIZE:
        cfg.step_num_head_dequant = min(64, N)
    else:
        cfg.step_num_head_dequant = min(16, N)

    cfg.vector_block_num = min(step_batch_size, aiv_num)

    cv_ratio = aiv_num // aic_num if aic_num > 0 else 1

    # Group compute optimization check
    enable_group_compute_opt = False
    enable_dequant_opt = False

    if (qm in (QuantMode.PARTIAL_QUANT_KV_NO_QUANT, QuantMode.PARTIAL_QUANT_KV_QUANT_PER_CHANNEL)
            and T == GROUP_COMPUTE_T_SIZE
            and Nkv == GROUP_COMPUTE_N_SIZE
            and aiv_num >= GROUP_COMPUTE_MIN_AIV_NUM
            and aic_num >= GROUP_COMPUTE_MIN_AIC_NUM
            and cv_ratio != 1):
        enable_group_compute_opt = True
        aiv_num = 32
        aic_num = 16
    elif (uqqr_dtype_size == 1 and N >= GROUP_COMPUTE_N_SIZE) or is_mxfp8(qm):
        enable_dequant_opt = True

    cfg.enable_group_compute_opt = enable_group_compute_opt
    cfg.enable_dequant_opt = enable_dequant_opt
    cfg.aic_num = aic_num
    cfg.aiv_num = aiv_num

    # === FillMatmul1Tiling (lines 321-335) ===
    align_num_mm1 = BLOCK_SIZE // wt_dtype_size
    single_core_hcq = calc_single_core_n(Hcq, aic_num, align_num_mm1)
    single_core_hcq = max(single_core_hcq, 64)
    cfg.mm1_single_core_n = single_core_hcq
    cfg.mm1_block_num = ceil_div(Hcq, single_core_hcq)

    # === FillMatmul2Tiling (lines 337-362) ===
    ckv_kr_total = Hckv + Dr
    if split_m_flag:
        cfg.mm2_single_core_n = ckv_kr_total
        cfg.mm2_block_num = aic_num
    elif aic_num >= 9:
        base_n = 64
        cfg.mm2_block_num = ckv_kr_total // base_n
        cfg.mm2_single_core_n = base_n
    else:
        align_num_mm2 = BLOCK_SIZE // wt_dtype_size
        cfg.mm2_single_core_n = calc_single_core_n(ckv_kr_total, aic_num, align_num_mm2)
        cfg.mm2_block_num = ceil_div(ckv_kr_total, cfg.mm2_single_core_n)

    # === FillMatmul3Tiling (lines 364-394) ===
    ori_m3 = N * (D + Dr)
    if enable_group_compute_opt:
        cfg.mm3_single_core_n = calc_single_core_n(
            N * D, GROUP_COMPUTE_CUBE_NUM_PER_GROUP, D)
    elif enable_dequant_opt:
        cfg.mm3_single_core_n = calc_single_core_n(ori_m3, aic_num, D + Dr)
    else:
        align_num_mm3 = BLOCK_SIZE // uqqr_dtype_size
        cfg.mm3_single_core_n = calc_single_core_n(ori_m3, aic_num, align_num_mm3)

    if split_m_flag:
        cfg.mm3_single_core_n = ori_m3
        cfg.mm3_block_num = aic_num
    else:
        cfg.mm3_block_num = ceil_div(ori_m3, cfg.mm3_single_core_n)

    # === FillMatmul4Tiling (lines 396-413) ===
    if split_m_flag:
        cfg.mm4_single_core_batch = N
        cfg.mm4_block_num = aic_num
    else:
        cfg.mm4_single_core_batch = ceil_div(N, aic_num)
        cfg.mm4_block_num = ceil_div(N, cfg.mm4_single_core_batch)

    # === CalcWorkSpace (lines 515-559) ===
    workspace = 0  # Skip libapiSize_ (unknown, assume 0 for modeling)
    full_quant_modes = (
        QuantMode.FULL_QUANT_KV_NO_QUANT,
        QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.FULL_QUANT_KV_QUANT_PERTILE,
        QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,
    )
    if qm in full_quant_modes:
        workspace += step_batch_size * Hcq * NUM_BYTES_INT32
        workspace += step_batch_size * Hcq * NUM_BYTES_BF16
        workspace += step_batch_size * (Hckv + Dr) * NUM_BYTES_INT32
        if qm in (QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
                   QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
                   QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
                   QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE):
            workspace += step_batch_size * N * Hckv * NUM_BYTES_BF16
    else:
        workspace += step_batch_size * Hcq * NUM_BYTES_BF16 * 2
        workspace += step_batch_size * (Hckv + Dr) * NUM_BYTES_BF16

    # QcQr buffer (always)
    headSizeQc = N * D
    headSizeQr = N * Dr
    workspace += step_batch_size * (headSizeQc + headSizeQr) * NUM_BYTES_INT32
    workspace += step_batch_size * N * D * NUM_BYTES_BF16

    if enable_group_compute_opt or enable_dequant_opt:
        workspace += step_batch_size * BLOCK_SIZE

    cfg.workspace_size = workspace
    return cfg


def search_best_tiling(
    params: OperatorParams,
    hw: AscendHWSpec = ASCEND_910B,
    step_candidates: Optional[List[int]] = None,
) -> List[Tuple[TilingConfig, float]]:
    """Explore tiling configurations and estimate kernel time for each.

    Returns list of (TilingConfig, estimated_time_us) sorted by time.
    """
    try:
        from .pipeline_model import estimate_kernel_time
    except ImportError:
        from pipeline_model import estimate_kernel_time

    if step_candidates is None:
        step_candidates = [16, 32, 64, 128]

    results = []
    for step in step_candidates:
        if step > params.T:
            continue
        tiling = compute_tiling(params, hw, override_step_batch_size=step)
        total_us = estimate_kernel_time(params, tiling, hw)
        results.append((tiling, total_us))

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Per-matmul block size search (requires per-cycle HW spec)
# ---------------------------------------------------------------------------

SEARCH_BASE_M = [16, 32, 64, 128, 192, 256, 512]
SEARCH_BASE_N = [64, 128, 192, 256, 512]
SEARCH_BASE_K = [64, 128, 256, 512]


def search_matmul_block_sizes(
    name: str,
    M: int, K: int, N_total: int,
    active_cores: int,
    hw: AscendHWSpec,
    dtype_a_size: int = 2,
    dtype_b_size: int = 2,
    dtype_c_size: int = 2,
    a_reused: bool = False,
    mode: str = "split_k",
) -> List[Tuple[MatmulBlockSpec, MatMulTiming]]:
    """Exhaustive search over (baseM, baseN, baseK) with stepK derived from L1.

    Returns list of (block_spec, timing) sorted by total_us ascending.
    """
    if not hw.has_cycle_spec:
        return []

    C = max(active_cores, 1)
    per_core_N = math.ceil(N_total / C)
    results = []

    if mode == "full_load":
        # For full_load mode, only search baseN (nSplitSize)
        for baseN in SEARCH_BASE_N:
            if baseN > per_core_N:
                baseN = per_core_N
            block = MatmulBlockSpec(
                name, baseM=M, baseN=baseN, baseK=K, stepK=1,
                mode="full_load",
                dtype_a_size=dtype_a_size, dtype_b_size=dtype_b_size,
                dtype_c_size=dtype_c_size,
            )
            if not check_cache_fit(block, hw, per_core_N, M=M, K=K):
                continue
            timing = estimate_matmul_detailed(
                name, M, K, N_total, active_cores, hw, block, a_reused)
            results.append((block, timing))
        results.sort(key=lambda x: x[1].total_us)
        return results

    # Build baseM candidates: always include actual M (aligned to 16 for cube)
    baseM_candidates = sorted(set(SEARCH_BASE_M + [M]))
    # Build baseN candidates: always include per_core_N if smaller than max
    baseN_candidates = sorted(set(SEARCH_BASE_N + [per_core_N]))

    for baseM in baseM_candidates:
        if baseM > M:
            continue
        for baseN in baseN_candidates:
            if baseN > per_core_N:
                continue
            for baseK in SEARCH_BASE_K:
                if baseK > K:
                    continue
                block = MatmulBlockSpec(
                    name, baseM=baseM, baseN=baseN, baseK=baseK, stepK=1,
                    mode="split_k",
                    dtype_a_size=dtype_a_size, dtype_b_size=dtype_b_size,
                    dtype_c_size=dtype_c_size,
                )
                # Check cache constraints (with actual M, K for buffer factor)
                if not check_cache_fit(block, hw, per_core_N, M=M, K=K):
                    continue
                # Derive stepK from L1
                stepK = derive_stepK(block, K, hw, per_core_N, M=M)
                if stepK < 1:
                    continue
                block.stepK = stepK

                timing = estimate_matmul_detailed(
                    name, M, K, N_total, active_cores, hw, block, a_reused)
                results.append((block, timing))

    results.sort(key=lambda x: x[1].total_us)
    return results


def search_all_matmul_blocks(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> Dict[str, List[Tuple[MatmulBlockSpec, MatMulTiming]]]:
    """Search optimal block sizes for all four matmuls.

    Returns dict mapping MM name -> sorted list of (block_spec, timing).
    """
    qm = params.quant_mode
    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    is_quant = is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm)
    mm3_wt_size = 1 if is_quant else 2
    T = tiling.step_batch_size

    results = {}

    # MM1: tokenX[T, He] x weightDq[He, Hcq]
    results["MM1_Cq"] = search_matmul_block_sizes(
        "MM1_Cq", T, params.He, params.Hcq,
        active_cores=tiling.mm1_block_num, hw=hw,
        dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
        a_reused=False,
    )

    # MM2: tokenX[T, He] x weightDkvKr[He, Hckv+Dr] (A reused from MM1)
    results["MM2_CkvKr"] = search_matmul_block_sizes(
        "MM2_CkvKr", T, params.He, params.Hckv + params.Dr,
        active_cores=tiling.mm2_block_num, hw=hw,
        dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
        a_reused=True,
    )

    # MM3: Cq[T, Hcq] x weightUqQr[Hcq, N*(D+Dr)]
    results["MM3_QcQr"] = search_matmul_block_sizes(
        "MM3_QcQr", T, params.Hcq, params.N * (params.D + params.Dr),
        active_cores=tiling.mm3_block_num, hw=hw,
        dtype_a_size=2, dtype_b_size=mm3_wt_size, dtype_c_size=out_size,
        a_reused=False,
    )

    # MM4: Qc[T, D] x Uk[D, Hckv] (full_load mode)
    results["MM4_Qn"] = search_matmul_block_sizes(
        "MM4_Qn", T, params.D, params.Hckv,
        active_cores=tiling.mm4_block_num, hw=hw,
        dtype_a_size=2, dtype_b_size=wt_size, dtype_c_size=2,
        a_reused=False, mode="full_load",
    )

    return results


# ---------------------------------------------------------------------------
# MM2 cross-core K-split search
# ---------------------------------------------------------------------------

@dataclass
class MM2SplitKResult:
    """Result of comparing split-N vs split-K for MM2."""
    k_cores: int
    mode: str             # "split_n" or "split_k"
    cube_timing: MatMulTiming = None
    accum_timing: VectorOpTiming = None
    cube_us: float = 0.0
    accum_us: float = 0.0
    total_us: float = 0.0


def search_mm2_split_k(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[MM2SplitKResult]:
    """Search split-K configurations for MM2 and compare with split-N baseline.

    Returns sorted list of MM2SplitKResult (best first).
    """
    if not hw.has_cycle_spec:
        return []

    T = tiling.step_batch_size
    He = params.He
    N = params.Hckv + params.Dr
    qm = params.quant_mode
    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size
    vec_cores = tiling.vector_block_num

    block_specs = get_default_block_specs(params, hw)
    mm2_block = block_specs["MM2_CkvKr"]

    results = []

    # Baseline: split-N (current approach)
    split_n_timing = estimate_matmul_detailed(
        "MM2_CkvKr", T, He, N,
        active_cores=tiling.mm2_block_num,
        hw=hw, block=mm2_block, a_reused=True,
    )
    results.append(MM2SplitKResult(
        k_cores=tiling.mm2_block_num,
        mode="split_n",
        cube_timing=split_n_timing,
        cube_us=split_n_timing.total_us,
        accum_us=0.0,
        total_us=split_n_timing.total_us,
    ))

    # Split-K candidates
    k_candidates = [9, 12, 16, 18, 24, 32]
    for k_cores in k_candidates:
        if k_cores > hw.aic_num:
            continue
        cube_t, accum_t = estimate_mm2_split_k_cross_core(
            T, He, N, k_cores, hw, mm2_block,
            a_reused=True, vec_cores=vec_cores, out_size=out_size,
        )
        results.append(MM2SplitKResult(
            k_cores=k_cores,
            mode="split_k",
            cube_timing=cube_t,
            accum_timing=accum_t,
            cube_us=cube_t.total_us,
            accum_us=accum_t.total_us,
            total_us=cube_t.total_us + accum_t.total_us,
        ))

    results.sort(key=lambda r: r.total_us)
    return results
