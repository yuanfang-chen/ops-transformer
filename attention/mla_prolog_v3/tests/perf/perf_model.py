#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardware specs and primitive timing models for Ascend NPU performance analysis."""

import math
from dataclasses import dataclass, field
from enum import IntEnum


class QuantMode(IntEnum):
    NO_QUANT = 0
    PARTIAL_QUANT_KV_NO_QUANT = 1
    PARTIAL_QUANT_KV_QUANT_PER_CHANNEL = 2
    FULL_QUANT_KV_NO_QUANT = 3
    FULL_QUANT_KV_QUANT_PER_TENSOR = 4
    PARTIAL_QUANT_KV_QUANT_PERTILE = 5
    FULL_QUANT_KV_QUANT_PERTILE = 6
    MXFP8_FULL_QUANT_KV_NO_QUANT = 7
    MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR = 8
    MXFP8_FULL_QUANT_KV_QUANT_PER_TILE = 9


def resolve_quant_mode(weight_quant_mode: int, kv_cache_quant_mode: int) -> QuantMode:
    """Derive the internal QUANT_MODE enum from user-facing weight/kv quant params."""
    if weight_quant_mode == 0:
        if kv_cache_quant_mode == 0:
            return QuantMode.NO_QUANT
        elif kv_cache_quant_mode == 3:
            return QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE
        return QuantMode.NO_QUANT
    elif weight_quant_mode in (1, 2):
        if weight_quant_mode == 1:
            if kv_cache_quant_mode == 0:
                return QuantMode.PARTIAL_QUANT_KV_NO_QUANT
            elif kv_cache_quant_mode == 1:
                return QuantMode.PARTIAL_QUANT_KV_QUANT_PER_CHANNEL
            elif kv_cache_quant_mode == 2:
                return QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE
            return QuantMode.PARTIAL_QUANT_KV_NO_QUANT
        else:
            if kv_cache_quant_mode == 0:
                return QuantMode.FULL_QUANT_KV_NO_QUANT
            elif kv_cache_quant_mode == 1:
                return QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR
            elif kv_cache_quant_mode == 2:
                return QuantMode.FULL_QUANT_KV_QUANT_PERTILE
            return QuantMode.FULL_QUANT_KV_NO_QUANT
    elif weight_quant_mode == 3:
        if kv_cache_quant_mode == 0:
            return QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT
        elif kv_cache_quant_mode == 1:
            return QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR
        elif kv_cache_quant_mode == 3:
            return QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE
        return QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT
    return QuantMode.NO_QUANT


def is_full_quant(qm: QuantMode) -> bool:
    return qm in (
        QuantMode.FULL_QUANT_KV_NO_QUANT,
        QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.FULL_QUANT_KV_QUANT_PERTILE,
        QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,
    )


def is_mxfp8(qm: QuantMode) -> bool:
    return qm in (
        QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,
    )


def is_int8_quant(qm: QuantMode) -> bool:
    return qm in (
        QuantMode.PARTIAL_QUANT_KV_NO_QUANT,
        QuantMode.PARTIAL_QUANT_KV_QUANT_PER_CHANNEL,
        QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE,
        QuantMode.FULL_QUANT_KV_NO_QUANT,
        QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.FULL_QUANT_KV_QUANT_PERTILE,
    )


@dataclass
class AscendHWSpec:
    name: str = "Ascend910B"
    aic_num: int = 24
    aiv_num: int = 48
    cube_tflops_bf16: float = 320.0
    cube_tops_int8: float = 640.0
    cube_tflops_fp8: float = 640.0
    hbm_bw_bytes_per_sec: float = 1.2e12
    l2_bw_bytes_per_sec: float = 6.4e12
    fixpipe_bw_bytes_per_sec: float = 1.2e12
    vector_ops_per_sec_bf16: float = 10.0e12
    mte2_bw_bytes_per_sec: float = 1.2e12
    mte3_bw_bytes_per_sec: float = 1.2e12
    sync_overhead_us: float = 1.0
    scatter_efficiency: float = 0.4

    @property
    def cube_flops_bf16(self) -> float:
        return self.cube_tflops_bf16 * 1e12

    @property
    def cube_ops_int8(self) -> float:
        return self.cube_tops_int8 * 1e12

    @property
    def cube_flops_fp8(self) -> float:
        return self.cube_tflops_fp8 * 1e12


ASCEND_910B = AscendHWSpec()


def dtype_size(quant_mode: QuantMode) -> int:
    """Return weight element size in bytes based on quant mode."""
    if is_full_quant(quant_mode) or is_int8_quant(quant_mode) or is_mxfp8(quant_mode):
        return 1
    return 2  # BF16


@dataclass
class MatMulTiming:
    name: str
    M: int
    K: int
    N: int
    active_cores: int
    load_A_hbm_us: float = 0.0
    load_A_l2_us: float = 0.0
    load_B_hbm_us: float = 0.0
    cube_us: float = 0.0
    fixpipe_us: float = 0.0
    total_us: float = 0.0
    bound: str = ""
    a_source: str = "HBM"

    @property
    def mte1_us(self) -> float:
        return max(self.load_A_hbm_us if self.a_source == "HBM" else self.load_A_l2_us,
                   self.load_B_hbm_us)


@dataclass
class VectorOpTiming:
    name: str
    mte2_load_us: float = 0.0
    vec_compute_us: float = 0.0
    mte3_store_us: float = 0.0
    total_us: float = 0.0
    bound: str = ""


def estimate_matmul(
    name: str,
    M: int, K: int, N: int,
    active_cores: int,
    hw: AscendHWSpec,
    dtype_a_size: int = 2,
    dtype_b_size: int = 2,
    dtype_c_size: int = 2,
    a_reused: bool = False,
) -> MatMulTiming:
    """Estimate matmul timing with HBM/L2 bandwidth separation.

    In split-N mode, all cores load the same A matrix but different B slices.
    A is shared (L2 reusable), B is unique per core (always HBM).
    """
    C = max(active_cores, 1)
    per_core_N = math.ceil(N / C)

    # A matrix load
    load_A_bytes = M * K * dtype_a_size
    load_A_hbm_time_s = load_A_bytes / hw.hbm_bw_bytes_per_sec
    # All C cores compete for bandwidth; the bottleneck bus limits all cores
    load_A_l2_time_s = load_A_bytes / (hw.l2_bw_bytes_per_sec / max(C, 1))

    if a_reused:
        # A already in L2 from prior matmul
        a_source = "L2"
        load_A_effective_s = load_A_l2_time_s
    else:
        a_source = "HBM"
        # First core pays HBM cost; but all cores contend on whichever bus saturates first
        load_A_effective_s = load_A_bytes / min(hw.hbm_bw_bytes_per_sec, hw.l2_bw_bytes_per_sec / C)

    # B matrix load (unique per core, all from HBM)
    load_B_bytes_per_core = K * per_core_N * dtype_b_size
    load_B_time_s = load_B_bytes_per_core / (hw.hbm_bw_bytes_per_sec / C)

    # MTE1 time (double-buffered overlap of A and B)
    mte1_time_s = max(load_A_effective_s, load_B_time_s)

    # Compute
    total_flops = 2.0 * M * per_core_N * K
    if dtype_a_size == 1:
        cube_power = hw.cube_flops_fp8 if is_mxfp8(QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT) else hw.cube_ops_int8
    else:
        cube_power = hw.cube_flops_bf16
    cube_time_s = total_flops / (cube_power / hw.aic_num)

    # Store via FixPipe
    store_C_bytes = M * per_core_N * dtype_c_size
    fixpipe_time_s = store_C_bytes / (hw.fixpipe_bw_bytes_per_sec / C)

    # Total with double-buffering (overlap MTE1, Cube, FixPipe)
    total_s = max(mte1_time_s, cube_time_s, fixpipe_time_s)

    # Determine bound
    components = {
        "HBM" if a_source == "HBM" else "L2": load_A_effective_s,
        "HBM_B": load_B_time_s,
        "CUBE": cube_time_s,
        "FIXPIPE": fixpipe_time_s,
    }
    # MTE1 bound is the max of A and B loads
    if mte1_time_s >= cube_time_s and mte1_time_s >= fixpipe_time_s:
        if load_A_effective_s >= load_B_time_s:
            bound = "HBM" if a_source == "HBM" else "L2"
        else:
            bound = "HBM"
    elif cube_time_s >= fixpipe_time_s:
        bound = "CUBE"
    else:
        bound = "FIXPIPE"

    us = 1e6
    return MatMulTiming(
        name=name,
        M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=load_A_hbm_time_s * us,
        load_A_l2_us=load_A_l2_time_s * us,
        load_B_hbm_us=load_B_time_s * us,
        cube_us=cube_time_s * us,
        fixpipe_us=fixpipe_time_s * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
    )


def estimate_vector_op(
    name: str,
    load_bytes: int,
    compute_elements: int,
    store_bytes: int,
    hw: AscendHWSpec,
    active_cores: int = 48,
    is_scatter: bool = False,
) -> VectorOpTiming:
    """Estimate vector operation timing (RMSNorm, RoPE, Scatter, Dequant, DynamicQuant)."""
    C = max(active_cores, 1)
    per_core_load = load_bytes / C
    per_core_store = store_bytes / C
    per_core_compute = compute_elements / C

    mte2_time_s = per_core_load / (hw.mte2_bw_bytes_per_sec / C)
    vec_time_s = per_core_compute / (hw.vector_ops_per_sec_bf16 / hw.aiv_num)

    effective_mte3_bw = hw.mte3_bw_bytes_per_sec
    if is_scatter:
        effective_mte3_bw *= hw.scatter_efficiency
    mte3_time_s = per_core_store / (effective_mte3_bw / C)

    total_s = max(mte2_time_s, vec_time_s, mte3_time_s)

    if total_s == mte2_time_s:
        bound = "MTE2"
    elif total_s == vec_time_s:
        bound = "VECTOR"
    else:
        bound = "MTE3"

    us = 1e6
    return VectorOpTiming(
        name=name,
        mte2_load_us=mte2_time_s * us,
        vec_compute_us=vec_time_s * us,
        mte3_store_us=mte3_time_s * us,
        total_us=total_s * us,
        bound=bound,
    )


@dataclass
class OperatorParams:
    """Parameters describing an MLA Prolog V3 invocation."""
    batch_size: int = 2
    seq_len: int = 4
    head_num: int = 8
    He: int = 7168
    Hcq: int = 1536
    Hckv: int = 512
    D: int = 128
    Dr: int = 64
    Nkv: int = 1
    block_size: int = 128
    weight_quant_mode: int = 0
    kv_cache_quant_mode: int = 0
    query_quant_mode: int = 0

    @property
    def T(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def N(self) -> int:
        return self.head_num

    @property
    def quant_mode(self) -> QuantMode:
        return resolve_quant_mode(self.weight_quant_mode, self.kv_cache_quant_mode)

    @property
    def weight_dtype_size(self) -> int:
        """Weight element size: 1 byte for INT8/FP8, 2 for BF16."""
        return dtype_size(self.quant_mode)

    @property
    def activation_dtype_size(self) -> int:
        """Activation (tokenX) element size."""
        if is_mxfp8(self.quant_mode):
            return 1
        return 2  # BF16

    @property
    def mm_output_dtype_size(self) -> int:
        """MatMul output size: 4 for INT32 (quant), 2 for BF16."""
        if is_full_quant(self.quant_mode) or is_mxfp8(self.quant_mode):
            return 4
        if is_int8_quant(self.quant_mode):
            return 4
        return 2


def estimate_all_stages(params: OperatorParams, tiling, hw: AscendHWSpec):
    """Estimate timing for all matmul and vector stages.

    Args:
        params: Operator parameters
        tiling: TilingConfig from tiling_sim
        hw: Hardware spec

    Returns:
        dict of stage_name -> MatMulTiming or VectorOpTiming
    """
    T = tiling.step_batch_size
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    qm = params.quant_mode

    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    stages = {}

    # MM1: tokenX[T, He] x weightDq[He, Hcq] -> Cq[T, Hcq]
    stages["MM1_Cq"] = estimate_matmul(
        "MM1_Cq", T, He, Hcq,
        active_cores=tiling.mm1_block_num,
        hw=hw,
        dtype_a_size=act_size,
        dtype_b_size=wt_size,
        dtype_c_size=out_size,
        a_reused=False,
    )

    # MM2: tokenX[T, He] x weightDkvKr[He, Hckv+Dr] -> CkvKr[T, Hckv+Dr]
    # tokenX reused from MM1 (L2 warm)
    stages["MM2_CkvKr"] = estimate_matmul(
        "MM2_CkvKr", T, He, Hckv + Dr,
        active_cores=tiling.mm2_block_num,
        hw=hw,
        dtype_a_size=act_size,
        dtype_b_size=wt_size,
        dtype_c_size=out_size,
        a_reused=True,
    )

    # MM3: Cq[T, Hcq] x weightUqQr[Hcq, N*(D+Dr)] -> QcQr[T, N*(D+Dr)]
    mm3_out_size = out_size
    if is_full_quant(qm) or is_mxfp8(qm):
        mm3_wt_size = 1
    elif is_int8_quant(qm):
        mm3_wt_size = 1
    else:
        mm3_wt_size = 2
    # Cq as A: comes from workspace, not reused from prior matmul HBM load
    stages["MM3_QcQr"] = estimate_matmul(
        "MM3_QcQr", T, Hcq, N * (D + Dr),
        active_cores=tiling.mm3_block_num,
        hw=hw,
        dtype_a_size=2,  # Cq is always BF16 after RMSNorm
        dtype_b_size=mm3_wt_size,
        dtype_c_size=mm3_out_size,
        a_reused=False,
    )

    # MM4: Qc[T, D] x Uk[D, Hckv] -> Qn, per head group
    # Runs N/mm4_cores iterations
    stages["MM4_Qn"] = estimate_matmul(
        "MM4_Qn", T, D, Hckv,
        active_cores=tiling.mm4_block_num,
        hw=hw,
        dtype_a_size=2,  # Qc is BF16
        dtype_b_size=wt_size,
        dtype_c_size=2,  # output BF16
        a_reused=False,
    )

    vec_cores = tiling.vector_block_num

    # RMSNorm Cq: load Cq + scale, compute norm, store normalized
    rmsnorm_cq_load = T * Hcq * out_size
    rmsnorm_cq_compute = T * Hcq * 3  # mul, add, rsqrt ~3 ops/element
    rmsnorm_cq_store = T * Hcq * 2  # output BF16
    stages["RmsNormCq"] = estimate_vector_op(
        "RmsNormCq", rmsnorm_cq_load, rmsnorm_cq_compute, rmsnorm_cq_store,
        hw, active_cores=vec_cores,
    )

    # RMSNorm CkvKr: similar to Cq but on Hckv+Dr
    ckv_kr_size = Hckv + Dr
    rmsnorm_ckvkr_load = T * ckv_kr_size * out_size
    rmsnorm_ckvkr_compute = T * Hckv * 3
    rmsnorm_ckvkr_store = T * Hckv * 2
    stages["RmsNormCkvKr"] = estimate_vector_op(
        "RmsNormCkvKr", rmsnorm_ckvkr_load, rmsnorm_ckvkr_compute, rmsnorm_ckvkr_store,
        hw, active_cores=vec_cores,
    )

    # RoPE Kr: apply rotary embedding to Kr portion
    rope_kr_load = T * Dr * 2 + T * Dr * 2  # Kr data + sin/cos
    rope_kr_compute = T * Dr * 4  # mul, sub, mul, add
    rope_kr_store = T * Dr * 2
    stages["RopeKr"] = estimate_vector_op(
        "RopeKr", rope_kr_load, rope_kr_compute, rope_kr_store,
        hw, active_cores=vec_cores,
    )

    # Scatter Ckv + Kr to KV cache (indexed write)
    scatter_store = T * (Hckv + Dr) * 2
    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", T * (Hckv + Dr) * 2, T * (Hckv + Dr), scatter_store,
        hw, active_cores=vec_cores, is_scatter=True,
    )

    # DequantQc or CastQc (only for quant modes)
    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        dequant_load = T * N * D * out_size
        dequant_compute = T * N * D * 2  # mul + add per element
        dequant_store = T * N * D * 2
        stages["DequantQc"] = estimate_vector_op(
            "DequantQc", dequant_load, dequant_compute, dequant_store,
            hw, active_cores=vec_cores,
        )

    # RoPE Qr
    rope_qr_load = T * N * Dr * 2 + T * Dr * 2  # Qr + sin/cos
    rope_qr_compute = T * N * Dr * 4
    rope_qr_store = T * N * Dr * 2
    stages["RopeQr"] = estimate_vector_op(
        "RopeQr", rope_qr_load, rope_qr_compute, rope_qr_store,
        hw, active_cores=vec_cores,
    )

    # DynamicQuant Qn (if query_quant_mode=1)
    if params.query_quant_mode == 1:
        dyn_quant_load = T * N * Hckv * 2
        dyn_quant_compute = T * N * Hckv * 3
        dyn_quant_store = T * N * Hckv * 1  # INT8 output
        stages["DynamicQuantQn"] = estimate_vector_op(
            "DynamicQuantQn", dyn_quant_load, dyn_quant_compute, dyn_quant_store,
            hw, active_cores=vec_cores,
        )

    # MulQr (if query_quant_mode=1)
    if params.query_quant_mode == 1:
        mul_qr_load = T * N * Dr * 2
        mul_qr_compute = T * N * Dr * 1
        mul_qr_store = T * N * Dr * 2
        stages["MulQr"] = estimate_vector_op(
            "MulQr", mul_qr_load, mul_qr_compute, mul_qr_store,
            hw, active_cores=vec_cores,
        )

    return stages
