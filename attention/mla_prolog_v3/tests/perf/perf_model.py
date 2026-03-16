#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardware specs and primitive timing models for Ascend NPU performance analysis."""

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple


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


# ---------------------------------------------------------------------------
# Cache and hardware specification
# ---------------------------------------------------------------------------

@dataclass
class CacheSpec:
    """Per-core on-chip cache sizes in bytes."""
    ub_size: int = 192 * 1024
    l1_size: int = 512 * 1024
    l0a_size: int = 64 * 1024
    l0b_size: int = 64 * 1024
    l0c_size: int = 128 * 1024


@dataclass
class AscendHWSpec:
    name: str = "Ascend910B"
    aic_num: int = 24
    aiv_num: int = 48

    # Chip-total performance (legacy 910B fields, kept for backward compat)
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

    # Per-cycle specs (0.0 freq means "use chip-total mode")
    freq_ghz: float = 0.0
    cache: CacheSpec = field(default_factory=CacheSpec)

    # Per-core bandwidth in bytes/cycle
    hbm_bw_per_cycle: float = 0.0
    l2_bw_per_cycle: float = 0.0
    l1_to_l0a_per_cycle: float = 0.0
    l1_to_l0b_per_cycle: float = 0.0
    l0c_to_out_per_cycle: float = 0.0
    ub_to_l1_per_cycle: float = 0.0
    ub_to_reg_per_cycle: float = 0.0

    # MMAD throughput: multiply-accumulate ops per cycle per core
    mmad_ops_per_cycle_fp8: int = 0
    mmad_ops_per_cycle_fp16: int = 0

    # Chip total bandwidth caps (bytes/sec)
    chip_hbm_bw_bytes_per_sec: float = 0.0
    chip_l2_bw_bytes_per_sec: float = 0.0

    @property
    def has_cycle_spec(self) -> bool:
        return self.freq_ghz > 0.0

    @property
    def freq_hz(self) -> float:
        return self.freq_ghz * 1e9

    @property
    def cube_flops_bf16(self) -> float:
        if self.has_cycle_spec:
            return self.mmad_ops_per_cycle_fp16 * 2.0 * self.freq_hz * self.aic_num
        return self.cube_tflops_bf16 * 1e12

    @property
    def cube_ops_int8(self) -> float:
        if self.has_cycle_spec:
            return self.mmad_ops_per_cycle_fp16 * 2.0 * self.freq_hz * self.aic_num
        return self.cube_tops_int8 * 1e12

    @property
    def cube_flops_fp8(self) -> float:
        if self.has_cycle_spec:
            return self.mmad_ops_per_cycle_fp8 * 2.0 * self.freq_hz * self.aic_num
        return self.cube_tflops_fp8 * 1e12

    def per_core_bw(self, bus: str, active_cores: int) -> float:
        """Return effective per-core bandwidth in bytes/sec with contention.

        For HBM/L2: min(per_cycle * freq, chip_total / active_cores).
        For L1/L0 local buses: per_cycle * freq (no cross-core contention).
        """
        C = max(active_cores, 1)
        if bus == "hbm":
            raw = self.hbm_bw_per_cycle * self.freq_hz
            cap = self.chip_hbm_bw_bytes_per_sec / C
            return min(raw, cap)
        elif bus == "l2":
            raw = self.l2_bw_per_cycle * self.freq_hz
            cap = self.chip_l2_bw_bytes_per_sec / C
            return min(raw, cap)
        elif bus == "l1_to_l0a":
            return self.l1_to_l0a_per_cycle * self.freq_hz
        elif bus == "l1_to_l0b":
            return self.l1_to_l0b_per_cycle * self.freq_hz
        elif bus == "l0c_to_out":
            return self.l0c_to_out_per_cycle * self.freq_hz
        elif bus == "ub_to_l1":
            return self.ub_to_l1_per_cycle * self.freq_hz
        elif bus == "ub_to_reg":
            return self.ub_to_reg_per_cycle * self.freq_hz
        return 0.0

    def effective_load_bw(self, active_cores: int, reuse_count: int = 1) -> float:
        """Effective per-core bandwidth mixing HBM (cold) and L2 (warm reuse).

        First access uses HBM, subsequent (reuse_count - 1) use L2.
        Returns weighted average in bytes/sec.
        """
        if reuse_count <= 1:
            return self.per_core_bw("hbm", active_cores)
        hbm = self.per_core_bw("hbm", active_cores)
        l2 = self.per_core_bw("l2", active_cores)
        return 1.0 / (1.0 / reuse_count / hbm + (reuse_count - 1.0) / reuse_count / l2)

    def mmad_throughput(self, dtype_size: int) -> float:
        """MMAD throughput in ops/sec for one core (each op = 1 MAC = 2 FLOPs)."""
        if dtype_size == 1:
            return self.mmad_ops_per_cycle_fp8 * self.freq_hz
        return self.mmad_ops_per_cycle_fp16 * self.freq_hz


ASCEND_910B = AscendHWSpec()

ASCEND_950 = AscendHWSpec(
    name="Ascend950",
    aic_num=32,
    aiv_num=64,
    # Chip-total fields (derived from per-cycle specs for backward compat)
    cube_tflops_bf16=4096 * 2 * 1.65e9 * 32 / 1e12,   # ~432.5 TFLOPS
    cube_tops_int8=4096 * 2 * 1.65e9 * 32 / 1e12,
    cube_tflops_fp8=8192 * 2 * 1.65e9 * 32 / 1e12,     # ~865.1 TFLOPS
    hbm_bw_bytes_per_sec=1.6e12,
    l2_bw_bytes_per_sec=5.2e12,
    fixpipe_bw_bytes_per_sec=256.0 * 1.65e9 * 32,       # l0c_to_out per core * cores
    vector_ops_per_sec_bf16=20.0e12,
    mte2_bw_bytes_per_sec=1.6e12,
    mte3_bw_bytes_per_sec=1.6e12,
    sync_overhead_us=0.8,
    scatter_efficiency=0.4,
    # Per-cycle specs
    freq_ghz=1.65,
    cache=CacheSpec(
        ub_size=256 * 1024,
        l1_size=512 * 1024,
        l0a_size=64 * 1024,
        l0b_size=256 * 1024,
        l0c_size=256 * 1024,
    ),
    hbm_bw_per_cycle=33.3,
    l2_bw_per_cycle=108.0,
    l1_to_l0a_per_cycle=256.0,
    l1_to_l0b_per_cycle=256.0,
    l0c_to_out_per_cycle=256.0,
    ub_to_l1_per_cycle=245.0,
    ub_to_reg_per_cycle=512.0,
    mmad_ops_per_cycle_fp8=8192,
    mmad_ops_per_cycle_fp16=4096,
    chip_hbm_bw_bytes_per_sec=1.6e12,
    chip_l2_bw_bytes_per_sec=5.2e12,
)

CHIP_REGISTRY = {"910b": ASCEND_910B, "950": ASCEND_950}


def dtype_size(quant_mode: QuantMode) -> int:
    """Return weight element size in bytes based on quant mode."""
    if is_full_quant(quant_mode) or is_int8_quant(quant_mode) or is_mxfp8(quant_mode):
        return 1
    return 2  # BF16


# ---------------------------------------------------------------------------
# MatMul block specification and cache validation
# ---------------------------------------------------------------------------

@dataclass
class MatmulBlockSpec:
    """Block sizes for a single matmul's inner loop."""
    name: str
    baseM: int
    baseN: int
    baseK: int
    stepK: int = 4
    mode: str = "split_k"   # "split_k" or "full_load"
    dtype_a_size: int = 2
    dtype_b_size: int = 2
    dtype_c_size: int = 2


def check_cache_fit(block: MatmulBlockSpec, hw: AscendHWSpec,
                    n_per_core: Optional[int] = None,
                    M: Optional[int] = None,
                    K: Optional[int] = None) -> bool:
    """Verify that block sizes fit in on-chip buffers.

    Default: double buffering (2x) for all buffers.
    Full-load optimization: if the entire matrix fits in the buffer AND
    using single buffer (1x) allows fitting, use single buffer.
    """
    n = n_per_core if n_per_core is not None else block.baseN
    actual_M = M if M is not None else block.baseM
    actual_K = K if K is not None else block.baseK

    tile_a = block.baseM * block.baseK * block.dtype_a_size
    tile_b = block.baseK * block.baseN * block.dtype_b_size
    tile_c = block.baseM * block.baseN * block.dtype_c_size

    # L0A: default double buffer; try full-load single buffer as fallback
    full_a = actual_M * actual_K * block.dtype_a_size
    if tile_a * 2 > hw.cache.l0a_size:
        # Double buffer doesn't fit — try full-load single buffer
        if full_a <= hw.cache.l0a_size and tile_a <= hw.cache.l0a_size:
            pass  # full-load OK
        else:
            return False

    # L0B: default double buffer; try full-load single buffer as fallback
    full_b = actual_K * n * block.dtype_b_size
    if tile_b * 2 > hw.cache.l0b_size:
        if full_b <= hw.cache.l0b_size and tile_b <= hw.cache.l0b_size:
            pass
        else:
            return False

    # L0C: default double buffer; try full-load single buffer as fallback
    full_c = actual_M * n * block.dtype_c_size
    if tile_c * 2 > hw.cache.l0c_size:
        if full_c <= hw.cache.l0c_size and tile_c <= hw.cache.l0c_size:
            pass
        else:
            return False

    # L1: default double buffer for per-step tile
    l1_per_step_a = block.baseM * block.baseK * block.dtype_a_size
    l1_per_step_b = block.baseK * n * block.dtype_b_size
    l1_per_step = l1_per_step_a + l1_per_step_b
    if l1_per_step * 2 > hw.cache.l1_size:
        # Double buffer doesn't fit — try full-load single buffer
        full_ab = full_a + full_b
        if full_ab <= hw.cache.l1_size and l1_per_step <= hw.cache.l1_size:
            pass
        else:
            return False
    return True


def derive_stepK(block: MatmulBlockSpec, K: int, hw: AscendHWSpec,
                 n_per_core: Optional[int] = None,
                 M: Optional[int] = None) -> int:
    """Derive optimal stepK from L1 size constraint.

    Default: double buffer (L1/2). If all K fits in a single L1 load
    with full-load single buffer AND it reduces kL1 loops, use it.
    """
    n = n_per_core if n_per_core is not None else block.baseN
    actual_M = M if M is not None else block.baseM

    l1_per_step_a = block.baseM * block.baseK * block.dtype_a_size
    l1_per_step_b = block.baseK * n * block.dtype_b_size
    per_step = l1_per_step_a + l1_per_step_b
    if per_step == 0:
        return 1

    k_iters = math.ceil(K / block.baseK) if block.baseK > 0 else 1

    # Default: double buffer (L1/2)
    double_buf_stepK = hw.cache.l1_size // 2 // per_step

    # Optimization: try full-load single buffer if it covers all K iterations
    # and produces a better result (fewer kL1 loops)
    single_buf_stepK = hw.cache.l1_size // per_step
    if single_buf_stepK >= k_iters:
        # Full load fits with single buffer → kL1Loops=1
        # Only use if double buffer wouldn't achieve the same (or doesn't fit)
        if double_buf_stepK < k_iters:
            return k_iters  # full-load optimization wins

    if double_buf_stepK < 1:
        # Double buffer doesn't fit — last resort: try single buffer
        if single_buf_stepK >= 1:
            return min(single_buf_stepK, k_iters)
        return 0  # invalid
    return min(double_buf_stepK, k_iters)


# ---------------------------------------------------------------------------
# Timing dataclasses
# ---------------------------------------------------------------------------

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
    # Inner-loop detail fields (populated by estimate_matmul_detailed)
    l0a_us: float = 0.0
    l0b_us: float = 0.0
    mmad_us: float = 0.0
    mte1_outer_us: float = 0.0
    kL1_loops: int = 0
    inner_bound: str = ""

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


# ---------------------------------------------------------------------------
# Chip-total matmul estimation (legacy 910B path)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-cycle inner-loop matmul estimation (950 detailed path)
# ---------------------------------------------------------------------------

def estimate_matmul_detailed(
    name: str,
    M: int, K: int, N: int,
    active_cores: int,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool = False,
) -> MatMulTiming:
    """Estimate matmul timing using per-cycle inner-loop model.

    Models the L1/L0 cache hierarchy:
    - Outer loop: HBM/L2 -> L1 (MTE1, double-buffered across kL1 iterations)
    - Inner loop: L1 -> L0A/L0B + MMAD (double-buffered across stepK iterations)
    - Output: L0C -> FixPipe
    """
    C = max(active_cores, 1)
    per_core_N = math.ceil(N / C)
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size

    us = 1e6

    if block.mode == "full_load":
        return _estimate_full_load(name, M, K, per_core_N, C, hw, block, a_reused)

    # --- split_k mode ---
    baseM = block.baseM
    baseK = block.baseK
    stepK = block.stepK
    kL1StepSize = baseK * stepK
    kL1Loops = math.ceil(K / kL1StepSize)

    # Bandwidth values (bytes/sec)
    hbm_bw = hw.per_core_bw("hbm", C)
    l2_bw = hw.per_core_bw("l2", C)
    l0a_bw = hw.per_core_bw("l1_to_l0a", C)
    l0b_bw = hw.per_core_bw("l1_to_l0b", C)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", C)
    mmad_tput = hw.mmad_throughput(dtype_a)  # ops/sec per core

    # --- Outer loop: GM/L2 -> L1 per kL1 iteration ---
    load_A_l1_bytes = baseM * kL1StepSize * dtype_a
    load_B_l1_bytes = kL1StepSize * per_core_N * dtype_b
    if a_reused:
        a_source = "L2"
        load_A_time_s = load_A_l1_bytes / l2_bw if l2_bw > 0 else 0
    else:
        a_source = "HBM"
        load_A_time_s = load_A_l1_bytes / hbm_bw if hbm_bw > 0 else 0
    load_B_time_s = load_B_l1_bytes / hbm_bw if hbm_bw > 0 else 0
    mte1_time_s = max(load_A_time_s, load_B_time_s)

    # --- Inner loop: L1 -> L0 + MMAD per stepK iteration ---
    l0a_bytes = baseM * baseK * dtype_a
    l0b_bytes = baseK * per_core_N * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    # MMAD: baseM * per_core_N * baseK MACs = baseM * per_core_N * baseK * 2 FLOPs
    mmad_ops = baseM * per_core_N * baseK
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    # Double-buffered: L0A/L0B loads overlap with MMAD
    inner_iter_s = max(l0a_time_s, l0b_time_s, mmad_time_s)
    total_inner_s = stepK * inner_iter_s

    # --- FixPipe: L0C -> output per kL1 iteration ---
    fixpipe_bytes = baseM * per_core_N * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    # Pipelined: MTE1, inner compute, and FixPipe overlap across kL1 iterations
    per_kl1_s = max(mte1_time_s, total_inner_s, fixpipe_time_s)
    total_s = kL1Loops * per_kl1_s

    # Determine inner bound
    candidates = {
        "MTE1_A": load_A_time_s,
        "MTE1_B": load_B_time_s,
        "L0A": l0a_time_s * stepK,
        "L0B": l0b_time_s * stepK,
        "MMAD": mmad_time_s * stepK,
        "FIXPIPE": fixpipe_time_s,
    }
    inner_bound = max(candidates, key=candidates.get)
    # Simplified bound name
    if per_kl1_s == mte1_time_s:
        bound = "HBM" if a_source == "HBM" and load_A_time_s >= load_B_time_s else "HBM"
        if a_source == "L2" and load_A_time_s >= load_B_time_s:
            bound = "L2"
    elif per_kl1_s == total_inner_s:
        if inner_iter_s == mmad_time_s:
            bound = "CUBE"
        elif inner_iter_s == l0a_time_s:
            bound = "L0A"
        else:
            bound = "L0B"
    else:
        bound = "FIXPIPE"

    return MatMulTiming(
        name=name, M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=(load_A_l1_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(load_A_l1_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=load_B_time_s * us,
        cube_us=mmad_time_s * stepK * kL1Loops * us,
        fixpipe_us=fixpipe_time_s * kL1Loops * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=mte1_time_s * us,
        kL1_loops=kL1Loops,
        inner_bound=inner_bound,
    )


def _estimate_full_load(
    name: str, M: int, K: int, per_core_N: int,
    active_cores: int, hw: AscendHWSpec, block: MatmulBlockSpec,
    a_reused: bool,
) -> MatMulTiming:
    """Inner-loop model for full_load mode (MM4): A fully loaded, N-split only."""
    us = 1e6
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size
    nSplitSize = block.baseN

    hbm_bw = hw.per_core_bw("hbm", active_cores)
    l2_bw = hw.per_core_bw("l2", active_cores)
    l0a_bw = hw.per_core_bw("l1_to_l0a", active_cores)
    l0b_bw = hw.per_core_bw("l1_to_l0b", active_cores)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", active_cores)
    mmad_tput = hw.mmad_throughput(dtype_a)

    a_source = "L2" if a_reused else "HBM"
    a_bw = l2_bw if a_reused else hbm_bw

    # Load A once
    load_A_bytes = M * K * dtype_a
    load_A_time_s = load_A_bytes / a_bw if a_bw > 0 else 0

    nSplits = math.ceil(per_core_N / nSplitSize) if nSplitSize > 0 else 1

    # Per N-split iteration
    load_B_bytes = K * nSplitSize * dtype_b
    load_B_time_s = load_B_bytes / hbm_bw if hbm_bw > 0 else 0
    l0a_bytes = M * K * dtype_a
    l0b_bytes = K * nSplitSize * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    mmad_ops = M * nSplitSize * K
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    fixpipe_bytes = M * nSplitSize * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    per_split_s = max(load_B_time_s, max(l0a_time_s, l0b_time_s, mmad_time_s), fixpipe_time_s)
    total_s = load_A_time_s + nSplits * per_split_s

    # Determine bound
    if per_split_s == load_B_time_s:
        bound = "HBM"
    elif max(l0a_time_s, l0b_time_s, mmad_time_s) >= max(load_B_time_s, fixpipe_time_s):
        if mmad_time_s >= l0a_time_s and mmad_time_s >= l0b_time_s:
            bound = "CUBE"
        elif l0a_time_s >= l0b_time_s:
            bound = "L0A"
        else:
            bound = "L0B"
    else:
        bound = "FIXPIPE"

    return MatMulTiming(
        name=name, M=M, K=K, N=per_core_N * max(active_cores, 1),
        active_cores=active_cores,
        load_A_hbm_us=(load_A_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(load_A_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=load_B_time_s * us,
        cube_us=mmad_time_s * nSplits * us,
        fixpipe_us=fixpipe_time_s * nSplits * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=load_A_time_s * us,
        kL1_loops=nSplits,
        inner_bound=bound,
    )


# ---------------------------------------------------------------------------
# Cross-core K-split model (for MM2 with small N)
# ---------------------------------------------------------------------------

def estimate_mm2_split_k_cross_core(
    M: int, K: int, N: int, k_cores: int,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool,
    vec_cores: int,
    out_size: int = 2,
) -> Tuple[MatMulTiming, VectorOpTiming]:
    """Model MM2 with K-axis split across cores + vector accumulation.

    Each cube core computes a partial sum on K/k_cores of K for the full N.
    FixPipe outputs float32 (4 bytes) for partial sums.
    Vector side accumulates k_cores partial sums in order, then stores as out_size.
    """
    K_per_core = math.ceil(K / k_cores)

    # Cube: override block with float output and reduced K
    block_kc = MatmulBlockSpec(
        block.name, block.baseM, block.baseN, block.baseK, block.stepK,
        mode="split_k",
        dtype_a_size=block.dtype_a_size,
        dtype_b_size=block.dtype_b_size,
        dtype_c_size=4,  # float32 partial sum output
    )
    # Re-derive stepK for the reduced K
    derived = derive_stepK(block_kc, K_per_core, hw, N, M=M)
    if derived > 0:
        block_kc.stepK = derived

    cube_timing = estimate_matmul_detailed(
        "MM2_CkvKr", M, K_per_core, N, k_cores, hw, block_kc, a_reused)

    # Vector accumulation: load all partial sums, reduce, store final
    accum_timing = estimate_vector_op(
        "AccumCkvKr",
        load_bytes=k_cores * M * N * 4,
        compute_elements=M * N * (k_cores - 1),
        store_bytes=M * N * out_size,
        hw=hw, active_cores=vec_cores,
    )

    return cube_timing, accum_timing


# ---------------------------------------------------------------------------
# Default block specs per matmul (from kernel code analysis)
# ---------------------------------------------------------------------------

def get_default_block_specs(params: 'OperatorParams', hw: AscendHWSpec):
    """Return default MatmulBlockSpec for each matmul based on quant mode."""
    qm = params.quant_mode
    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    is_fp8 = is_mxfp8(qm)
    is_quant = is_full_quant(qm) or is_int8_quant(qm) or is_fp8

    # MM3 weight dtype
    mm3_wt_size = 1 if is_quant else 2
    mm3_act_size = 2  # Cq is always BF16 after RMSNorm

    if is_fp8 or is_quant:
        mm1_baseK = 256
        mm2_baseK = 256
        mm3_baseK = 128
    else:
        mm1_baseK = 128
        mm2_baseK = 128
        mm3_baseK = 64

    mm1_baseN = 64 if is_fp8 else 128
    mm2_baseN = 128
    mm3_baseN = 128 if (is_fp8 or is_quant) else 256

    specs = {}
    specs["MM1_Cq"] = MatmulBlockSpec(
        "MM1_Cq", baseM=32, baseN=mm1_baseN, baseK=mm1_baseK, stepK=4,
        mode="split_k", dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
    )
    specs["MM2_CkvKr"] = MatmulBlockSpec(
        "MM2_CkvKr", baseM=32, baseN=mm2_baseN, baseK=mm2_baseK, stepK=4,
        mode="split_k", dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
    )
    specs["MM3_QcQr"] = MatmulBlockSpec(
        "MM3_QcQr", baseM=32, baseN=mm3_baseN, baseK=mm3_baseK, stepK=4,
        mode="split_k", dtype_a_size=mm3_act_size, dtype_b_size=mm3_wt_size, dtype_c_size=out_size,
    )
    specs["MM4_Qn"] = MatmulBlockSpec(
        "MM4_Qn", baseM=32, baseN=128, baseK=128, stepK=1,
        mode="full_load", dtype_a_size=2, dtype_b_size=wt_size, dtype_c_size=2,
    )

    # Derive stepK from L1 constraint for each spec
    mm_dims = {
        "MM1_Cq": (params.He, params.Hcq),
        "MM2_CkvKr": (params.He, params.Hckv + params.Dr),
        "MM3_QcQr": (params.Hcq, params.N * (params.D + params.Dr)),
        "MM4_Qn": (params.D, params.Hckv),
    }
    T = params.T
    for mm_name, (K, N_total) in mm_dims.items():
        spec = specs[mm_name]
        if spec.mode == "full_load":
            continue
        n_per_core = math.ceil(N_total / hw.aic_num)
        derived = derive_stepK(spec, K, hw, n_per_core, M=T)
        if derived > 0:
            spec.stepK = derived

    return specs


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


def estimate_all_stages(params: OperatorParams, tiling, hw: AscendHWSpec,
                        block_specs=None, mm2_split_k_cores: int = 0):
    """Estimate timing for all matmul and vector stages.

    When hw.has_cycle_spec is True, uses the detailed inner-loop model.
    Otherwise falls back to the chip-total estimate_matmul().

    Args:
        mm2_split_k_cores: If > 0, use cross-core K-split for MM2 with this
            many cores. Adds an "AccumCkvKr" vector stage for partial sum reduction.
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

    use_detailed = hw.has_cycle_spec
    if use_detailed and block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    stages = {}

    if use_detailed:
        # MM1: tokenX[T, He] x weightDq[He, Hcq] -> Cq[T, Hcq]
        stages["MM1_Cq"] = estimate_matmul_detailed(
            "MM1_Cq", T, He, Hcq,
            active_cores=tiling.mm1_block_num,
            hw=hw, block=block_specs["MM1_Cq"], a_reused=False,
        )
        # MM2: tokenX reused from MM1 (L2 warm)
        if mm2_split_k_cores > 0 and hw.has_cycle_spec:
            cube_t, accum_t = estimate_mm2_split_k_cross_core(
                T, He, Hckv + Dr, mm2_split_k_cores,
                hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
                vec_cores=tiling.vector_block_num, out_size=out_size,
            )
            stages["MM2_CkvKr"] = cube_t
            stages["AccumCkvKr"] = accum_t
        else:
            stages["MM2_CkvKr"] = estimate_matmul_detailed(
                "MM2_CkvKr", T, He, Hckv + Dr,
                active_cores=tiling.mm2_block_num,
                hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
            )
        # MM3: Cq[T, Hcq] x weightUqQr[Hcq, N*(D+Dr)] -> QcQr
        stages["MM3_QcQr"] = estimate_matmul_detailed(
            "MM3_QcQr", T, Hcq, N * (D + Dr),
            active_cores=tiling.mm3_block_num,
            hw=hw, block=block_specs["MM3_QcQr"], a_reused=False,
        )
        # MM4: Qc[T, D] x Uk[D, Hckv] -> Qn
        stages["MM4_Qn"] = estimate_matmul_detailed(
            "MM4_Qn", T, D, Hckv,
            active_cores=tiling.mm4_block_num,
            hw=hw, block=block_specs["MM4_Qn"], a_reused=False,
        )
    else:
        # Legacy chip-total path
        stages["MM1_Cq"] = estimate_matmul(
            "MM1_Cq", T, He, Hcq,
            active_cores=tiling.mm1_block_num, hw=hw,
            dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
            a_reused=False,
        )
        stages["MM2_CkvKr"] = estimate_matmul(
            "MM2_CkvKr", T, He, Hckv + Dr,
            active_cores=tiling.mm2_block_num, hw=hw,
            dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
            a_reused=True,
        )
        mm3_wt_size = 1 if (is_full_quant(qm) or is_mxfp8(qm) or is_int8_quant(qm)) else 2
        stages["MM3_QcQr"] = estimate_matmul(
            "MM3_QcQr", T, Hcq, N * (D + Dr),
            active_cores=tiling.mm3_block_num, hw=hw,
            dtype_a_size=2, dtype_b_size=mm3_wt_size, dtype_c_size=out_size,
            a_reused=False,
        )
        stages["MM4_Qn"] = estimate_matmul(
            "MM4_Qn", T, D, Hckv,
            active_cores=tiling.mm4_block_num, hw=hw,
            dtype_a_size=2, dtype_b_size=wt_size, dtype_c_size=2,
            a_reused=False,
        )

    vec_cores = tiling.vector_block_num

    # RMSNorm Cq
    rmsnorm_cq_load = T * Hcq * out_size
    rmsnorm_cq_compute = T * Hcq * 3
    rmsnorm_cq_store = T * Hcq * 2
    stages["RmsNormCq"] = estimate_vector_op(
        "RmsNormCq", rmsnorm_cq_load, rmsnorm_cq_compute, rmsnorm_cq_store,
        hw, active_cores=vec_cores,
    )

    # RMSNorm CkvKr
    ckv_kr_size = Hckv + Dr
    rmsnorm_ckvkr_load = T * ckv_kr_size * out_size
    rmsnorm_ckvkr_compute = T * Hckv * 3
    rmsnorm_ckvkr_store = T * Hckv * 2
    stages["RmsNormCkvKr"] = estimate_vector_op(
        "RmsNormCkvKr", rmsnorm_ckvkr_load, rmsnorm_ckvkr_compute, rmsnorm_ckvkr_store,
        hw, active_cores=vec_cores,
    )

    # RoPE Kr
    rope_kr_load = T * Dr * 2 + T * Dr * 2
    rope_kr_compute = T * Dr * 4
    rope_kr_store = T * Dr * 2
    stages["RopeKr"] = estimate_vector_op(
        "RopeKr", rope_kr_load, rope_kr_compute, rope_kr_store,
        hw, active_cores=vec_cores,
    )

    # Scatter CkvKr
    scatter_store = T * (Hckv + Dr) * 2
    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", T * (Hckv + Dr) * 2, T * (Hckv + Dr), scatter_store,
        hw, active_cores=vec_cores, is_scatter=True,
    )

    # DequantQc
    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        dequant_load = T * N * D * out_size
        dequant_compute = T * N * D * 2
        dequant_store = T * N * D * 2
        stages["DequantQc"] = estimate_vector_op(
            "DequantQc", dequant_load, dequant_compute, dequant_store,
            hw, active_cores=vec_cores,
        )

    # RoPE Qr
    rope_qr_load = T * N * Dr * 2 + T * Dr * 2
    rope_qr_compute = T * N * Dr * 4
    rope_qr_store = T * N * Dr * 2
    stages["RopeQr"] = estimate_vector_op(
        "RopeQr", rope_qr_load, rope_qr_compute, rope_qr_store,
        hw, active_cores=vec_cores,
    )

    # DynamicQuant Qn
    if params.query_quant_mode == 1:
        dyn_quant_load = T * N * Hckv * 2
        dyn_quant_compute = T * N * Hckv * 3
        dyn_quant_store = T * N * Hckv * 1
        stages["DynamicQuantQn"] = estimate_vector_op(
            "DynamicQuantQn", dyn_quant_load, dyn_quant_compute, dyn_quant_store,
            hw, active_cores=vec_cores,
        )

    # MulQr
    if params.query_quant_mode == 1:
        mul_qr_load = T * N * Dr * 2
        mul_qr_compute = T * N * Dr * 1
        mul_qr_store = T * N * Dr * 2
        stages["MulQr"] = estimate_vector_op(
            "MulQr", mul_qr_load, mul_qr_compute, mul_qr_store,
            hw, active_cores=vec_cores,
        )

    return stages
