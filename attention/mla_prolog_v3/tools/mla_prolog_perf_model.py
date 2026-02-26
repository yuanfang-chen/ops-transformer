#!/usr/bin/env python3
"""Theoretical performance model for MlaPrologV3 on Ascend 910B.

This models the split-N kernel path using source-level tiling and a roofline-style
HBM/L2 traffic estimate. Matmul FLOPs are exact; vector-side FLOPs are coefficient-
based approximations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPORT_VERSION = "1.0"
BLOCK_SIZE = 32
HIGH_THROUGHPUT_D_SIZE = 128
GROUP_COMPUTE_CUBE_NUM_PER_GROUP = 8
GROUP_COMPUTE_T_SIZE = 1
GROUP_COMPUTE_N_SIZE = 8
GROUP_COMPUTE_MIN_AIC_NUM = 16
GROUP_COMPUTE_MIN_AIV_NUM = 32
FP8_BLOCK = 32
BYTE_BLOCK = 32

DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "int8": 1,
    "fp8_e4m3": 1,
    "fp8_e8m0": 1,
    "int32": 4,
    "float": 4,
    "int64": 8,
}

CACHE_MODES = {"PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"}


class ModelError(ValueError):
    pass


@dataclass
class HardwareProfile:
    name: str = "ascend910b"
    aic_num: int = 20
    aiv_num: int = 40
    ub_size_bytes: int = 196_608
    l1_size_bytes: int = 524_288
    l2_size_bytes: int = 201_326_592
    cube_peak_tflops: float = 160.0
    vector_peak_tflops: float = 20.0
    hbm_bw_gbps: float = 1200.0
    l2_bw_gbps: float = 3500.0
    cube_eff: float = 0.65
    vector_eff: float = 0.55
    hbm_eff: float = 0.75
    l2_eff: float = 0.70
    l2_reserve_bytes: Optional[int] = None
    l2_reserve_ratio: float = 0.30

    @property
    def cv_ratio(self) -> int:
        if self.aic_num <= 0:
            return 1
        return max(1, self.aiv_num // self.aic_num)

    @property
    def effective_cube_ops(self) -> float:
        return self.cube_peak_tflops * 1e12 * self.cube_eff

    @property
    def effective_vector_ops(self) -> float:
        return self.vector_peak_tflops * 1e12 * self.vector_eff

    @property
    def effective_hbm_bw(self) -> float:
        return self.hbm_bw_gbps * 1e9 * self.hbm_eff

    @property
    def effective_l2_bw(self) -> float:
        return self.l2_bw_gbps * 1e9 * self.l2_eff

    @property
    def retained_l2_budget(self) -> int:
        reserve = self.l2_reserve_bytes
        if reserve is None:
            reserve = int(self.l2_size_bytes * self.l2_reserve_ratio)
        reserve = min(max(reserve, 0), self.l2_size_bytes)
        return self.l2_size_bytes - reserve


@dataclass
class ProblemSpec:
    T: int
    He: int
    Hcq: int
    N: int
    D: int
    Dr: int
    Hckv: int
    Nkv: int
    Dtile: int
    B: int = 1
    S: Optional[int] = None
    blockNum: int = 0
    blockSize: int = 128
    cache_mode: str = "PA_BSND"
    quant_mode: int = 0
    query_norm_flag: int = 0
    weight_quant_mode: Optional[int] = None
    kv_cache_quant_mode: Optional[int] = None
    query_quant_mode: int = 0
    ckvkr_repo_mode: int = 0
    quant_scale_repo_mode: int = 0
    tile_size: int = 128
    qc_qr_scale: float = 1.0
    kc_scale: float = 1.0

    def validate(self) -> None:
        dims = {
            "T": self.T,
            "He": self.He,
            "Hcq": self.Hcq,
            "N": self.N,
            "D": self.D,
            "Dr": self.Dr,
            "Hckv": self.Hckv,
            "Nkv": self.Nkv,
            "Dtile": self.Dtile,
        }
        for k, v in dims.items():
            if v <= 0:
                raise ModelError(f"{k} must be > 0 for v1 model, got {v}")
        if self.cache_mode not in CACHE_MODES:
            raise ModelError(f"Unsupported cache_mode: {self.cache_mode}")
        if self.quant_mode < 0 or self.quant_mode > 9:
            raise ModelError(f"quant_mode must be in [0, 9], got {self.quant_mode}")
        if self.query_norm_flag not in (0, 1):
            raise ModelError("query_norm_flag must be 0 or 1")
        if self.B <= 0:
            raise ModelError("B must be > 0")
        if self.S is not None and self.S > 0 and self.B * self.S != self.T:
            raise ModelError(f"B*S must equal T when S is provided, got {self.B}*{self.S}!={self.T}")
        if self.S is None:
            self.S = self.T // self.B if self.T % self.B == 0 else None
        if self.Dtile < self.Hckv:
            # legal in some packed modes, but warn in notes via caller
            pass

    @property
    def head_size_qc(self) -> int:
        return self.N * self.D

    @property
    def head_size_qr(self) -> int:
        return self.N * self.Dr


@dataclass
class BaseParamsLike:
    batchSize: int
    stepBatchSize: int
    tokenSize: int
    seq1Size: int
    seq2Size: int
    headSizeX: int
    headSizeCq: int
    headSizeCkv: int
    headSizeQc: int
    headSizeQr: int
    headSizeKr: int
    numHeadSize: int
    numHeadKvSize: int
    dimHeadSizeQc: int
    dimHeadRope: int
    blockNum: int
    blockSize: int
    dtileSize: int
    mm1BlockNum: int
    mm2BlockNum: int
    mm3BlockNum: int
    mm4BlockNum: int
    vectorBlockNum: int
    mm1SingleCoreN: int
    mm2SingleCoreN: int
    mm3SingleCoreN: int
    mm4SingleCoreBatch: int
    stepNumHeadDequant: int
    queryNormFlag: int = 0
    kvQuantMode: int = 0
    tileSize: int = 128
    ckvkrRepoMode: int = 0
    quantScaleRepoMode: int = 0
    qcQrScale: float = 1.0
    kcScale: float = 1.0


@dataclass
class QuantDTypePlan:
    quant_mode: int
    label: str
    mm_input_type: str
    mm_qcqr_input_type: str
    mm_qn_input_type: str
    mm_cq_output_type: str
    mm_ckvkr_output_type: str
    mm_qcqr_output_type: str
    kv_cache_type: str
    kr_cache_type: str
    query_out_type: str
    query_rope_out_type: str
    rmsnorm_cq_output_type: str
    rmsnorm_ckv_output_type: str
    dequant_scale_qnorm_type: str
    is_pertile: bool = False


@dataclass
class TilingSpec:
    splitMFlag: int
    stepBatchSize: int
    vectorBlockNum: int
    stepNumHeadDequant: int
    mm1BlockNum: int
    mm2BlockNum: int
    mm3BlockNum: int
    mm4BlockNum: int
    mm1SingleCoreN: int
    mm2SingleCoreN: int
    mm3SingleCoreN: int
    mm4SingleCoreBatch: int
    enableDequantOpt: bool
    enableGroupComputeOpt: bool
    aic_num: int
    aiv_num: int
    cv_ratio: int
    notes: List[str] = field(default_factory=list)


@dataclass
class TrafficBreakdown:
    logical_read_bytes: int = 0
    logical_write_bytes: int = 0
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    l2_read_bytes: int = 0
    l2_write_bytes: int = 0

    def add(self, other: "TrafficBreakdown") -> None:
        self.logical_read_bytes += other.logical_read_bytes
        self.logical_write_bytes += other.logical_write_bytes
        self.hbm_read_bytes += other.hbm_read_bytes
        self.hbm_write_bytes += other.hbm_write_bytes
        self.l2_read_bytes += other.l2_read_bytes
        self.l2_write_bytes += other.l2_write_bytes

    def scaled(self, n: int) -> "TrafficBreakdown":
        return TrafficBreakdown(
            logical_read_bytes=self.logical_read_bytes * n,
            logical_write_bytes=self.logical_write_bytes * n,
            hbm_read_bytes=self.hbm_read_bytes * n,
            hbm_write_bytes=self.hbm_write_bytes * n,
            l2_read_bytes=self.l2_read_bytes * n,
            l2_write_bytes=self.l2_write_bytes * n,
        )

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class StageCost:
    name: str
    engine: str
    flops: float
    traffic: TrafficBreakdown
    t_compute_s: float
    t_hbm_mem_s: float
    t_l2_mem_s: float
    t_mem_s: float
    t_stage_s: float
    bottleneck: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "flops": self.flops,
            "traffic": self.traffic.to_dict(),
            "t_compute_s": self.t_compute_s,
            "t_hbm_mem_s": self.t_hbm_mem_s,
            "t_l2_mem_s": self.t_l2_mem_s,
            "t_mem_s": self.t_mem_s,
            "t_stage_s": self.t_stage_s,
            "bottleneck": self.bottleneck,
            "notes": self.notes,
        }


@dataclass
class ScenarioEstimate:
    name: str
    step_batch_size: int
    num_steps: int
    tail_step_batch_size: int
    full_step_time_s: float
    tail_step_time_s: float
    total_time_s: float
    stages_full_step: List[StageCost]
    stages_tail_step: List[StageCost]
    traffic_total: TrafficBreakdown
    critical_path_full_step: List[str]
    critical_path_tail_step: List[str]
    schedule_full_step: Dict[str, Dict[str, float]]
    schedule_tail_step: Dict[str, Dict[str, float]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "step_batch_size": self.step_batch_size,
            "num_steps": self.num_steps,
            "tail_step_batch_size": self.tail_step_batch_size,
            "full_step_time_s": self.full_step_time_s,
            "tail_step_time_s": self.tail_step_time_s,
            "total_time_s": self.total_time_s,
            "stages_full_step": [s.to_dict() for s in self.stages_full_step],
            "stages_tail_step": [s.to_dict() for s in self.stages_tail_step],
            "traffic_total": self.traffic_total.to_dict(),
            "critical_path_full_step": self.critical_path_full_step,
            "critical_path_tail_step": self.critical_path_tail_step,
            "schedule_full_step": self.schedule_full_step,
            "schedule_tail_step": self.schedule_tail_step,
            "notes": self.notes,
        }


@dataclass
class KernelEstimateReport:
    report_version: str
    inputs: Dict[str, Any]
    assumptions: Dict[str, Any]
    tiling: Dict[str, Any]
    matmul_flops_exact: Dict[str, Any]
    vector_flops_approx: Dict[str, Any]
    traffic_cold: Dict[str, Any]
    traffic_warm: Dict[str, Any]
    timing_cold: Dict[str, Any]
    timing_warm: Dict[str, Any]
    critical_path_nodes: Dict[str, Any]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCoefficients:
    rmsnorm_flops_per_elem: float = 6.0
    rmsnorm_flops_per_row_extra: float = 2.0
    rope_flops_per_elem: float = 6.0
    dequant_flops_per_elem: float = 2.0
    dynamic_quant_flops_per_elem: float = 4.0
    special_op_flop_equiv: float = 8.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class StageModelContext:
    problem: ProblemSpec
    base: BaseParamsLike
    quant: QuantDTypePlan
    tiling: TilingSpec
    hw: HardwareProfile
    coeffs: ModelCoefficients
    warm_resident: Dict[str, bool]


QUANT_MODE_TABLE: Dict[int, QuantDTypePlan] = {
    0: QuantDTypePlan(0, "NO_QUANT", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "float", False),
    1: QuantDTypePlan(1, "PARTIAL_QUANT_KV_NO_QUANT", "bf16", "int8", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "bf16", "int8", "bf16", "float", False),
    2: QuantDTypePlan(2, "PARTIAL_QUANT_KV_PER_CHANNEL", "bf16", "int8", "bf16", "bf16", "bf16", "bf16", "int8", "int8", "int8", "bf16", "int8", "int8", "float", False),
    3: QuantDTypePlan(3, "FULL_INT8_KV_NO_QUANT", "int8", "int8", "bf16", "int32", "int32", "int32", "bf16", "bf16", "bf16", "bf16", "int8", "bf16", "float", False),
    4: QuantDTypePlan(4, "FULL_INT8_KV_PER_TENSOR", "int8", "int8", "bf16", "int32", "int32", "int32", "int8", "bf16", "int8", "bf16", "int8", "int8", "float", False),
    5: QuantDTypePlan(5, "PARTIAL_QUANT_KV_PER_TILE", "bf16", "int8", "bf16", "bf16", "bf16", "bf16", "int8", "bf16", "bf16", "bf16", "int8", "int8", "float", True),
    6: QuantDTypePlan(6, "FULL_INT8_KV_PER_TILE", "int8", "int8", "bf16", "int32", "int32", "int32", "int8", "bf16", "bf16", "bf16", "int8", "int8", "float", True),
    7: QuantDTypePlan(7, "MXFP8_KV_NO_QUANT", "fp8_e4m3", "fp8_e4m3", "bf16", "float", "float", "float", "bf16", "bf16", "bf16", "bf16", "fp8_e4m3", "bf16", "fp8_e8m0", False),
    8: QuantDTypePlan(8, "MXFP8_KV_PER_TENSOR", "fp8_e4m3", "fp8_e4m3", "bf16", "float", "float", "float", "fp8_e4m3", "bf16", "fp8_e4m3", "bf16", "fp8_e4m3", "fp8_e4m3", "fp8_e8m0", False),
    9: QuantDTypePlan(9, "MXFP8_KV_PER_TILE", "fp8_e4m3", "fp8_e4m3", "bf16", "float", "float", "float", "fp8_e4m3", "bf16", "bf16", "bf16", "fp8_e4m3", "fp8_e4m3", "fp8_e8m0", True),
}


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        raise ModelError("division by zero")
    return (a + b - 1) // b


def align(n: int, a: int) -> int:
    if a == 0:
        return 0
    return ceil_div(n, a) * a


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ModelError(f"Unsupported dtype: {dtype}") from exc


def calc_single_core_n(n: int, core_num: int, align_num: int) -> int:
    return ceil_div(n, align_num * core_num) * align_num


def infer_quant_mode_from_weight_kv(weight_quant_mode: Optional[int], kv_cache_quant_mode: Optional[int]) -> int:
    if weight_quant_mode is None and kv_cache_quant_mode is None:
        return 0
    w = 0 if weight_quant_mode is None else int(weight_quant_mode)
    k = 0 if kv_cache_quant_mode is None else int(kv_cache_quant_mode)
    mapping = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 2): 2,
        (1, 3): 5,
        (2, 0): 3,
        (2, 1): 4,
        (2, 3): 6,
        (3, 0): 7,
        (3, 1): 8,
        (3, 3): 9,
    }
    if (w, k) not in mapping:
        raise ModelError(f"Cannot infer quant_mode from weight_quant_mode={w}, kv_cache_quant_mode={k}")
    return mapping[(w, k)]


def parse_problem_from_shape(data: Mapping[str, Any], args: argparse.Namespace) -> ProblemSpec:
    try:
        quant_mode = args.quant_mode if args.quant_mode is not None else int(data.get("quant_mode", infer_quant_mode_from_weight_kv(data.get("weight_quant_mode"), data.get("kv_cache_quant_mode"))))
        p = ProblemSpec(
            T=int(data["T"]),
            He=int(data["He"]),
            Hcq=int(data["Hcq"]),
            N=int(data["N"]),
            D=int(data["D"]),
            Dr=int(data["Dr"]),
            Hckv=int(data["Hckv"]),
            Nkv=int(data["Nkv"]),
            Dtile=int(data["Dtile"]),
            B=int(data.get("B", 1)),
            S=None if data.get("S") is None else int(data.get("S")),
            blockNum=int(data.get("blockNum", 0)),
            blockSize=int(data.get("blockSize", 128)),
            cache_mode=str(args.cache_mode or data.get("cache_mode", "PA_BSND")),
            quant_mode=quant_mode,
            query_norm_flag=int(args.query_norm_flag if args.query_norm_flag is not None else data.get("query_norm_flag", 0)),
            weight_quant_mode=None if data.get("weight_quant_mode") is None else int(data.get("weight_quant_mode")),
            kv_cache_quant_mode=None if data.get("kv_cache_quant_mode") is None else int(data.get("kv_cache_quant_mode")),
            query_quant_mode=int(data.get("query_quant_mode", 0)),
            ckvkr_repo_mode=int(data.get("ckvkr_repo_mode", 0)),
            quant_scale_repo_mode=int(data.get("quant_scale_repo_mode", 0)),
            tile_size=int(data.get("tile_size", 128)),
            qc_qr_scale=float(data.get("qc_qr_scale", 1.0)),
            kc_scale=float(data.get("kc_scale", 1.0)),
        )
    except KeyError as exc:
        raise ModelError(f"Missing required shape-mode key: {exc.args[0]}") from exc
    p.validate()
    return p


def parse_base_params_mode(data: Mapping[str, Any], args: argparse.Namespace) -> Tuple[ProblemSpec, BaseParamsLike]:
    base_src = data.get("base_params", data)
    try:
        quant_mode = args.quant_mode if args.quant_mode is not None else int(data.get("quant_mode", infer_quant_mode_from_weight_kv(data.get("weight_quant_mode"), data.get("kv_cache_quant_mode"))))
        cache_mode = str(args.cache_mode or data.get("cache_mode", "PA_BSND"))
        query_norm_flag = int(args.query_norm_flag if args.query_norm_flag is not None else data.get("query_norm_flag", base_src.get("queryNormFlag", 0)))
        problem = ProblemSpec(
            T=int(base_src["tokenSize"]),
            He=int(base_src["headSizeX"]),
            Hcq=int(base_src["headSizeCq"]),
            N=int(base_src["numHeadSize"]),
            D=int(base_src["dimHeadSizeQc"]),
            Dr=int(base_src["dimHeadRope"]),
            Hckv=int(base_src["headSizeCkv"]),
            Nkv=int(base_src.get("numHeadKvSize", base_src.get("seq2Size", 1))),
            Dtile=int(base_src.get("dtileSize", base_src["headSizeCkv"])),
            B=int(base_src.get("batchSize", 1)),
            S=int(base_src.get("seq1Size", 0)) or None,
            blockNum=int(base_src.get("blockNum", 0)),
            blockSize=int(base_src.get("blockSize", 128)),
            cache_mode=cache_mode,
            quant_mode=quant_mode,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=None if data.get("weight_quant_mode") is None else int(data.get("weight_quant_mode")),
            kv_cache_quant_mode=None if data.get("kv_cache_quant_mode") is None else int(data.get("kv_cache_quant_mode")),
            query_quant_mode=int(data.get("query_quant_mode", 0)),
            ckvkr_repo_mode=int(data.get("ckvkr_repo_mode", base_src.get("ckvkrRepoMode", 0))),
            quant_scale_repo_mode=int(data.get("quant_scale_repo_mode", base_src.get("quantScaleRepoMode", 0))),
            tile_size=int(data.get("tile_size", base_src.get("tileSize", 128))),
            qc_qr_scale=float(data.get("qc_qr_scale", base_src.get("qcQrScale", 1.0))),
            kc_scale=float(data.get("kc_scale", base_src.get("kcScale", 1.0))),
        )
        problem.validate()
        base = BaseParamsLike(
            batchSize=int(base_src["batchSize"]),
            stepBatchSize=int(base_src["stepBatchSize"]),
            tokenSize=int(base_src["tokenSize"]),
            seq1Size=int(base_src["seq1Size"]),
            seq2Size=int(base_src["seq2Size"]),
            headSizeX=int(base_src["headSizeX"]),
            headSizeCq=int(base_src["headSizeCq"]),
            headSizeCkv=int(base_src["headSizeCkv"]),
            headSizeQc=int(base_src["headSizeQc"]),
            headSizeQr=int(base_src["headSizeQr"]),
            headSizeKr=int(base_src["headSizeKr"]),
            numHeadSize=int(base_src["numHeadSize"]),
            numHeadKvSize=int(base_src.get("numHeadKvSize", problem.Nkv)),
            dimHeadSizeQc=int(base_src["dimHeadSizeQc"]),
            dimHeadRope=int(base_src["dimHeadRope"]),
            blockNum=int(base_src["blockNum"]),
            blockSize=int(base_src["blockSize"]),
            dtileSize=int(base_src.get("dtileSize", problem.Dtile)),
            mm1BlockNum=int(base_src["mm1BlockNum"]),
            mm2BlockNum=int(base_src["mm2BlockNum"]),
            mm3BlockNum=int(base_src["mm3BlockNum"]),
            mm4BlockNum=int(base_src["mm4BlockNum"]),
            vectorBlockNum=int(base_src["vectorBlockNum"]),
            mm1SingleCoreN=int(base_src["mm1SingleCoreN"]),
            mm2SingleCoreN=int(base_src["mm2SingleCoreN"]),
            mm3SingleCoreN=int(base_src["mm3SingleCoreN"]),
            mm4SingleCoreBatch=int(base_src["mm4SingleCoreBatch"]),
            stepNumHeadDequant=int(base_src.get("stepNumHeadDequant", min(64, problem.N))),
            queryNormFlag=query_norm_flag,
            kvQuantMode=int(base_src.get("kvQuantMode", 0)),
            tileSize=int(base_src.get("tileSize", 128)),
            ckvkrRepoMode=int(base_src.get("ckvkrRepoMode", 0)),
            quantScaleRepoMode=int(base_src.get("quantScaleRepoMode", 0)),
            qcQrScale=float(base_src.get("qcQrScale", problem.qc_qr_scale)),
            kcScale=float(base_src.get("kcScale", problem.kc_scale)),
        )
    except KeyError as exc:
        raise ModelError(f"Missing required base-params key: {exc.args[0]}") from exc
    return problem, base


def quant_plan(quant_mode: int) -> QuantDTypePlan:
    if quant_mode not in QUANT_MODE_TABLE:
        raise ModelError(f"Unsupported quant_mode: {quant_mode}")
    return QUANT_MODE_TABLE[quant_mode]


def derive_tiling(problem: ProblemSpec, hw: HardwareProfile, q: QuantDTypePlan) -> Tuple[TilingSpec, BaseParamsLike]:
    notes: List[str] = []
    split_m = 0
    aic_num = hw.aic_num
    aiv_num = hw.aiv_num
    enable_group = False
    enable_dequant = False

    step_batch_size = min(128, problem.T)
    if step_batch_size <= 0:
        raise ModelError("Empty tensor mode is unsupported in v1 model")
    vector_block_num = min(step_batch_size, aiv_num)
    step_num_head_dequant = min(64, problem.N) if problem.D == HIGH_THROUGHPUT_D_SIZE else min(16, problem.N)

    cv_ratio = max(1, aiv_num // aic_num)
    if (problem.quant_mode in (1, 2)
        and problem.T == GROUP_COMPUTE_T_SIZE
        and problem.Nkv == GROUP_COMPUTE_N_SIZE
        and aiv_num >= GROUP_COMPUTE_MIN_AIV_NUM
        and aic_num >= GROUP_COMPUTE_MIN_AIC_NUM
        and cv_ratio != 1):
        enable_group = True
        aiv_num = 32
        aic_num = 16
        cv_ratio = 2
        notes.append("enableGroupComputeOpt=true (source tiling heuristic matched)")
    elif (q.mm_qcqr_input_type == "int8" and problem.N >= GROUP_COMPUTE_N_SIZE) or q.mm_qcqr_input_type == "fp8_e4m3":
        enable_dequant = True
        notes.append("enableDequantOpt=true (source tiling heuristic matched)")

    align1 = BLOCK_SIZE // dtype_size(q.mm_input_type)
    mm1_single = calc_single_core_n(problem.Hcq, aic_num, align1)
    mm1_single = max(mm1_single, 64)
    mm1_blocks = ceil_div(problem.Hcq, mm1_single)

    hckv_kr = problem.Hckv + problem.Dr
    if aic_num >= 9:
        mm2_single = 64
        mm2_blocks_raw = hckv_kr // 64
        if mm2_blocks_raw <= 0 or hckv_kr % 64 != 0:
            mm2_blocks = ceil_div(hckv_kr, 64)
            notes.append(
                "mm2BlockNum non-64-aligned shape detected; model uses ceil_div safeguard instead of raw floor division"
            )
        else:
            mm2_blocks = mm2_blocks_raw
    else:
        align2 = BLOCK_SIZE // dtype_size(q.mm_input_type)
        mm2_single = calc_single_core_n(hckv_kr, aic_num, align2)
        mm2_blocks = ceil_div(hckv_kr, mm2_single)

    mm3_total_n = problem.N * (problem.D + problem.Dr)
    if enable_group:
        mm3_single = calc_single_core_n(problem.N * problem.D, GROUP_COMPUTE_CUBE_NUM_PER_GROUP, problem.D)
    elif enable_dequant:
        mm3_single = calc_single_core_n(mm3_total_n, aic_num, problem.D + problem.Dr)
    else:
        align3 = BLOCK_SIZE // dtype_size(q.mm_qcqr_input_type)
        mm3_single = calc_single_core_n(mm3_total_n, aic_num, align3)
    mm3_blocks = ceil_div(mm3_total_n, mm3_single)

    mm4_single = ceil_div(problem.N, aic_num)
    mm4_blocks = ceil_div(problem.N, mm4_single)

    tiling = TilingSpec(
        splitMFlag=split_m,
        stepBatchSize=step_batch_size,
        vectorBlockNum=vector_block_num,
        stepNumHeadDequant=step_num_head_dequant,
        mm1BlockNum=mm1_blocks,
        mm2BlockNum=mm2_blocks,
        mm3BlockNum=mm3_blocks,
        mm4BlockNum=mm4_blocks,
        mm1SingleCoreN=mm1_single,
        mm2SingleCoreN=mm2_single,
        mm3SingleCoreN=mm3_single,
        mm4SingleCoreBatch=mm4_single,
        enableDequantOpt=enable_dequant,
        enableGroupComputeOpt=enable_group,
        aic_num=aic_num,
        aiv_num=aiv_num,
        cv_ratio=cv_ratio,
        notes=notes,
    )

    base = BaseParamsLike(
        batchSize=problem.B,
        stepBatchSize=step_batch_size,
        tokenSize=problem.T,
        seq1Size=problem.S or 0,
        seq2Size=problem.Nkv,
        headSizeX=problem.He,
        headSizeCq=problem.Hcq,
        headSizeCkv=problem.Hckv,
        headSizeQc=problem.N * problem.D,
        headSizeQr=problem.N * problem.Dr,
        headSizeKr=problem.Dr,
        numHeadSize=problem.N,
        numHeadKvSize=problem.Nkv,
        dimHeadSizeQc=problem.D,
        dimHeadRope=problem.Dr,
        blockNum=problem.blockNum,
        blockSize=problem.blockSize,
        dtileSize=problem.Dtile,
        mm1BlockNum=mm1_blocks,
        mm2BlockNum=mm2_blocks,
        mm3BlockNum=mm3_blocks,
        mm4BlockNum=mm4_blocks,
        vectorBlockNum=vector_block_num,
        mm1SingleCoreN=mm1_single,
        mm2SingleCoreN=mm2_single,
        mm3SingleCoreN=mm3_single,
        mm4SingleCoreBatch=mm4_single,
        stepNumHeadDequant=step_num_head_dequant,
        queryNormFlag=problem.query_norm_flag,
        kvQuantMode=problem.kv_cache_quant_mode or 0,
        tileSize=problem.tile_size,
        ckvkrRepoMode=problem.ckvkr_repo_mode,
        quantScaleRepoMode=problem.quant_scale_repo_mode,
        qcQrScale=problem.qc_qr_scale,
        kcScale=problem.kc_scale,
    )
    return tiling, base


def build_hw_from_args(args: argparse.Namespace, data: Mapping[str, Any]) -> HardwareProfile:
    hw_data = data.get("hardware", {})
    h = HardwareProfile(
        name=str(hw_data.get("name", "ascend910b")),
        aic_num=int(hw_data.get("aic_num", 20)),
        aiv_num=int(hw_data.get("aiv_num", 40)),
        ub_size_bytes=int(hw_data.get("ub_size_bytes", 196608)),
        l1_size_bytes=int(hw_data.get("l1_size_bytes", 524288)),
        l2_size_bytes=int(hw_data.get("l2_size_bytes", 201326592)),
        cube_peak_tflops=float(hw_data.get("cube_peak_tflops", 160.0)),
        vector_peak_tflops=float(hw_data.get("vector_peak_tflops", 20.0)),
        hbm_bw_gbps=float(hw_data.get("hbm_bw_gbps", 1200.0)),
        l2_bw_gbps=float(hw_data.get("l2_bw_gbps", 3500.0)),
        cube_eff=float(hw_data.get("cube_eff", 0.65)),
        vector_eff=float(hw_data.get("vector_eff", 0.55)),
        hbm_eff=float(hw_data.get("hbm_eff", 0.75)),
        l2_eff=float(hw_data.get("l2_eff", 0.70)),
        l2_reserve_bytes=(None if hw_data.get("l2_reserve_bytes") is None else int(hw_data.get("l2_reserve_bytes"))),
        l2_reserve_ratio=float(hw_data.get("l2_reserve_ratio", 0.30)),
    )
    overrides = {
        "cube_peak_tflops": args.cube_peak_tflops,
        "vector_peak_tflops": args.vector_peak_tflops,
        "hbm_bw_gbps": args.hbm_bw_gbps,
        "l2_bw_gbps": args.l2_bw_gbps,
        "cube_eff": args.cube_eff,
        "vector_eff": args.vector_eff,
        "hbm_eff": args.hbm_eff,
        "l2_eff": args.l2_eff,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(h, k, v)
    if args.l2_reserve_bytes is not None:
        h.l2_reserve_bytes = int(args.l2_reserve_bytes)
    elif args.l2_reserve_ratio is not None:
        h.l2_reserve_ratio = float(args.l2_reserve_ratio)
    return h


def build_coeffs(data: Mapping[str, Any]) -> ModelCoefficients:
    c = data.get("coefficients", {})
    return ModelCoefficients(
        rmsnorm_flops_per_elem=float(c.get("rmsnorm_flops_per_elem", 6.0)),
        rmsnorm_flops_per_row_extra=float(c.get("rmsnorm_flops_per_row_extra", 2.0)),
        rope_flops_per_elem=float(c.get("rope_flops_per_elem", 6.0)),
        dequant_flops_per_elem=float(c.get("dequant_flops_per_elem", 2.0)),
        dynamic_quant_flops_per_elem=float(c.get("dynamic_quant_flops_per_elem", 4.0)),
        special_op_flop_equiv=float(c.get("special_op_flop_equiv", 8.0)),
    )


def bytes_weight_dq(p: ProblemSpec, q: QuantDTypePlan) -> int:
    return p.He * p.Hcq * dtype_size(q.mm_input_type)


def bytes_weight_dkvkr(p: ProblemSpec, q: QuantDTypePlan) -> int:
    return p.He * (p.Hckv + p.Dr) * dtype_size(q.mm_input_type)


def bytes_weight_uqqr(p: ProblemSpec, q: QuantDTypePlan) -> int:
    return p.Hcq * p.N * (p.D + p.Dr) * dtype_size(q.mm_qcqr_input_type)


def bytes_weight_uk(p: ProblemSpec, q: QuantDTypePlan) -> int:
    _ = q
    return p.N * p.D * p.Hckv * dtype_size("bf16")


def bytes_token_x(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.He * dtype_size(q.mm_input_type)


def bytes_dequant_scale_x_per_token(p: ProblemSpec, q: QuantDTypePlan) -> int:
    if q.mm_input_type == "int8":
        return dtype_size("float")
    if q.mm_input_type == "fp8_e4m3":
        return ceil_div(p.He, FP8_BLOCK) * dtype_size("fp8_e8m0")
    return 0


def bytes_dequant_scale_x(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * bytes_dequant_scale_x_per_token(p, q)


def bytes_dequant_scale_wdq(p: ProblemSpec, q: QuantDTypePlan) -> int:
    if q.mm_input_type == "int8":
        return p.Hcq * dtype_size("float")
    if q.mm_input_type == "fp8_e4m3":
        return p.Hcq * ceil_div(p.He, FP8_BLOCK) * dtype_size("fp8_e8m0")
    return 0


def bytes_dequant_scale_wdkvkr(p: ProblemSpec, q: QuantDTypePlan) -> int:
    if q.mm_input_type == "int8":
        return (p.Hckv + p.Dr) * dtype_size("float")
    if q.mm_input_type == "fp8_e4m3":
        return (p.Hckv + p.Dr) * ceil_div(p.He, FP8_BLOCK) * dtype_size("fp8_e8m0")
    return 0


def bytes_deq_scale_qcqrw(p: ProblemSpec, q: QuantDTypePlan) -> int:
    ncols = p.N * (p.D + p.Dr)
    if q.mm_qcqr_input_type == "int8":
        return ncols * dtype_size("float")
    if q.mm_qcqr_input_type == "fp8_e4m3":
        return ncols * ceil_div(p.Hcq, FP8_BLOCK) * dtype_size("fp8_e8m0")
    return 0


def bytes_smooth_scale_cq(p: ProblemSpec, q: QuantDTypePlan) -> int:
    return p.Hcq * dtype_size("float") if q.mm_qcqr_input_type == "int8" else 0


def bytes_quant_scale_ckv(p: ProblemSpec, q: QuantDTypePlan) -> int:
    if q.rmsnorm_ckv_output_type in ("int8", "fp8_e4m3"):
        if q.mm_ckvkr_output_type == "int32":
            return 32
        return p.Hckv * dtype_size("float")
    return 0


def bytes_quant_scale_ckr(p: ProblemSpec, q: QuantDTypePlan) -> int:
    return p.Dr * dtype_size("float") if q.kr_cache_type == "int8" else 0


def bytes_gamma_cq(p: ProblemSpec) -> int:
    return p.Hcq * dtype_size("bf16")


def bytes_gamma_ckv(p: ProblemSpec) -> int:
    return p.Hckv * dtype_size("bf16")


def bytes_rope_sincos(m: int, p: ProblemSpec) -> int:
    return m * p.Dr * dtype_size("bf16") * 2


def bytes_cache_index(m: int, p: ProblemSpec) -> int:
    if p.cache_mode in {"BSND", "TND"}:
        return 0
    return m * dtype_size("int64")


def bytes_query_rope_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.N * p.Dr * dtype_size(q.query_rope_out_type)


def bytes_query_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.N * p.Hckv * dtype_size(q.query_out_type)


def bytes_dequant_scale_qnope_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    _ = q
    return m * p.N * dtype_size("float")


def bytes_query_norm_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.Hcq * dtype_size(q.rmsnorm_cq_output_type)


def bytes_dequant_scale_qnorm_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    if q.mm_qcqr_input_type == "int8":
        return m * dtype_size("float")
    if q.mm_qcqr_input_type == "fp8_e4m3":
        return m * ceil_div(p.Hcq, FP8_BLOCK) * dtype_size("fp8_e8m0")
    return 0


def bytes_kv_cache_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.Nkv * p.Dtile * dtype_size(q.kv_cache_type)


def bytes_kr_cache_out(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.Nkv * p.Dr * dtype_size(q.kr_cache_type)


def bytes_workspace_mm_cq(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.Hcq * dtype_size(q.mm_cq_output_type)


def bytes_workspace_rmsnorm_cq(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.Hcq * dtype_size(q.rmsnorm_cq_output_type)


def bytes_workspace_mm_ckvkr(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * (p.Hckv + p.Dr) * dtype_size(q.mm_ckvkr_output_type)


def bytes_workspace_mm_qcqr(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * (p.N * (p.D + p.Dr)) * dtype_size(q.mm_qcqr_output_type)


def bytes_workspace_mm_qc_slice(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * (p.N * p.D) * dtype_size(q.mm_qcqr_output_type)


def bytes_workspace_mm_qr_slice(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * (p.N * p.Dr) * dtype_size(q.mm_qcqr_output_type)


def bytes_workspace_mm_qcqr_dequant(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    _ = q
    return m * (p.N * p.D) * dtype_size("bf16")


def bytes_workspace_mm_qn(m: int, p: ProblemSpec, q: QuantDTypePlan) -> int:
    return m * p.N * p.Hckv * dtype_size("bf16")


def matmul_flops_exact(p: ProblemSpec) -> Dict[str, float]:
    return {
        "MatmulCq": float(2 * p.T * p.He * p.Hcq),
        "MatmulCkvKr": float(2 * p.T * p.He * (p.Hckv + p.Dr)),
        "MatmulQcQr": float(2 * p.T * p.Hcq * (p.N * (p.D + p.Dr))),
        "MatmulQn": float(2 * p.T * p.N * p.D * p.Hckv),
    }


def matmul_flops_step(m: int, p: ProblemSpec) -> Dict[str, float]:
    return {
        "mm_cq": float(2 * m * p.He * p.Hcq),
        "mm_ckvkr": float(2 * m * p.He * (p.Hckv + p.Dr)),
        "mm_qcqr": float(2 * m * p.Hcq * (p.N * (p.D + p.Dr))),
        "mm_qn": float(2 * m * p.N * p.D * p.Hckv),
    }


def vector_flops_step(m: int, ctx: StageModelContext) -> Dict[str, float]:
    p, q, t, c = ctx.problem, ctx.quant, ctx.tiling, ctx.coeffs
    rmsnorm_cq = m * p.Hcq * c.rmsnorm_flops_per_elem + m * c.rmsnorm_flops_per_row_extra * c.special_op_flop_equiv
    if abs(p.qc_qr_scale - 1.0) > 1e-8:
        rmsnorm_cq += m * p.Hcq
    # dequant for full int8 path is fused in vector rmsnorm/scatter branches; model as vector work.
    if q.mm_input_type == "int8":
        rmsnorm_cq += m * p.Hcq * c.dequant_flops_per_elem
    rmsnorm_ckv = m * p.Hckv * c.rmsnorm_flops_per_elem + m * c.rmsnorm_flops_per_row_extra * c.special_op_flop_equiv
    if abs(p.kc_scale - 1.0) > 1e-8:
        rmsnorm_ckv += m * p.Hckv
    if q.mm_input_type == "int8":
        rmsnorm_ckv += m * (p.Hckv + p.Dr) * c.dequant_flops_per_elem
    if q.mm_input_type == "fp8_e4m3":
        # FP8 mm dequant scale application mostly inside cube matmul kernels; keep vector-side dequant minimal.
        pass
    rope_kr = m * p.Nkv * p.Dr * c.rope_flops_per_elem
    rope_qr = m * p.N * p.Dr * c.rope_flops_per_elem

    qc_post = 0.0
    if q.mm_qcqr_input_type == "int8":
        qc_post = m * p.N * p.D * c.dequant_flops_per_elem
    elif q.mm_qcqr_input_type == "fp8_e4m3":
        qc_post = m * p.N * p.D * c.dequant_flops_per_elem

    need_qn_dynamic_quant = ((q.mm_input_type == "int8" and q.kv_cache_type == "int8") or (q.mm_input_type == "fp8_e4m3" and q.kv_cache_type == "fp8_e4m3")) and (not q.is_pertile)
    dynamic_qn = 0.0
    if need_qn_dynamic_quant:
        elems = m * p.N * p.Hckv
        rows = m * p.N
        dynamic_qn = elems * c.dynamic_quant_flops_per_elem + rows * c.special_op_flop_equiv

    if t.enableDequantOpt and q.mm_qcqr_input_type in ("int8", "fp8_e4m3"):
        # Split-N dequant/rope path has extra coordination and fragmented vector work.
        qc_post *= 1.05
        rope_qr *= 1.02

    return {
        "rmsnorm_cq": float(rmsnorm_cq),
        "ckvkr_post": float(rmsnorm_ckv + rope_kr),
        "qc_post": float(qc_post),
        "qr_post": float(rope_qr),
        "dynamic_quant_qn": float(dynamic_qn),
    }


def persistent_tensor_bytes(problem: ProblemSpec, q: QuantDTypePlan) -> List[Tuple[str, int, int]]:
    # priority: lower number = keep first in L2
    items = [
        ("weightUk", bytes_weight_uk(problem, q), 1),
        ("weightDq", bytes_weight_dq(problem, q), 2),
        ("weightUqQr", bytes_weight_uqqr(problem, q), 3),
        ("weightDkvKr", bytes_weight_dkvkr(problem, q), 4),
        ("rmsnormGammaCq", bytes_gamma_cq(problem), 5),
        ("rmsnormGammaCkv", bytes_gamma_ckv(problem), 5),
        ("deqScaleQcQrW", bytes_deq_scale_qcqrw(problem, q), 5),
        ("dequantScaleWDq", bytes_dequant_scale_wdq(problem, q), 5),
        ("dequantScaleWDkvKr", bytes_dequant_scale_wdkvkr(problem, q), 5),
        ("smoothScaleCq", bytes_smooth_scale_cq(problem, q), 5),
        ("quantScaleCkv", bytes_quant_scale_ckv(problem, q), 5),
        ("quantScaleCkr", bytes_quant_scale_ckr(problem, q), 5),
    ]
    return [(n, b, p) for (n, b, p) in items if b > 0]


def choose_warm_l2_resident(hw: HardwareProfile, persistent: List[Tuple[str, int, int]]) -> Tuple[Dict[str, bool], List[str]]:
    notes: List[str] = []
    budget = hw.retained_l2_budget
    used = 0
    chosen: Dict[str, bool] = {}
    for name, size, _prio in sorted(persistent, key=lambda x: (x[2], x[0])):
        if used + size <= budget:
            chosen[name] = True
            used += size
        else:
            chosen[name] = False
            notes.append(f"L2 warm retention evicts {name} ({size} B) due to budget")
    notes.append(f"Warm L2 retained bytes: {used} / {budget}")
    return chosen, notes


def make_stage_cost(name: str, engine: str, flops: float, traffic: TrafficBreakdown, hw: HardwareProfile, notes: Optional[List[str]] = None) -> StageCost:
    t_compute = 0.0
    if flops > 0:
        peak = hw.effective_cube_ops if engine == "cube" else hw.effective_vector_ops
        t_compute = flops / peak if peak > 0 else math.inf
    t_hbm = (traffic.hbm_read_bytes + traffic.hbm_write_bytes) / hw.effective_hbm_bw if (traffic.hbm_read_bytes + traffic.hbm_write_bytes) > 0 else 0.0
    t_l2 = (traffic.l2_read_bytes + traffic.l2_write_bytes) / hw.effective_l2_bw if (traffic.l2_read_bytes + traffic.l2_write_bytes) > 0 else 0.0
    t_mem = t_hbm + t_l2
    t_stage = max(t_compute, t_mem)
    bottleneck = classify_bottleneck(t_compute, t_hbm, t_l2)
    return StageCost(
        name=name,
        engine=engine,
        flops=flops,
        traffic=traffic,
        t_compute_s=t_compute,
        t_hbm_mem_s=t_hbm,
        t_l2_mem_s=t_l2,
        t_mem_s=t_mem,
        t_stage_s=t_stage,
        bottleneck=bottleneck,
        notes=notes or [],
    )


def classify_bottleneck(t_compute: float, t_hbm: float, t_l2: float) -> str:
    t_mem = t_hbm + t_l2
    t_stage = max(t_compute, t_mem)
    if t_stage <= 0:
        return "none"
    eps = t_stage * 1e-3
    comp = abs(t_compute - t_stage) <= eps and t_compute > 0
    mem = abs(t_mem - t_stage) <= eps and t_mem > 0
    if comp and mem:
        return "mixed_memory_bound"
    if comp:
        return "compute_bound"
    if mem:
        if t_hbm > t_l2 * 1.2:
            return "hbm_bound"
        if t_l2 > t_hbm * 1.2:
            return "l2_bound"
        return "mixed_memory_bound"
    return "mixed_memory_bound"


def add_read(traffic: TrafficBreakdown, bytes_: int, level: str) -> None:
    if bytes_ <= 0:
        return
    traffic.logical_read_bytes += bytes_
    if level == "HBM":
        traffic.hbm_read_bytes += bytes_
    elif level == "L2":
        traffic.l2_read_bytes += bytes_
    else:
        raise ModelError(f"Unknown memory level {level}")


def add_write(traffic: TrafficBreakdown, bytes_: int, level: str) -> None:
    if bytes_ <= 0:
        return
    traffic.logical_write_bytes += bytes_
    if level == "HBM":
        traffic.hbm_write_bytes += bytes_
    elif level == "L2":
        traffic.l2_write_bytes += bytes_
    else:
        raise ModelError(f"Unknown memory level {level}")


def persistent_level(name: str, scenario_kind: str, ctx: StageModelContext) -> str:
    if scenario_kind == "warm" and ctx.warm_resident.get(name, False):
        return "L2"
    return "HBM"


def build_step_stages(m: int, scenario_kind: str, ctx: StageModelContext, include_once_vector_params: bool) -> List[StageCost]:
    p, b, q, t, hw = ctx.problem, ctx.base, ctx.quant, ctx.tiling, ctx.hw
    c = ctx.coeffs
    mm_flops = matmul_flops_step(m, p)
    vec_flops = vector_flops_step(m, ctx)
    stages: List[StageCost] = []

    # Optional one-time vector parameter copy (small but source-realistic)
    if include_once_vector_params:
        tr = TrafficBreakdown()
        for n, size in [
            ("rmsnormGammaCq", bytes_gamma_cq(p)),
            ("rmsnormGammaCkv", bytes_gamma_ckv(p)),
            ("smoothScaleCq", bytes_smooth_scale_cq(p, q)),
            ("quantScaleCkv", bytes_quant_scale_ckv(p, q)),
            ("quantScaleCkr", bytes_quant_scale_ckr(p, q)),
        ]:
            if size > 0:
                add_read(tr, size, persistent_level(n, scenario_kind, ctx))
        if tr.logical_read_bytes > 0:
            stages.append(make_stage_cost("vector_param_once", "vector", 0.0, tr, hw, ["AIV CopyGlobalParams() one-time param copy"] ))

    # mm_cq
    tr = TrafficBreakdown()
    add_read(tr, bytes_token_x(m, p, q), "HBM")
    add_read(tr, bytes_weight_dq(p, q), persistent_level("weightDq", scenario_kind, ctx))
    if q.mm_input_type == "fp8_e4m3":
        add_read(tr, bytes_dequant_scale_x(m, p, q), "HBM")
        add_read(tr, bytes_dequant_scale_wdq(p, q), persistent_level("dequantScaleWDq", scenario_kind, ctx))
    add_write(tr, bytes_workspace_mm_cq(m, p, q), "L2")
    stages.append(make_stage_cost("mm_cq", "cube", mm_flops["mm_cq"], tr, hw))

    # rmsnorm_cq
    tr = TrafficBreakdown()
    add_read(tr, bytes_workspace_mm_cq(m, p, q), "L2")
    add_read(tr, bytes_gamma_cq(p), persistent_level("rmsnormGammaCq", scenario_kind, ctx))
    if q.mm_input_type == "int8":
        add_read(tr, bytes_dequant_scale_x(m, p, q), "HBM")
        add_read(tr, bytes_dequant_scale_wdq(p, q), persistent_level("dequantScaleWDq", scenario_kind, ctx))
    if q.mm_qcqr_input_type == "int8":
        add_read(tr, bytes_smooth_scale_cq(p, q), persistent_level("smoothScaleCq", scenario_kind, ctx))
    if p.query_norm_flag:
        add_write(tr, bytes_query_norm_out(m, p, q), "HBM")
        qnorm_scale_out = bytes_dequant_scale_qnorm_out(m, p, q)
        if qnorm_scale_out:
            add_write(tr, qnorm_scale_out, "HBM")
    else:
        add_write(tr, bytes_workspace_rmsnorm_cq(m, p, q), "L2")
    stages.append(make_stage_cost("rmsnorm_cq", "vector", vec_flops["rmsnorm_cq"], tr, hw))

    # mm_ckvkr
    tr = TrafficBreakdown()
    # tokenX reused immediately after mm_cq, model from L2
    add_read(tr, bytes_token_x(m, p, q), "L2")
    add_read(tr, bytes_weight_dkvkr(p, q), persistent_level("weightDkvKr", scenario_kind, ctx))
    if q.mm_input_type == "fp8_e4m3":
        add_read(tr, bytes_dequant_scale_x(m, p, q), "L2")
        add_read(tr, bytes_dequant_scale_wdkvkr(p, q), persistent_level("dequantScaleWDkvKr", scenario_kind, ctx))
    add_write(tr, bytes_workspace_mm_ckvkr(m, p, q), "L2")
    stages.append(make_stage_cost("mm_ckvkr", "cube", mm_flops["mm_ckvkr"], tr, hw))

    # ckvkr_post (RmsNormCkv + RopeKr + Scatter kv/kr)
    tr = TrafficBreakdown()
    add_read(tr, bytes_workspace_mm_ckvkr(m, p, q), "L2")
    add_read(tr, bytes_gamma_ckv(p), persistent_level("rmsnormGammaCkv", scenario_kind, ctx))
    add_read(tr, bytes_rope_sincos(m, p), "HBM")
    ci = bytes_cache_index(m, p)
    if ci:
        add_read(tr, ci, "HBM")
    if q.mm_input_type == "int8":
        add_read(tr, bytes_dequant_scale_x(m, p, q), "L2")
        add_read(tr, bytes_dequant_scale_wdkvkr(p, q), persistent_level("dequantScaleWDkvKr", scenario_kind, ctx))
    qckv = bytes_quant_scale_ckv(p, q)
    if qckv:
        add_read(tr, qckv, persistent_level("quantScaleCkv", scenario_kind, ctx))
    qckr = bytes_quant_scale_ckr(p, q)
    if qckr:
        add_read(tr, qckr, persistent_level("quantScaleCkr", scenario_kind, ctx))
    add_write(tr, bytes_kv_cache_out(m, p, q), "HBM")
    add_write(tr, bytes_kr_cache_out(m, p, q), "HBM")
    stages.append(make_stage_cost("ckvkr_post", "vector", vec_flops["ckvkr_post"], tr, hw))

    # mm_qcqr
    tr = TrafficBreakdown()
    add_read(tr, bytes_workspace_rmsnorm_cq(m, p, q) if not p.query_norm_flag else bytes_query_norm_out(m, p, q), "L2")
    add_read(tr, bytes_weight_uqqr(p, q), persistent_level("weightUqQr", scenario_kind, ctx))
    if q.mm_qcqr_input_type == "fp8_e4m3":
        add_read(tr, bytes_deq_scale_qcqrw(p, q), persistent_level("deqScaleQcQrW", scenario_kind, ctx))
    add_write(tr, bytes_workspace_mm_qcqr(m, p, q), "L2")
    stages.append(make_stage_cost("mm_qcqr", "cube", mm_flops["mm_qcqr"], tr, hw))

    has_qc_post = q.mm_qcqr_input_type in ("int8", "fp8_e4m3")
    need_qn_dynamic_quant = ((q.mm_input_type == "int8" and q.kv_cache_type == "int8") or (q.mm_input_type == "fp8_e4m3" and q.kv_cache_type == "fp8_e4m3")) and (not q.is_pertile)

    if has_qc_post:
        tr = TrafficBreakdown()
        add_read(tr, bytes_workspace_mm_qc_slice(m, p, q), "L2")
        if q.mm_qcqr_input_type == "int8":
            add_read(tr, bytes_deq_scale_qcqrw(p, q), persistent_level("deqScaleQcQrW", scenario_kind, ctx))
        add_write(tr, bytes_workspace_mm_qcqr_dequant(m, p, q), "L2")
        note = []
        if t.enableDequantOpt:
            note.append("Split-N dequant/cast path modeled as aggregated qc_post")
        stages.append(make_stage_cost("qc_post", "vector", vec_flops["qc_post"], tr, hw, note))

    # qr_post (RopeQr). Rope sin/cos modeled here once per step.
    tr = TrafficBreakdown()
    add_read(tr, bytes_workspace_mm_qr_slice(m, p, q), "L2")
    # Sin/cos GM traffic is attributed in ckvkr_post (CopyInSinCos happens once before both RopeKr/RopeQr).
    add_write(tr, bytes_query_rope_out(m, p, q), "HBM")
    stages.append(make_stage_cost("qr_post", "vector", vec_flops["qr_post"], tr, hw))

    # mm_qn
    tr = TrafficBreakdown()
    add_read(tr, bytes_workspace_mm_qcqr_dequant(m, p, q) if has_qc_post else m * p.N * p.D * dtype_size("bf16"), "L2")
    add_read(tr, bytes_weight_uk(p, q), persistent_level("weightUk", scenario_kind, ctx))
    if need_qn_dynamic_quant:
        add_write(tr, bytes_workspace_mm_qn(m, p, q), "L2")
    else:
        add_write(tr, bytes_query_out(m, p, q), "HBM")
    stages.append(make_stage_cost("mm_qn", "cube", mm_flops["mm_qn"], tr, hw, ["WeightUk preload modeled as part of mm_qn weight traffic"]))

    if need_qn_dynamic_quant:
        tr = TrafficBreakdown()
        add_read(tr, bytes_workspace_mm_qn(m, p, q), "L2")
        if (not t.enableDequantOpt):
            # source sync may wait for RopeQr completion before dynamic quant in this branch.
            pass
        add_write(tr, bytes_query_out(m, p, q), "HBM")
        add_write(tr, bytes_dequant_scale_qnope_out(m, p, q), "HBM")
        stages.append(make_stage_cost("dynamic_quant_qn", "vector", vec_flops["dynamic_quant_qn"], tr, hw))

    return stages


def schedule_stages(stages: Sequence[StageCost], q: QuantDTypePlan, tiling: TilingSpec) -> Tuple[float, Dict[str, Dict[str, float]], List[str]]:
    stage_map = {s.name: s for s in stages}
    deps: Dict[str, List[str]] = {s.name: [] for s in stages}

    def edge(a: str, b: str) -> None:
        if a in deps and b in deps:
            deps[b].append(a)

    edge("mm_cq", "rmsnorm_cq")
    edge("rmsnorm_cq", "mm_qcqr")
    edge("mm_ckvkr", "ckvkr_post")
    edge("mm_qcqr", "qc_post")
    edge("mm_qcqr", "qr_post")
    if "qc_post" in deps:
        edge("qc_post", "mm_qn")
    else:
        edge("mm_qcqr", "mm_qn")
    if "dynamic_quant_qn" in deps:
        edge("mm_qn", "dynamic_quant_qn")
        if not tiling.enableDequantOpt:
            edge("qr_post", "dynamic_quant_qn")

    order: List[str] = []
    indeg = {k: len(v) for k, v in deps.items()}
    ready = sorted([k for k, d in indeg.items() if d == 0])
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in deps:
            if n in deps[m]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
                    ready.sort()
    if len(order) != len(stages):
        raise ModelError("Stage DAG cycle detected")

    resource_free = {"cube": 0.0, "vector": 0.0}
    timing: Dict[str, Dict[str, float]] = {}
    for name in order:
        s = stage_map[name]
        dep_end = max((timing[d]["end_s"] for d in deps[name]), default=0.0)
        start = max(dep_end, resource_free[s.engine])
        end = start + s.t_stage_s
        timing[name] = {"start_s": start, "end_s": end, "duration_s": s.t_stage_s}
        resource_free[s.engine] = end

    total = max((v["end_s"] for v in timing.values()), default=0.0)
    crit = critical_path_from_schedule(order, deps, timing)
    return total, timing, crit


def critical_path_from_schedule(order: Sequence[str], deps: Mapping[str, List[str]], timing: Mapping[str, Mapping[str, float]]) -> List[str]:
    if not order:
        return []
    # Longest path in DAG using scheduled durations (dependency-only critical path).
    dp: Dict[str, float] = {}
    prev: Dict[str, Optional[str]] = {}
    for n in order:
        dur = float(timing[n]["duration_s"])
        if not deps[n]:
            dp[n] = dur
            prev[n] = None
        else:
            best = max(deps[n], key=lambda d: dp[d])
            dp[n] = dp[best] + dur
            prev[n] = best
    end_node = max(order, key=lambda n: dp[n])
    path: List[str] = []
    cur: Optional[str] = end_node
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def aggregate_traffic(stages: Iterable[StageCost]) -> TrafficBreakdown:
    t = TrafficBreakdown()
    for s in stages:
        t.add(s.traffic)
    return t


def scenario_kernel_estimate(name: str, ctx: StageModelContext, scenario_kind: str) -> ScenarioEstimate:
    T = ctx.problem.T
    step_bs = ctx.tiling.stepBatchSize
    num_steps = ceil_div(T, step_bs)
    tail_bs = T - (num_steps - 1) * step_bs if num_steps > 0 else 0
    full_bs = step_bs

    full_stages = build_step_stages(full_bs, scenario_kind, ctx, include_once_vector_params=True)
    full_time, full_sched, full_cp = schedule_stages(full_stages, ctx.quant, ctx.tiling)

    if tail_bs == full_bs:
        tail_stages = [s for s in full_stages]
        tail_time = full_time
        tail_sched = dict(full_sched)
        tail_cp = list(full_cp)
    else:
        tail_stages = build_step_stages(tail_bs, scenario_kind, ctx, include_once_vector_params=True)
        tail_time, tail_sched, tail_cp = schedule_stages(tail_stages, ctx.quant, ctx.tiling)

    if num_steps <= 1:
        total_time = tail_time
        traffic_total = aggregate_traffic(tail_stages)
    else:
        mid_count = num_steps - 2
        # One-time vector_param_once is included on every per-step build; remove for replicated middle/full steps.
        full_no_once = [s for s in build_step_stages(full_bs, scenario_kind, ctx, include_once_vector_params=False)]
        full_no_once_time, _, _ = schedule_stages(full_no_once, ctx.quant, ctx.tiling)
        tail_no_once = build_step_stages(tail_bs, scenario_kind, ctx, include_once_vector_params=False)
        tail_no_once_time, _, _ = schedule_stages(tail_no_once, ctx.quant, ctx.tiling)

        total_time = full_time
        if mid_count > 0:
            total_time += mid_count * full_no_once_time
        total_time += tail_no_once_time

        traffic_total = aggregate_traffic(full_stages)
        if mid_count > 0:
            traffic_total.add(aggregate_traffic(full_no_once).scaled(mid_count))
        traffic_total.add(aggregate_traffic(tail_no_once))

    return ScenarioEstimate(
        name=name,
        step_batch_size=step_bs,
        num_steps=num_steps,
        tail_step_batch_size=tail_bs,
        full_step_time_s=full_time,
        tail_step_time_s=tail_time,
        total_time_s=total_time,
        stages_full_step=full_stages,
        stages_tail_step=tail_stages,
        traffic_total=traffic_total,
        critical_path_full_step=full_cp,
        critical_path_tail_step=tail_cp,
        schedule_full_step=full_sched,
        schedule_tail_step=tail_sched,
        notes=[],
    )


def mixed_cold_then_warm_estimate(ctx: StageModelContext, cold_step: ScenarioEstimate, warm_step: ScenarioEstimate) -> ScenarioEstimate:
    T = ctx.problem.T
    step_bs = ctx.tiling.stepBatchSize
    num_steps = ceil_div(T, step_bs)
    tail_bs = T - (num_steps - 1) * step_bs if num_steps > 0 else 0

    if num_steps <= 1:
        total_time = cold_step.tail_step_time_s
        traffic_total = cold_step.traffic_total
    else:
        # cold first full step + warm middle full steps + warm tail step (without once-cost duplication)
        full_warm_no_once = [s for s in build_step_stages(step_bs, "warm", ctx, include_once_vector_params=False)]
        full_warm_no_once_time, _, _ = schedule_stages(full_warm_no_once, ctx.quant, ctx.tiling)
        tail_warm_no_once_stages = full_warm_no_once if tail_bs == step_bs else build_step_stages(tail_bs, "warm", ctx, include_once_vector_params=False)
        tail_warm_no_once_time, tail_sched, tail_cp = schedule_stages(tail_warm_no_once_stages, ctx.quant, ctx.tiling)

        mid_count = num_steps - 2
        total_time = cold_step.full_step_time_s + max(mid_count, 0) * full_warm_no_once_time + tail_warm_no_once_time
        traffic_total = aggregate_traffic(cold_step.stages_full_step)
        if mid_count > 0:
            traffic_total.add(aggregate_traffic(full_warm_no_once).scaled(mid_count))
        traffic_total.add(aggregate_traffic(tail_warm_no_once_stages))

        return ScenarioEstimate(
            name="cold_then_warm",
            step_batch_size=step_bs,
            num_steps=num_steps,
            tail_step_batch_size=tail_bs,
            full_step_time_s=cold_step.full_step_time_s,
            tail_step_time_s=tail_warm_no_once_time,
            total_time_s=total_time,
            stages_full_step=cold_step.stages_full_step,
            stages_tail_step=tail_warm_no_once_stages,
            traffic_total=traffic_total,
            critical_path_full_step=cold_step.critical_path_full_step,
            critical_path_tail_step=tail_cp,
            schedule_full_step=cold_step.schedule_full_step,
            schedule_tail_step=tail_sched,
            notes=["First step cold (persistent tensors from HBM), remaining steps warm (L2-retained set)"]
        )

    return ScenarioEstimate(
        name="cold_then_warm",
        step_batch_size=step_bs,
        num_steps=num_steps,
        tail_step_batch_size=tail_bs,
        full_step_time_s=cold_step.full_step_time_s,
        tail_step_time_s=cold_step.tail_step_time_s,
        total_time_s=total_time,
        stages_full_step=cold_step.stages_full_step,
        stages_tail_step=cold_step.stages_tail_step,
        traffic_total=traffic_total,
        critical_path_full_step=cold_step.critical_path_full_step,
        critical_path_tail_step=cold_step.critical_path_tail_step,
        schedule_full_step=cold_step.schedule_full_step,
        schedule_tail_step=cold_step.schedule_tail_step,
        notes=["Single-step case; cold_then_warm equals cold step"]
    )


def vector_flops_total_approx(problem: ProblemSpec, ctx: StageModelContext) -> Dict[str, Any]:
    step_bs = ctx.tiling.stepBatchSize
    num_steps = ceil_div(problem.T, step_bs)
    tail = problem.T - (num_steps - 1) * step_bs if num_steps else 0
    full = vector_flops_step(step_bs, ctx)
    total = {k: 0.0 for k in full}
    if num_steps == 0:
        pass
    elif num_steps == 1:
        total = vector_flops_step(tail, ctx)
    else:
        for k, v in full.items():
            total[k] += v * (num_steps - 1)
        tail_v = vector_flops_step(tail, ctx)
        for k, v in tail_v.items():
            total[k] += v
    total_sum = float(sum(total.values()))
    return {
        "per_full_step": full,
        "per_tail_step": vector_flops_step(tail, ctx),
        "total": total,
        "total_sum": total_sum,
    }


def matmul_flops_report(problem: ProblemSpec, ctx: StageModelContext) -> Dict[str, Any]:
    exact = matmul_flops_exact(problem)
    total = sum(exact.values())
    step_bs = ctx.tiling.stepBatchSize
    num_steps = ceil_div(problem.T, step_bs)
    tail = problem.T - (num_steps - 1) * step_bs if num_steps else 0
    per_full = matmul_flops_step(step_bs, problem)
    per_tail = matmul_flops_step(tail, problem)
    shares = {k: (v / total if total else 0.0) for k, v in exact.items()}
    return {
        "total": exact,
        "total_cube_flops": float(total),
        "per_full_step": per_full,
        "per_tail_step": per_tail,
        "shares": shares,
        "num_steps": num_steps,
        "step_batch_size": step_bs,
        "tail_step_batch_size": tail,
    }


def build_report(problem: ProblemSpec, base: BaseParamsLike, quant: QuantDTypePlan, tiling: TilingSpec, hw: HardwareProfile, coeffs: ModelCoefficients) -> KernelEstimateReport:
    persistent = persistent_tensor_bytes(problem, quant)
    warm_resident, l2_notes = choose_warm_l2_resident(hw, persistent)
    ctx = StageModelContext(problem, base, quant, tiling, hw, coeffs, warm_resident)

    cold_step = scenario_kernel_estimate("cold_all_steps", ctx, "cold")
    warm_step = scenario_kernel_estimate("warm_all_steps", ctx, "warm")
    cold_mixed = mixed_cold_then_warm_estimate(ctx, cold_step, warm_step)

    matmul_report = matmul_flops_report(problem, ctx)
    vector_report = vector_flops_total_approx(problem, ctx)
    fused_flops_total = matmul_report["total_cube_flops"] + vector_report["total_sum"]

    notes = list(tiling.notes) + l2_notes
    if problem.Dtile < problem.Hckv:
        notes.append("Dtile < Hckv: cache write bytes use physical Dtile while compute FLOPs use logical Hckv")
    if tiling.splitMFlag != 0:
        notes.append("splitMFlag was forced to 0 by model assumptions")

    assumptions = {
        "split_n_only": True,
        "split_m_supported_in_model": False,
        "mac_to_flops": 2,
        "vector_flops_are_approximate": True,
        "hbm_l2_model": "heuristic_source_level",
        "immediate_workspace_reuse_treated_as_l2": True,
        "persistent_l2_retention_policy": {
            "budget_bytes": hw.retained_l2_budget,
            "reserve_bytes": hw.l2_reserve_bytes,
            "reserve_ratio": hw.l2_reserve_ratio,
            "pin_order": ["weightUk", "weightDq", "weightUqQr", "weightDkvKr", "gamma/scales/static"],
        },
        "coefficients": coeffs.to_dict(),
        "hardware_defaults_overrideable": True,
    }

    inputs = {
        "problem": asdict(problem),
        "base_params": asdict(base),
        "hardware": asdict(hw),
        "quant_plan": asdict(quant),
    }

    return KernelEstimateReport(
        report_version=REPORT_VERSION,
        inputs=inputs,
        assumptions=assumptions,
        tiling=asdict(tiling),
        matmul_flops_exact={**matmul_report, "fused_kernel_total_flops_estimate": fused_flops_total},
        vector_flops_approx=vector_report,
        traffic_cold={
            "cold_then_warm_kernel_total": cold_mixed.traffic_total.to_dict(),
            "cold_full_step": aggregate_traffic(cold_mixed.stages_full_step).to_dict(),
            "warm_tail_or_step": aggregate_traffic(cold_mixed.stages_tail_step).to_dict(),
        },
        traffic_warm={
            "warm_kernel_total": warm_step.traffic_total.to_dict(),
            "warm_full_step": aggregate_traffic(warm_step.stages_full_step).to_dict(),
            "warm_tail_step": aggregate_traffic(warm_step.stages_tail_step).to_dict(),
        },
        timing_cold={
            "cold_then_warm": cold_mixed.to_dict(),
            "cold_all_steps_reference": cold_step.to_dict(),
        },
        timing_warm={
            "warm_all_steps": warm_step.to_dict(),
        },
        critical_path_nodes={
            "cold_then_warm_full_step": cold_mixed.critical_path_full_step,
            "cold_then_warm_tail_step": cold_mixed.critical_path_tail_step,
            "warm_full_step": warm_step.critical_path_full_step,
            "warm_tail_step": warm_step.critical_path_tail_step,
        },
        notes=notes,
    )


def fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if abs(x) < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"


def fmt_time_s(x: float) -> str:
    if x < 1e-6:
        return f"{x * 1e9:.2f} ns"
    if x < 1e-3:
        return f"{x * 1e6:.2f} us"
    if x < 1.0:
        return f"{x * 1e3:.2f} ms"
    return f"{x:.4f} s"


def fmt_flops(x: float) -> str:
    units = [(1e15, "PF"), (1e12, "TF"), (1e9, "GF"), (1e6, "MF"), (1e3, "KF")]
    for scale, suffix in units:
        if abs(x) >= scale:
            return f"{x / scale:.3f} {suffix}LOPs"
    return f"{x:.0f} FLOPs"


def stage_rows(stages: Sequence[StageCost]) -> List[List[str]]:
    rows: List[List[str]] = []
    for s in stages:
        rows.append([
            s.name,
            s.engine,
            fmt_flops(s.flops),
            fmt_bytes(s.traffic.hbm_read_bytes + s.traffic.hbm_write_bytes),
            fmt_bytes(s.traffic.l2_read_bytes + s.traffic.l2_write_bytes),
            fmt_time_s(s.t_stage_s),
            s.bottleneck,
        ])
    return rows


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def print_report_table(report: Any) -> None:
    d = report.to_dict() if hasattr(report, "to_dict") else report
    p = d["inputs"]["problem"]
    q = d["inputs"]["quant_plan"]
    hw = d["inputs"]["hardware"]
    tiling = d["tiling"]
    cold = d.get("timing_cold", {}).get("cold_then_warm")
    warm = d.get("timing_warm", {}).get("warm_all_steps")
    matmuls = d["matmul_flops_exact"]
    vec = d["vector_flops_approx"]

    print("== MlaPrologV3 Theoretical Performance Model ==")
    print(f"Report version: {d['report_version']}")
    print(f"Problem: T={p['T']} He={p['He']} Hcq={p['Hcq']} N={p['N']} D={p['D']} Dr={p['Dr']} Hckv={p['Hckv']} Nkv={p['Nkv']} Dtile={p['Dtile']}")
    print(f"Quant mode: {q['quant_mode']} ({q['label']}) | Cache mode: {p['cache_mode']} | QueryNormFlag: {p['query_norm_flag']}")
    print(f"Hardware: AIC={hw['aic_num']} AIV={hw['aiv_num']} L2={fmt_bytes(hw['l2_size_bytes'])} | effective HBM BW={hw['hbm_bw_gbps']*hw['hbm_eff']:.1f} GB/s | effective L2 BW={hw['l2_bw_gbps']*hw['l2_eff']:.1f} GB/s")
    print()

    print("-- Tiling --")
    tiling_rows = [
        ["stepBatchSize", str(tiling["stepBatchSize"])],
        ["vectorBlockNum", str(tiling["vectorBlockNum"])],
        ["mm1/mm2/mm3/mm4 BlockNum", f"{tiling['mm1BlockNum']}/{tiling['mm2BlockNum']}/{tiling['mm3BlockNum']}/{tiling['mm4BlockNum']}"],
        ["mm1/mm2/mm3 SingleCoreN", f"{tiling['mm1SingleCoreN']}/{tiling['mm2SingleCoreN']}/{tiling['mm3SingleCoreN']}"],
        ["mm4SingleCoreBatch", str(tiling["mm4SingleCoreBatch"])],
        ["enableDequantOpt / enableGroupComputeOpt", f"{tiling['enableDequantOpt']} / {tiling['enableGroupComputeOpt']}"],
        ["cv_ratio", str(tiling["cv_ratio"])],
    ]
    print_table(["Key", "Value"], tiling_rows)
    print()

    print("-- Exact Matmul FLOPs --")
    mm_rows = []
    total_cube = matmuls["total_cube_flops"]
    for name, fl in matmuls["total"].items():
        share = 100.0 * matmuls["shares"][name]
        mm_rows.append([name, fmt_flops(fl), f"{share:.2f}%"])
    mm_rows.append(["Cube Total", fmt_flops(total_cube), "100.00%"])
    mm_rows.append(["Fused Total (cube exact + vector approx)", fmt_flops(matmuls["fused_kernel_total_flops_estimate"]), "-"])
    print_table(["Matmul", "FLOPs", "Share"], mm_rows)
    print()

    print("-- Vector Approx FLOPs (Total) --")
    vec_rows = [[k, fmt_flops(v)] for k, v in vec["total"].items() if v > 0]
    vec_rows.append(["Vector Total Approx", fmt_flops(vec["total_sum"])])
    print_table(["Stage", "Approx FLOPs"], vec_rows)
    print()

    print("-- Timing Summary --")
    timing_rows: List[List[str]] = []
    if cold:
        timing_rows.append(["Cold->Warm kernel total", fmt_time_s(cold["total_time_s"]), f"{(p['T']/cold['total_time_s']) if cold['total_time_s']>0 else float('inf'):.2f} tok/s"])
        timing_rows.append(["Cold full step", fmt_time_s(cold["full_step_time_s"]), "-"])
        timing_rows.append(["Tail step (cold->warm view)", fmt_time_s(cold["tail_step_time_s"]), "-"])
    if warm:
        timing_rows.append(["Warm kernel total", fmt_time_s(warm["total_time_s"]), f"{(p['T']/warm['total_time_s']) if warm['total_time_s']>0 else float('inf'):.2f} tok/s"])
        timing_rows.append(["Warm full step", fmt_time_s(warm["full_step_time_s"]), "-"])
        if not cold:
            timing_rows.append(["Tail step (warm view)", fmt_time_s(warm["tail_step_time_s"]), "-"])
    print_table(["Metric", "Time", "Throughput"], timing_rows)
    print()

    print("-- Traffic Summary --")
    traffic_rows: List[List[str]] = []
    tc = d.get("traffic_cold", {}).get("cold_then_warm_kernel_total")
    tw = d.get("traffic_warm", {}).get("warm_kernel_total")
    if tc:
        traffic_rows.extend([
            ["Cold->Warm HBM read", fmt_bytes(tc["hbm_read_bytes"]), "-"],
            ["Cold->Warm HBM write", fmt_bytes(tc["hbm_write_bytes"]), "-"],
            ["Cold->Warm L2 read", fmt_bytes(tc["l2_read_bytes"]), "-"],
            ["Cold->Warm L2 write", fmt_bytes(tc["l2_write_bytes"]), "-"],
        ])
    if tw:
        traffic_rows.extend([
            ["Warm HBM read", fmt_bytes(tw["hbm_read_bytes"]), "-"],
            ["Warm HBM write", fmt_bytes(tw["hbm_write_bytes"]), "-"],
            ["Warm L2 read", fmt_bytes(tw["l2_read_bytes"]), "-"],
            ["Warm L2 write", fmt_bytes(tw["l2_write_bytes"]), "-"],
        ])
    print_table(["Metric", "Bytes", "Note"], traffic_rows)
    print()

    stage_src = None
    stage_title = ""
    if warm:
        stage_src = warm["stages_full_step"]
        stage_title = "Warm Full Step"
    elif cold:
        stage_src = cold["stages_full_step"]
        stage_title = "Cold Full Step"

    if stage_src is not None:
        print(f"-- Stage Costs ({stage_title}) --")
        stage_objs = [StageCost(**{
            "name": s["name"],
            "engine": s["engine"],
            "flops": s["flops"],
            "traffic": TrafficBreakdown(**s["traffic"]),
            "t_compute_s": s["t_compute_s"],
            "t_hbm_mem_s": s["t_hbm_mem_s"],
            "t_l2_mem_s": s["t_l2_mem_s"],
            "t_mem_s": s["t_mem_s"],
            "t_stage_s": s["t_stage_s"],
            "bottleneck": s["bottleneck"],
            "notes": s.get("notes", []),
        }) for s in stage_src]
        print_table(["Stage", "Eng", "FLOPs", "HBM", "L2", "Time", "Bottleneck"], stage_rows(stage_objs))
        print()

    cp_nodes = []
    cp_label = ""
    if warm and d.get("critical_path_nodes", {}).get("warm_full_step"):
        cp_nodes = d["critical_path_nodes"]["warm_full_step"]
        cp_label = "warm full step"
    elif cold and d.get("critical_path_nodes", {}).get("cold_then_warm_full_step"):
        cp_nodes = d["critical_path_nodes"]["cold_then_warm_full_step"]
        cp_label = "cold full step"
    if cp_nodes:
        print(f"Critical path ({cp_label}):", " -> ".join(cp_nodes))
    if d["notes"]:
        print("Notes:")
        for n in d["notes"]:
            print(f"- {n}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MlaPrologV3 theoretical performance model (Ascend 910B-oriented)")
    p.add_argument("--input-mode", choices=["shape", "base-params"], required=True)
    p.add_argument("--input-json", required=True, help="Path to JSON input")
    p.add_argument("--quant-mode", type=int, help="Override quant mode [0..9]")
    p.add_argument("--cache-mode", choices=sorted(CACHE_MODES))
    p.add_argument("--query-norm-flag", type=int, choices=[0, 1])
    p.add_argument("--l2-mode", choices=["cold", "warm", "both"], default="both")
    p.add_argument("--report", choices=["table", "json", "both"], default="both")
    p.add_argument("--json-out", help="Optional path to write JSON report")
    p.add_argument("--cube-peak-tflops", type=float)
    p.add_argument("--vector-peak-tflops", type=float)
    p.add_argument("--hbm-bw-gbps", type=float)
    p.add_argument("--l2-bw-gbps", type=float)
    p.add_argument("--cube-eff", type=float)
    p.add_argument("--vector-eff", type=float)
    p.add_argument("--hbm-eff", type=float)
    p.add_argument("--l2-eff", type=float)
    p.add_argument("--l2-reserve-bytes", type=int)
    p.add_argument("--l2-reserve-ratio", type=float)
    return p


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ModelError("Input JSON root must be an object")
    return data


def apply_l2_mode_filter(report_dict: Dict[str, Any], l2_mode: str) -> Dict[str, Any]:
    if l2_mode == "both":
        return report_dict
    out = dict(report_dict)
    if l2_mode == "cold":
        out["traffic_warm"] = {}
        out["timing_warm"] = {}
        out["critical_path_nodes"] = {k: v for k, v in out.get("critical_path_nodes", {}).items() if "warm" not in k or "cold_then_warm" in k}
    elif l2_mode == "warm":
        out["traffic_cold"] = {}
        out["timing_cold"] = {}
        out["critical_path_nodes"] = {k: v for k, v in out.get("critical_path_nodes", {}).items() if "cold" not in k}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        data = load_json(args.input_json)
        hw = build_hw_from_args(args, data)
        coeffs = build_coeffs(data)
        if args.input_mode == "shape":
            problem = parse_problem_from_shape(data, args)
            q = quant_plan(problem.quant_mode)
            tiling, base = derive_tiling(problem, hw, q)
        else:
            problem, base = parse_base_params_mode(data, args)
            q = quant_plan(problem.quant_mode)
            tiling = TilingSpec(
                splitMFlag=0,
                stepBatchSize=base.stepBatchSize,
                vectorBlockNum=base.vectorBlockNum,
                stepNumHeadDequant=base.stepNumHeadDequant,
                mm1BlockNum=base.mm1BlockNum,
                mm2BlockNum=base.mm2BlockNum,
                mm3BlockNum=base.mm3BlockNum,
                mm4BlockNum=base.mm4BlockNum,
                mm1SingleCoreN=base.mm1SingleCoreN,
                mm2SingleCoreN=base.mm2SingleCoreN,
                mm3SingleCoreN=base.mm3SingleCoreN,
                mm4SingleCoreBatch=base.mm4SingleCoreBatch,
                enableDequantOpt=((q.mm_qcqr_input_type == "int8" and problem.N >= GROUP_COMPUTE_N_SIZE) or q.mm_qcqr_input_type == "fp8_e4m3"),
                enableGroupComputeOpt=False,
                aic_num=hw.aic_num,
                aiv_num=hw.aiv_num,
                cv_ratio=hw.cv_ratio,
                notes=["base-params mode: tiling fields taken from input; splitM modeled as disabled"],
            )
            if base.tokenSize != problem.T:
                raise ModelError("base_params.tokenSize must match derived ProblemSpec.T")

        report = build_report(problem, base, q, tiling, hw, coeffs)
        report_dict = apply_l2_mode_filter(report.to_dict(), args.l2_mode)

        if args.report in ("table", "both"):
            print_report_table(report_dict)
            if args.report == "both":
                print()
        if args.report in ("json", "both"):
            json_text = json.dumps(report_dict, indent=2, ensure_ascii=False)
            if args.report == "json":
                print(json_text)
            else:
                print(json_text)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    except ModelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
