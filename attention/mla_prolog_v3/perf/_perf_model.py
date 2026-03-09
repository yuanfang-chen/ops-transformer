from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


KERNEL_MAX_UB_BYTES = 192 * 1024
KERNEL_L1_A_BYTES = 128 * 1024
KERNEL_L1_B_BYTES = 128 * 1024
KERNEL_L0A_BYTES = 32 * 1024
KERNEL_L0B_BYTES = 32 * 1024
KERNEL_L0C_BYTES = 64 * 1024

BYTE_BLOCK = 32
ALIGN_BLOCK_SIZE = 32
BLOCK_CUBE_SIZE = 16
FP8_E4M3_BLOCK_SIZE = 32
QC_CORE_NUM = 8
QR_CORE_NUM = 4
INT8_AFULLLOAD_MAX_MSIZE = 64
BF16_AFULLLOAD_MAX_MSIZE = 32

WEIGHT_QUANT_NO = 0
WEIGHT_QUANT_PARTIAL = 1
WEIGHT_QUANT_FULL = 2
WEIGHT_QUANT_MXFP8 = 3

KV_QUANT_NO = 0
KV_QUANT_PER_TENSOR = 1
KV_QUANT_PER_CHANNEL = 2
KV_QUANT_PER_TILE = 3

QUERY_QUANT_NO = 0
QUERY_QUANT_PER_TOKEN_HEAD = 1

CACHE_MODES = {
    "BSND",
    "TND",
    "PA_BSND",
    "PA_NZ",
    "PA_BLK_BSND",
    "PA_BLK_NZ",
}

ACTUAL_SEQ_DISABLED = "DISABLED"
ACTUAL_SEQ_EN_Q_LEN = "EN_Q_LEN"

DTYPE_SIZES = {
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
    "fp8_scale": 1,
    "fp32": 4,
    "int32": 4,
    "int64": 8,
}

ENGINE_CUBE = "cube"
ENGINE_VECTOR = "vector"
ENGINE_DMA = "dma"

SCOPE_ONCE = "once_per_kernel"
SCOPE_STEP = "per_step"

BOUND_CUBE = "cube_compute"
BOUND_VECTOR = "vector_compute"
BOUND_GM = "gm_bandwidth"
BOUND_L2 = "l2_bandwidth"
BOUND_SYNC = "sync_wait"

QUANT_MODE_ORDER = {
    "NO_QUANT": 0,
    "PARTIAL_QUANT_KV_NO_QUANT": 1,
    "PARTIAL_QUANT_KV_QUANT_PER_CHANNEL": 2,
    "FULL_QUANT_KV_NO_QUANT": 3,
    "FULL_QUANT_KV_QUANT_PER_TENSOR": 4,
    "PARTIAL_QUANT_KV_QUANT_PER_TILE": 5,
    "FULL_QUANT_KV_QUANT_PER_TILE": 6,
    "MXFP8_FULL_QUANT_KV_NO_QUANT": 7,
    "MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR": 8,
    "MXFP8_FULL_QUANT_KV_QUANT_PER_TILE": 9,
}

QMODE_TO_LABEL = {value: key for key, value in QUANT_MODE_ORDER.items()}

_PATH_TABLE: Dict[str, Dict[str, Any]] = {
    "NO_QUANT": {
        "mm_input_dtype": "bf16",
        "mm_qcqr_input_dtype": "bf16",
        "mm_cq_output_dtype": "bf16",
        "mm_ckvkr_output_dtype": "bf16",
        "mm_qcqr_output_dtype": "bf16",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "bf16",
        "rmsnorm_ckv_output_dtype": "bf16",
        "kv_cache_dtype": "bf16",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": None,
        "is_pertile": False,
    },
    "PARTIAL_QUANT_KV_NO_QUANT": {
        "mm_input_dtype": "bf16",
        "mm_qcqr_input_dtype": "int8",
        "mm_cq_output_dtype": "bf16",
        "mm_ckvkr_output_dtype": "bf16",
        "mm_qcqr_output_dtype": "bf16",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "int8",
        "rmsnorm_ckv_output_dtype": "bf16",
        "kv_cache_dtype": "bf16",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp32",
        "is_pertile": False,
    },
    "PARTIAL_QUANT_KV_QUANT_PER_CHANNEL": {
        "mm_input_dtype": "bf16",
        "mm_qcqr_input_dtype": "int8",
        "mm_cq_output_dtype": "bf16",
        "mm_ckvkr_output_dtype": "bf16",
        "mm_qcqr_output_dtype": "bf16",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "int8",
        "rmsnorm_ckv_output_dtype": "int8",
        "kv_cache_dtype": "int8",
        "kr_cache_dtype": "int8",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp32",
        "is_pertile": False,
    },
    "FULL_QUANT_KV_NO_QUANT": {
        "mm_input_dtype": "int8",
        "mm_qcqr_input_dtype": "int8",
        "mm_cq_output_dtype": "int32",
        "mm_ckvkr_output_dtype": "int32",
        "mm_qcqr_output_dtype": "int32",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "int8",
        "rmsnorm_ckv_output_dtype": "bf16",
        "kv_cache_dtype": "bf16",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp32",
        "is_pertile": False,
    },
    "FULL_QUANT_KV_QUANT_PER_TENSOR": {
        "mm_input_dtype": "int8",
        "mm_qcqr_input_dtype": "int8",
        "mm_cq_output_dtype": "int32",
        "mm_ckvkr_output_dtype": "int32",
        "mm_qcqr_output_dtype": "int32",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "int8",
        "rmsnorm_ckv_output_dtype": "int8",
        "kv_cache_dtype": "int8",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "int8",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": "fp32",
        "dequant_scale_qnorm_dtype": "fp32",
        "is_pertile": False,
    },
    "PARTIAL_QUANT_KV_QUANT_PER_TILE": {
        "mm_input_dtype": "bf16",
        "mm_qcqr_input_dtype": "int8",
        "mm_cq_output_dtype": "bf16",
        "mm_ckvkr_output_dtype": "bf16",
        "mm_qcqr_output_dtype": "bf16",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "int8",
        "rmsnorm_ckv_output_dtype": "int8",
        "kv_cache_dtype": "int8",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp32",
        "is_pertile": True,
    },
    "FULL_QUANT_KV_QUANT_PER_TILE": {
        "mm_input_dtype": "int8",
        "mm_qcqr_input_dtype": "int8",
        "mm_cq_output_dtype": "int32",
        "mm_ckvkr_output_dtype": "int32",
        "mm_qcqr_output_dtype": "int32",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "int8",
        "rmsnorm_ckv_output_dtype": "int8",
        "kv_cache_dtype": "int8",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp32",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp32",
        "is_pertile": True,
    },
    "MXFP8_FULL_QUANT_KV_NO_QUANT": {
        "mm_input_dtype": "fp8",
        "mm_qcqr_input_dtype": "fp8",
        "mm_cq_output_dtype": "fp32",
        "mm_ckvkr_output_dtype": "fp32",
        "mm_qcqr_output_dtype": "fp32",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "fp8",
        "rmsnorm_ckv_output_dtype": "bf16",
        "kv_cache_dtype": "bf16",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp8_scale",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp8_scale",
        "is_pertile": False,
    },
    "MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR": {
        "mm_input_dtype": "fp8",
        "mm_qcqr_input_dtype": "fp8",
        "mm_cq_output_dtype": "fp32",
        "mm_ckvkr_output_dtype": "fp32",
        "mm_qcqr_output_dtype": "fp32",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "fp8",
        "rmsnorm_ckv_output_dtype": "fp8",
        "kv_cache_dtype": "fp8",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "fp8",
        "dequant_scale_dtype": "fp8_scale",
        "dequant_scale_qnope_dtype": "fp32",
        "dequant_scale_qnorm_dtype": "fp8_scale",
        "is_pertile": False,
    },
    "MXFP8_FULL_QUANT_KV_QUANT_PER_TILE": {
        "mm_input_dtype": "fp8",
        "mm_qcqr_input_dtype": "fp8",
        "mm_cq_output_dtype": "fp32",
        "mm_ckvkr_output_dtype": "fp32",
        "mm_qcqr_output_dtype": "fp32",
        "mm_qn_input_dtype": "bf16",
        "mm_qn_output_dtype": "bf16",
        "rmsnorm_cq_output_dtype": "fp8",
        "rmsnorm_ckv_output_dtype": "fp8",
        "kv_cache_dtype": "fp8",
        "kr_cache_dtype": "bf16",
        "query_output_dtype": "bf16",
        "dequant_scale_dtype": "fp8_scale",
        "dequant_scale_qnope_dtype": None,
        "dequant_scale_qnorm_dtype": "fp8_scale",
        "is_pertile": True,
    },
}


@dataclass(frozen=True)
class HardwareConfig:
    soc_name: str
    aic_num: int
    aiv_num: int
    cv_mode: str
    gm_bandwidth_gbps: float
    l2_bandwidth_gbps: float
    frequency_ghz_cube: float
    frequency_ghz_vector: float
    ub_size_bytes: int
    l1_size_bytes: int
    l0a_size_bytes: int
    l0b_size_bytes: int
    l0c_size_bytes: int
    cube_bf16_tflops_total: float
    cube_int8_tops_total: float
    cube_fp8_tops_total: float
    vector_fp32_tflops_total: float
    vector_bf16_tflops_total: float
    vector_int8_tops_total: float
    vector_fp8_tops_total: float
    memory_efficiency: Dict[str, float]
    micro_kernel_costs: Dict[str, float]
    cube_formula: Dict[str, Sequence[float]] = field(default_factory=dict)
    vector_formula: Dict[str, Sequence[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseConfig:
    B: int
    S: int
    T: int
    He: int
    Hcq: int
    Hckv: int
    N: int
    Nkv: int
    D: int
    Dr: int
    block_size: int
    cache_mode: str
    actual_seq_mode: str
    weight_quant_mode: int
    kv_quant_mode: int
    query_quant_mode: int
    ckvkr_repo_mode: int
    quant_scale_repo_mode: int
    tile_size: int
    query_norm_flag: int
    qc_qr_scale: float
    kc_scale: float
    smooth_scales_enabled: bool


@dataclass(frozen=True)
class PathInfo:
    quant_mode: str
    is_pertile: bool
    enable_dequant_opt: bool
    enable_group_compute_opt: bool
    need_dequant_or_cast_qc: bool
    need_dynamic_quant_qn_mul_qr: bool
    mm_input_dtype: str
    mm_qcqr_input_dtype: str
    mm_cq_output_dtype: str
    mm_ckvkr_output_dtype: str
    mm_qcqr_output_dtype: str
    mm_qn_input_dtype: str
    mm_qn_output_dtype: str
    rmsnorm_cq_output_dtype: str
    rmsnorm_ckv_output_dtype: str
    kv_cache_dtype: str
    kr_cache_dtype: str
    query_output_dtype: str
    dequant_scale_dtype: str
    dequant_scale_qnope_dtype: Optional[str]
    dequant_scale_qnorm_dtype: Optional[str]


@dataclass(frozen=True)
class TilingParams:
    aic_num: int
    aiv_num: int
    cv_ratio: int
    step_batch_size: int
    vector_block_num: int
    vector_core_num_runtime: int
    cur_vec_token_max: int
    step_num_head_dequant: int
    mm1_single_core_n: int
    mm2_single_core_n: int
    mm3_single_core_n: int
    mm4_single_core_batch: int
    mm1_block_num: int
    mm2_block_num: int
    mm3_block_num: int
    mm4_block_num: int
    mm1_base_n: int
    mm1_base_k: int
    mm1_step_k: int
    mm2_base_n: int
    mm2_base_k: int
    mm2_step_k: int
    mm3_base_n: int
    mm3_base_k: int
    mm3_step_k: int
    mm4_k: int


@dataclass(frozen=True)
class BufferFit:
    ub_bytes_used: int
    ub_bytes_available: int
    l1_a_bytes_required: int
    l1_b_bytes_required: int
    l1_a_bytes_available: int
    l1_b_bytes_available: int
    l0a_bytes_required: int
    l0b_bytes_required: int
    l0c_bytes_required: int
    l0a_bytes_available: int
    l0b_bytes_available: int
    l0c_bytes_available: int
    ub_ok: bool
    l1_ok: bool
    l0_ok: bool
    notes: List[str]


@dataclass
class StageMetrics:
    name: str
    engine: str
    scope: str
    dependencies: List[str]
    ops: float
    gm_bytes: float
    l2_bytes: float
    active_aic: int
    active_aiv: int
    compute_time_us: float
    gm_time_us: float
    l2_time_us: float
    duration_us: float
    bound_label: str
    notes: List[str] = field(default_factory=list)
    chunk_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ops"] = round(self.ops, 4)
        data["gm_bytes"] = round(self.gm_bytes, 4)
        data["l2_bytes"] = round(self.l2_bytes, 4)
        data["compute_time_us"] = round(self.compute_time_us, 6)
        data["gm_time_us"] = round(self.gm_time_us, 6)
        data["l2_time_us"] = round(self.l2_time_us, 6)
        data["duration_us"] = round(self.duration_us, 6)
        return data


@dataclass(frozen=True)
class ScheduleEntry:
    name: str
    start_us: float
    end_us: float


@dataclass
class AnalysisResult:
    case: CaseConfig
    path: PathInfo
    tiling: TilingParams
    first_step_stages: List[StageMetrics]
    steady_step_stages: List[StageMetrics]
    first_step_schedule: List[ScheduleEntry]
    steady_step_schedule: List[ScheduleEntry]
    first_step_time_us: float
    steady_step_time_us: float
    tail_time_us: float
    step_loop_count: int
    tail_tokens: int
    total_kernel_time_us: float
    buffer_fit: BufferFit
    top_bottlenecks: List[StageMetrics]
    critical_path_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": asdict(self.case),
            "path": asdict(self.path),
            "tiling": asdict(self.tiling),
            "first_step_time_us": round(self.first_step_time_us, 6),
            "steady_step_time_us": round(self.steady_step_time_us, 6),
            "tail_time_us": round(self.tail_time_us, 6),
            "step_loop_count": self.step_loop_count,
            "tail_tokens": self.tail_tokens,
            "total_kernel_time_us": round(self.total_kernel_time_us, 6),
            "critical_path_type": self.critical_path_type,
            "buffer_fit": asdict(self.buffer_fit),
            "top_bottlenecks": [stage.to_dict() for stage in self.top_bottlenecks],
            "first_step_stages": [stage.to_dict() for stage in self.first_step_stages],
            "steady_step_stages": [stage.to_dict() for stage in self.steady_step_stages],
            "first_step_schedule": [asdict(entry) for entry in self.first_step_schedule],
            "steady_step_schedule": [asdict(entry) for entry in self.steady_step_schedule],
        }


@dataclass(frozen=True)
class SearchCandidate:
    mode: str
    tiling: TilingParams
    analysis: AnalysisResult
    requires_host_changes: List[str]
    search_objective_us: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "search_objective_us": round(self.search_objective_us, 6),
            "requires_host_changes": list(self.requires_host_changes),
            "tiling": asdict(self.tiling),
            "analysis": self.analysis.to_dict(),
        }


def ceil_div(num: int, den: int) -> int:
    if den == 0:
        return 0
    return (num + den - 1) // den


def align_up(num: int, align: int) -> int:
    if align == 0:
        return 0
    return ceil_div(num, align) * align


def gbps_to_bytes_per_us(gbps: float) -> float:
    return gbps * 1e3


def peak_tops_to_ops_per_us(peak: float) -> float:
    return peak * 1e6


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    return float(value)


def _int_from_keys(data: Mapping[str, Any], *keys: str, default: Optional[int] = None) -> int:
    for key in keys:
        if key in data and data[key] is not None:
            return int(data[key])
    if default is None:
        raise KeyError(f"missing required integer field, checked {keys}")
    return int(default)


def _float_from_keys(data: Mapping[str, Any], *keys: str, default: Optional[float] = None) -> float:
    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])
    if default is None:
        raise KeyError(f"missing required float field, checked {keys}")
    return float(default)


def load_json(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path | str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_hardware_config(path: Path | str) -> HardwareConfig:
    raw = load_json(path)
    cube_formula = raw.get(
        "cube_formula",
        {
            "bf16": [16, 16, 16, 2],
            "int8": [16, 16, 32, 2],
            "fp8": [16, 16, 32, 2],
        },
    )
    vector_formula = raw.get(
        "vector_formula",
        {
            "fp32": [16, 16, 8, 2],
            "bf16": [16, 16, 16, 2],
            "int8": [16, 16, 32, 2],
            "fp8": [16, 16, 32, 2],
        },
    )
    aic_num = int(raw["aic_num"])
    aiv_num = int(raw["aiv_num"])
    freq_cube = float(raw["frequency_ghz_cube"])
    freq_vector = float(raw["frequency_ghz_vector"])

    def derive_total(formula: Sequence[float], core_count: int, freq_ghz: float) -> float:
        value = 1.0
        for item in formula:
            value *= float(item)
        return value * core_count * freq_ghz / 1_000.0

    cube_bf16 = _float_or_none(raw.get("cube_bf16_tflops_total"))
    cube_int8 = _float_or_none(raw.get("cube_int8_tops_total"))
    cube_fp8 = _float_or_none(raw.get("cube_fp8_tops_total"))
    vector_fp32 = _float_or_none(raw.get("vector_fp32_tflops_total"))
    vector_bf16 = _float_or_none(raw.get("vector_bf16_tflops_total"))
    vector_int8 = _float_or_none(raw.get("vector_int8_tops_total"))
    vector_fp8 = _float_or_none(raw.get("vector_fp8_tops_total"))

    if cube_bf16 is None:
        cube_bf16 = derive_total(cube_formula["bf16"], aic_num, freq_cube)
    if cube_int8 is None:
        cube_int8 = derive_total(cube_formula["int8"], aic_num, freq_cube)
    if cube_fp8 is None:
        cube_fp8 = derive_total(cube_formula["fp8"], aic_num, freq_cube)
    if vector_fp32 is None:
        vector_fp32 = derive_total(vector_formula["fp32"], aiv_num, freq_vector)
    if vector_bf16 is None:
        vector_bf16 = derive_total(vector_formula["bf16"], aiv_num, freq_vector)
    if vector_int8 is None:
        vector_int8 = derive_total(vector_formula["int8"], aiv_num, freq_vector)
    if vector_fp8 is None:
        vector_fp8 = derive_total(vector_formula["fp8"], aiv_num, freq_vector)

    return HardwareConfig(
        soc_name=str(raw["soc_name"]),
        aic_num=aic_num,
        aiv_num=aiv_num,
        cv_mode=str(raw["cv_mode"]),
        gm_bandwidth_gbps=float(raw["gm_bandwidth_gbps"]),
        l2_bandwidth_gbps=float(raw["l2_bandwidth_gbps"]),
        frequency_ghz_cube=freq_cube,
        frequency_ghz_vector=freq_vector,
        ub_size_bytes=int(raw["ub_size_bytes"]),
        l1_size_bytes=int(raw["l1_size_bytes"]),
        l0a_size_bytes=int(raw["l0a_size_bytes"]),
        l0b_size_bytes=int(raw["l0b_size_bytes"]),
        l0c_size_bytes=int(raw["l0c_size_bytes"]),
        cube_bf16_tflops_total=cube_bf16,
        cube_int8_tops_total=cube_int8,
        cube_fp8_tops_total=cube_fp8,
        vector_fp32_tflops_total=vector_fp32,
        vector_bf16_tflops_total=vector_bf16,
        vector_int8_tops_total=vector_int8,
        vector_fp8_tops_total=vector_fp8,
        memory_efficiency={str(key): float(value) for key, value in raw["memory_efficiency"].items()},
        micro_kernel_costs={str(key): float(value) for key, value in raw["micro_kernel_costs"].items()},
        cube_formula={str(key): list(value) for key, value in cube_formula.items()},
        vector_formula={str(key): list(value) for key, value in vector_formula.items()},
    )


def load_case_config(path: Path | str) -> CaseConfig:
    return case_from_mapping(load_json(path))


def case_from_mapping(data: Mapping[str, Any]) -> CaseConfig:
    B = _int_from_keys(data, "B", "batch_size", default=1)
    S = _int_from_keys(data, "S", "q_seq", default=1)
    T = data.get("T")
    if T is None:
        T = B * S
    else:
        T = int(T)
        if S == 1 and B > 0 and T % B == 0:
            S = T // B
    cache_mode = str(data.get("cache_mode", "BSND")).upper()
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"unsupported cache_mode {cache_mode}")
    actual_seq_mode = str(data.get("actual_seq_mode", ACTUAL_SEQ_DISABLED)).upper()
    if actual_seq_mode not in {ACTUAL_SEQ_DISABLED, ACTUAL_SEQ_EN_Q_LEN}:
        raise ValueError(f"unsupported actual_seq_mode {actual_seq_mode}")

    return CaseConfig(
        B=B,
        S=S,
        T=T,
        He=_int_from_keys(data, "He"),
        Hcq=_int_from_keys(data, "Hcq"),
        Hckv=_int_from_keys(data, "Hckv"),
        N=_int_from_keys(data, "N", "q_head_num"),
        Nkv=_int_from_keys(data, "Nkv", "kv_head_num", default=1),
        D=_int_from_keys(data, "D", "head_dim"),
        Dr=_int_from_keys(data, "Dr", "rope_head_dim"),
        block_size=_int_from_keys(data, "block_size", default=128),
        cache_mode=cache_mode,
        actual_seq_mode=actual_seq_mode,
        weight_quant_mode=_int_from_keys(data, "weight_quant_mode", default=0),
        kv_quant_mode=_int_from_keys(data, "kv_quant_mode", default=0),
        query_quant_mode=_int_from_keys(data, "query_quant_mode", default=0),
        ckvkr_repo_mode=_int_from_keys(data, "ckvkr_repo_mode", default=0),
        quant_scale_repo_mode=_int_from_keys(data, "quant_scale_repo_mode", default=0),
        tile_size=_int_from_keys(data, "tile_size", default=128),
        query_norm_flag=_int_from_keys(data, "query_norm_flag", default=0),
        qc_qr_scale=_float_from_keys(data, "qc_qr_scale", default=1.0),
        kc_scale=_float_from_keys(data, "kc_scale", default=1.0),
        smooth_scales_enabled=bool(data.get("smooth_scales_enabled", False)),
    )


def derive_quant_mode(case: CaseConfig) -> str:
    if case.weight_quant_mode == WEIGHT_QUANT_NO:
        if case.kv_quant_mode == KV_QUANT_NO:
            return "NO_QUANT"
        raise ValueError("weight_quant_mode=0 only supports kv_quant_mode=0")
    if case.weight_quant_mode == WEIGHT_QUANT_PARTIAL:
        if case.kv_quant_mode == KV_QUANT_NO:
            return "PARTIAL_QUANT_KV_NO_QUANT"
        if case.kv_quant_mode == KV_QUANT_PER_CHANNEL:
            return "PARTIAL_QUANT_KV_QUANT_PER_CHANNEL"
        if case.kv_quant_mode == KV_QUANT_PER_TILE:
            return "PARTIAL_QUANT_KV_QUANT_PER_TILE"
        raise ValueError("weight_quant_mode=1 only supports kv_quant_mode in {0,2,3}")
    if case.weight_quant_mode == WEIGHT_QUANT_FULL:
        if case.kv_quant_mode == KV_QUANT_NO:
            return "FULL_QUANT_KV_NO_QUANT"
        if case.kv_quant_mode == KV_QUANT_PER_TENSOR:
            return "FULL_QUANT_KV_QUANT_PER_TENSOR"
        if case.kv_quant_mode == KV_QUANT_PER_TILE:
            return "FULL_QUANT_KV_QUANT_PER_TILE"
        raise ValueError("weight_quant_mode=2 only supports kv_quant_mode in {0,1,3}")
    if case.weight_quant_mode == WEIGHT_QUANT_MXFP8:
        if case.kv_quant_mode == KV_QUANT_NO:
            return "MXFP8_FULL_QUANT_KV_NO_QUANT"
        if case.kv_quant_mode == KV_QUANT_PER_TENSOR:
            return "MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR"
        if case.kv_quant_mode == KV_QUANT_PER_TILE:
            return "MXFP8_FULL_QUANT_KV_QUANT_PER_TILE"
        raise ValueError("weight_quant_mode=3 only supports kv_quant_mode in {0,1,3}")
    raise ValueError(f"unsupported weight_quant_mode {case.weight_quant_mode}")


def derive_path_info(case: CaseConfig, hw: HardwareConfig) -> PathInfo:
    quant_mode = derive_quant_mode(case)
    enable_group_compute = (
        quant_mode in {"PARTIAL_QUANT_KV_NO_QUANT", "PARTIAL_QUANT_KV_QUANT_PER_CHANNEL"}
        and case.T == 1
        and case.Nkv == 8
        and hw.aic_num >= 16
        and hw.aiv_num >= 32
        and hw.cv_mode != "1:1"
    )
    enable_dequant_opt = False
    if not enable_group_compute:
        if case.weight_quant_mode in {WEIGHT_QUANT_PARTIAL, WEIGHT_QUANT_FULL} and case.N >= 8:
            enable_dequant_opt = True
        if case.weight_quant_mode == WEIGHT_QUANT_MXFP8:
            enable_dequant_opt = True
    path_raw = _PATH_TABLE[quant_mode]
    need_dequant_or_cast_qc = path_raw["mm_qcqr_input_dtype"] in {"int8", "fp8"}
    need_dynamic_quant = (
        case.query_quant_mode == QUERY_QUANT_PER_TOKEN_HEAD
        and quant_mode in {"FULL_QUANT_KV_QUANT_PER_TENSOR", "MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR"}
    )
    return PathInfo(
        quant_mode=quant_mode,
        is_pertile=bool(path_raw["is_pertile"]),
        enable_dequant_opt=enable_dequant_opt,
        enable_group_compute_opt=enable_group_compute,
        need_dequant_or_cast_qc=need_dequant_or_cast_qc,
        need_dynamic_quant_qn_mul_qr=need_dynamic_quant,
        mm_input_dtype=path_raw["mm_input_dtype"],
        mm_qcqr_input_dtype=path_raw["mm_qcqr_input_dtype"],
        mm_cq_output_dtype=path_raw["mm_cq_output_dtype"],
        mm_ckvkr_output_dtype=path_raw["mm_ckvkr_output_dtype"],
        mm_qcqr_output_dtype=path_raw["mm_qcqr_output_dtype"],
        mm_qn_input_dtype=path_raw["mm_qn_input_dtype"],
        mm_qn_output_dtype=path_raw["mm_qn_output_dtype"],
        rmsnorm_cq_output_dtype=path_raw["rmsnorm_cq_output_dtype"],
        rmsnorm_ckv_output_dtype=path_raw["rmsnorm_ckv_output_dtype"],
        kv_cache_dtype=path_raw["kv_cache_dtype"],
        kr_cache_dtype=path_raw["kr_cache_dtype"],
        query_output_dtype=path_raw["query_output_dtype"],
        dequant_scale_dtype=path_raw["dequant_scale_dtype"],
        dequant_scale_qnope_dtype=path_raw["dequant_scale_qnope_dtype"],
        dequant_scale_qnorm_dtype=path_raw["dequant_scale_qnorm_dtype"],
    )


def calc_single_core_n(n: int, core_num: int, align_num: int) -> int:
    return ceil_div(n, align_num * core_num) * align_num


def current_host_tiling(case: CaseConfig, hw: HardwareConfig) -> TilingParams:
    path = derive_path_info(case, hw)
    aic_num = hw.aic_num
    aiv_num = hw.aiv_num
    if path.enable_group_compute_opt:
        aic_num = 16
        aiv_num = 32

    step_batch_size = min(128, case.T)
    vector_block_num = min(step_batch_size, aiv_num)
    step_num_head_dequant = min(64 if case.D == 128 else 16, case.N)

    align_mm1 = BYTE_BLOCK // DTYPE_SIZES[path.mm_input_dtype]
    mm1_single_core_n = calc_single_core_n(case.Hcq, aic_num, align_mm1)
    mm1_single_core_n = max(mm1_single_core_n, 64)
    mm1_block_num = ceil_div(case.Hcq, mm1_single_core_n)

    mm2_total_n = case.Hckv + case.Dr
    if aic_num >= 9:
        mm2_single_core_n = 64
        mm2_block_num = ceil_div(mm2_total_n, mm2_single_core_n)
    else:
        align_mm2 = BYTE_BLOCK // DTYPE_SIZES[path.mm_input_dtype]
        mm2_single_core_n = calc_single_core_n(mm2_total_n, aic_num, align_mm2)
        mm2_block_num = ceil_div(mm2_total_n, mm2_single_core_n)

    mm3_total_n = case.N * (case.D + case.Dr)
    if path.enable_group_compute_opt:
        mm3_single_core_n = calc_single_core_n(case.N * case.D, QC_CORE_NUM, case.D)
    elif path.enable_dequant_opt:
        mm3_single_core_n = calc_single_core_n(mm3_total_n, aic_num, case.D + case.Dr)
    else:
        align_mm3 = BYTE_BLOCK // DTYPE_SIZES[path.mm_qcqr_input_dtype]
        mm3_single_core_n = calc_single_core_n(mm3_total_n, aic_num, align_mm3)
    mm3_block_num = ceil_div(mm3_total_n, mm3_single_core_n)

    mm4_single_core_batch = ceil_div(case.N, aic_num)
    mm4_block_num = ceil_div(case.N, mm4_single_core_batch)

    mm1_base_k = 256 if DTYPE_SIZES[path.mm_input_dtype] == 1 else 128
    mm1_base_n = 64 if path.mm_input_dtype == "fp8" else 128
    mm1_step_k = 4 if (case.He // mm1_base_k) % 4 == 0 else 3

    mm2_base_k = 256 if DTYPE_SIZES[path.mm_input_dtype] == 1 else 128
    mm2_base_n = 128
    mm2_step_k = 4 if (case.He // mm2_base_k) % 4 == 0 else 3

    mm3_base_k = 128 if DTYPE_SIZES[path.mm_qcqr_input_dtype] == 1 else 64
    if path.enable_group_compute_opt:
        mm3_base_n = 128
    elif path.mm_input_dtype == "fp8":
        mm3_base_n = 128
    else:
        mm3_base_n = 256 if step_batch_size <= 64 else 128
    mm3_step_k = 4

    vector_core_num_runtime = vector_block_num
    cv_ratio = 1 if hw.cv_mode == "1:1" else 2
    if cv_ratio != 1 and vector_core_num_runtime < step_batch_size:
        vector_core_num_runtime *= 2
    cur_vec_token_max = max(1, ceil_div(step_batch_size, max(1, vector_core_num_runtime)))

    return TilingParams(
        aic_num=aic_num,
        aiv_num=aiv_num,
        cv_ratio=cv_ratio,
        step_batch_size=step_batch_size,
        vector_block_num=vector_block_num,
        vector_core_num_runtime=vector_core_num_runtime,
        cur_vec_token_max=cur_vec_token_max,
        step_num_head_dequant=step_num_head_dequant,
        mm1_single_core_n=mm1_single_core_n,
        mm2_single_core_n=mm2_single_core_n,
        mm3_single_core_n=mm3_single_core_n,
        mm4_single_core_batch=mm4_single_core_batch,
        mm1_block_num=mm1_block_num,
        mm2_block_num=mm2_block_num,
        mm3_block_num=mm3_block_num,
        mm4_block_num=mm4_block_num,
        mm1_base_n=mm1_base_n,
        mm1_base_k=mm1_base_k,
        mm1_step_k=mm1_step_k,
        mm2_base_n=mm2_base_n,
        mm2_base_k=mm2_base_k,
        mm2_step_k=mm2_step_k,
        mm3_base_n=mm3_base_n,
        mm3_base_k=mm3_base_k,
        mm3_step_k=mm3_step_k,
        mm4_k=case.D,
    )


def override_tiling(
    case: CaseConfig,
    hw: HardwareConfig,
    *,
    step_batch_size: Optional[int] = None,
    vector_block_num: Optional[int] = None,
    step_num_head_dequant: Optional[int] = None,
    mm1_single_core_n: Optional[int] = None,
    mm2_single_core_n: Optional[int] = None,
    mm3_single_core_n: Optional[int] = None,
    mm4_single_core_batch: Optional[int] = None,
    mm3_base_n: Optional[int] = None,
) -> TilingParams:
    base = current_host_tiling(case, hw)
    path = derive_path_info(case, hw)
    step_batch_size = base.step_batch_size if step_batch_size is None else int(step_batch_size)
    step_batch_size = max(1, min(step_batch_size, case.T))
    vector_block_num = base.vector_block_num if vector_block_num is None else int(vector_block_num)
    vector_block_num = max(1, min(vector_block_num, step_batch_size, base.aiv_num))
    step_num_head_dequant = base.step_num_head_dequant if step_num_head_dequant is None else int(step_num_head_dequant)
    step_num_head_dequant = max(1, min(step_num_head_dequant, case.N))
    mm1_single_core_n = base.mm1_single_core_n if mm1_single_core_n is None else int(mm1_single_core_n)
    mm2_single_core_n = base.mm2_single_core_n if mm2_single_core_n is None else int(mm2_single_core_n)
    mm3_single_core_n = base.mm3_single_core_n if mm3_single_core_n is None else int(mm3_single_core_n)
    mm4_single_core_batch = base.mm4_single_core_batch if mm4_single_core_batch is None else int(mm4_single_core_batch)
    mm3_base_n = base.mm3_base_n if mm3_base_n is None else int(mm3_base_n)

    mm1_block_num = ceil_div(case.Hcq, mm1_single_core_n)
    mm2_block_num = ceil_div(case.Hckv + case.Dr, mm2_single_core_n)
    mm3_block_num = ceil_div(case.N * (case.D + case.Dr), mm3_single_core_n)
    mm4_block_num = ceil_div(case.N, mm4_single_core_batch)

    vector_core_num_runtime = vector_block_num
    if base.cv_ratio != 1 and vector_core_num_runtime < step_batch_size:
        vector_core_num_runtime *= 2
    cur_vec_token_max = max(1, ceil_div(step_batch_size, max(1, vector_core_num_runtime)))

    return TilingParams(
        aic_num=base.aic_num,
        aiv_num=base.aiv_num,
        cv_ratio=base.cv_ratio,
        step_batch_size=step_batch_size,
        vector_block_num=vector_block_num,
        vector_core_num_runtime=vector_core_num_runtime,
        cur_vec_token_max=cur_vec_token_max,
        step_num_head_dequant=step_num_head_dequant,
        mm1_single_core_n=mm1_single_core_n,
        mm2_single_core_n=mm2_single_core_n,
        mm3_single_core_n=mm3_single_core_n,
        mm4_single_core_batch=mm4_single_core_batch,
        mm1_block_num=mm1_block_num,
        mm2_block_num=mm2_block_num,
        mm3_block_num=mm3_block_num,
        mm4_block_num=mm4_block_num,
        mm1_base_n=base.mm1_base_n,
        mm1_base_k=base.mm1_base_k,
        mm1_step_k=base.mm1_step_k,
        mm2_base_n=base.mm2_base_n,
        mm2_base_k=base.mm2_base_k,
        mm2_step_k=base.mm2_step_k,
        mm3_base_n=mm3_base_n,
        mm3_base_k=base.mm3_base_k,
        mm3_step_k=base.mm3_step_k,
        mm4_k=case.D,
    )


def dtype_size(dtype_name: Optional[str]) -> int:
    if dtype_name is None:
        return 0
    return DTYPE_SIZES[dtype_name]


def cube_peak_for_dtype(hw: HardwareConfig, dtype_name: str) -> float:
    if dtype_name == "bf16":
        return hw.cube_bf16_tflops_total
    if dtype_name == "int8":
        return hw.cube_int8_tops_total
    if dtype_name == "fp8":
        return hw.cube_fp8_tops_total
    return hw.cube_bf16_tflops_total


def vector_peak_for_dtype(hw: HardwareConfig, dtype_name: str) -> float:
    if dtype_name == "fp32":
        return hw.vector_fp32_tflops_total
    if dtype_name == "bf16":
        return hw.vector_bf16_tflops_total
    if dtype_name == "int8":
        return hw.vector_int8_tops_total
    if dtype_name == "fp8":
        return hw.vector_fp8_tops_total
    return hw.vector_fp32_tflops_total


def stage_compute_time_us(engine: str, dtype_name: str, ops: float, active_aic: int, active_aiv: int, hw: HardwareConfig) -> float:
    if ops <= 0.0:
        return 0.0
    if engine == ENGINE_CUBE:
        total_peak = cube_peak_for_dtype(hw, dtype_name)
        active_scale = active_aic / max(1, hw.aic_num)
        peak_ops_per_us = peak_tops_to_ops_per_us(total_peak * active_scale)
    else:
        total_peak = vector_peak_for_dtype(hw, dtype_name)
        active_scale = active_aiv / max(1, hw.aiv_num)
        peak_ops_per_us = peak_tops_to_ops_per_us(total_peak * active_scale)
    if peak_ops_per_us <= 0.0:
        return 0.0
    return ops / peak_ops_per_us


def stage_memory_time_us(bytes_count: float, bandwidth_gbps: float, efficiency: float) -> float:
    if bytes_count <= 0.0:
        return 0.0
    usable = gbps_to_bytes_per_us(bandwidth_gbps) * max(efficiency, 1e-6)
    return bytes_count / usable


def make_stage(
    *,
    name: str,
    engine: str,
    scope: str,
    dependencies: Sequence[str],
    ops: float,
    gm_bytes: float,
    l2_bytes: float,
    active_aic: int,
    active_aiv: int,
    compute_dtype: str,
    memory_efficiency: float,
    hw: HardwareConfig,
    notes: Optional[Sequence[str]] = None,
    chunk_count: int = 1,
) -> StageMetrics:
    compute_time = stage_compute_time_us(engine, compute_dtype, ops, active_aic, active_aiv, hw)
    gm_time = stage_memory_time_us(gm_bytes, hw.gm_bandwidth_gbps, memory_efficiency)
    l2_time = stage_memory_time_us(l2_bytes, hw.l2_bandwidth_gbps, memory_efficiency)
    duration = max(compute_time, gm_time, l2_time)
    if duration == compute_time:
        bound = BOUND_CUBE if engine == ENGINE_CUBE else BOUND_VECTOR
    elif duration == gm_time:
        bound = BOUND_GM
    else:
        bound = BOUND_L2
    return StageMetrics(
        name=name,
        engine=engine,
        scope=scope,
        dependencies=list(dependencies),
        ops=ops,
        gm_bytes=gm_bytes,
        l2_bytes=l2_bytes,
        active_aic=active_aic,
        active_aiv=active_aiv,
        compute_time_us=compute_time,
        gm_time_us=gm_time,
        l2_time_us=l2_time,
        duration_us=duration,
        bound_label=bound,
        notes=list(notes or []),
        chunk_count=max(1, chunk_count),
    )


def _cache_efficiency(case: CaseConfig, hw: HardwareConfig) -> float:
    if case.cache_mode in {"BSND", "TND"}:
        return hw.memory_efficiency["contiguous"]
    if case.cache_mode in {"PA_BSND", "PA_NZ"}:
        return hw.memory_efficiency["pa"]
    return hw.memory_efficiency["pa_blk"]


def _kernel_ub_cap(mode: str, hw: HardwareConfig) -> int:
    if mode == "exploratory":
        return hw.ub_size_bytes
    return min(hw.ub_size_bytes, KERNEL_MAX_UB_BYTES)


def _mm_k_l1_step(base_k: int, step_k: int) -> int:
    return base_k * step_k


def _mm_n_l1_loops(n_value: int, base_n: int) -> int:
    return max(1, ceil_div(n_value, base_n))


def _share_buffer_rmsnorm_cq(case: CaseConfig, path: PathInfo, tiling: TilingParams) -> int:
    cur_vec_token = tiling.cur_vec_token_max
    output_local = case.Hcq * dtype_size(path.rmsnorm_cq_output_dtype)
    if path.rmsnorm_cq_output_dtype == "fp8":
        scale_local = cur_vec_token * align_up(case.Hcq // FP8_E4M3_BLOCK_SIZE, BYTE_BLOCK)
        dequant_scale_x = align_up(case.He // FP8_E4M3_BLOCK_SIZE, BYTE_BLOCK)
        tmp = (4 * case.Hcq + 8) * 4
    elif path.rmsnorm_cq_output_dtype == "int8":
        scale_local = cur_vec_token * BYTE_BLOCK
        dequant_scale_x = BYTE_BLOCK
        tmp = (4 * case.Hcq + 8) * 4
    else:
        scale_local = cur_vec_token * BYTE_BLOCK
        dequant_scale_x = 0
        tmp = 2 * case.Hcq * 4 + ALIGN_BLOCK_SIZE
    return output_local + scale_local + dequant_scale_x + tmp


def _share_buffer_rmsnorm_ckv_scatter(case: CaseConfig, path: PathInfo) -> int:
    tile_num = ceil_div(case.Hckv, max(case.tile_size, 1))
    if path.is_pertile:
        output_local = align_up(case.Hckv * dtype_size(path.kv_cache_dtype) + tile_num * 4, BYTE_BLOCK)
        quant_tmp = (align_up(tile_num, 8) + tile_num * 8 + case.Hckv) * 4
    else:
        output_local = case.Hckv * dtype_size(path.kv_cache_dtype)
        if case.kv_quant_mode == KV_QUANT_PER_CHANNEL:
            quant_tmp = case.Hckv * 4
        elif case.kv_quant_mode == KV_QUANT_PER_TENSOR:
            quant_tmp = case.Hckv * 4
        else:
            quant_tmp = 0
    rmsnorm_tmp = (4 * case.Hckv + 8) * 4 if path.rmsnorm_ckv_output_dtype in {"int8", "fp8"} else 2 * case.Hckv * 4 + ALIGN_BLOCK_SIZE
    return output_local + max(rmsnorm_tmp, quant_tmp)


def _share_buffer_rope_kr(case: CaseConfig, path: PathInfo) -> int:
    cnt = case.Dr
    output_local = cnt * dtype_size(path.kr_cache_dtype)
    rope_tmp = cnt * 5 * 4
    quant_tmp = cnt * 4 if path.kr_cache_dtype == "int8" else 0
    return output_local + rope_tmp + quant_tmp


def _share_buffer_rope_qr(case: CaseConfig, path: PathInfo) -> int:
    cnt = case.N * case.Dr
    output_local = cnt * 2
    rope_tmp = cnt * 5 * 4
    scale_tmp = cnt * 4 if path.mm_qcqr_input_dtype in {"int8", "fp8"} else 0
    return output_local + rope_tmp + scale_tmp


def _share_buffer_dequant_or_cast(case: CaseConfig, path: PathInfo, tiling: TilingParams) -> int:
    cnt = tiling.step_num_head_dequant * case.D
    input_bytes = cnt * dtype_size(path.mm_qcqr_output_dtype)
    output_bytes = cnt * dtype_size(path.mm_qn_input_dtype)
    if path.mm_qcqr_input_dtype == "int8":
        compute_bytes = cnt * 4
        scale_bytes = max(case.D * 4, BYTE_BLOCK)
        return input_bytes + compute_bytes + scale_bytes + output_bytes
    if path.mm_qcqr_input_dtype == "fp8":
        return input_bytes + output_bytes + BYTE_BLOCK
    return 0


def _share_buffer_dynamic_quant(case: CaseConfig, tiling: TilingParams) -> int:
    row = ceil_div(tiling.step_batch_size, max(1, tiling.cv_ratio))
    sub_row = min(row, 8 if tiling.cv_ratio == 1 else 16)
    compute_size = sub_row * case.Hckv
    rope_total = row * case.Dr
    rope_compute = min(row, 64) * case.Dr
    return (compute_size * 2) + (rope_total * 2 + BYTE_BLOCK) + (rope_compute * 6)


def estimate_buffer_fit(case: CaseConfig, hw: HardwareConfig, tiling: TilingParams, *, mode: str = "host_legal") -> BufferFit:
    path = derive_path_info(case, hw)
    ub_fixed = 0
    if path.mm_input_dtype == "int8":
        ub_fixed += case.Hcq * 4
        ub_fixed += (case.Hckv + case.Dr) * 4
    if path.mm_input_dtype == "fp8":
        ub_fixed += case.Hcq * 4
        ub_fixed += (case.Hckv + case.Dr) * 4
    ub_fixed += case.Hcq * 2
    ub_fixed += case.Hckv * 2
    if path.mm_qcqr_input_dtype == "int8" and case.smooth_scales_enabled:
        ub_fixed += case.Hcq * 4
    if path.kr_cache_dtype == "int8":
        ub_fixed += case.Dr * 4
    if path.rmsnorm_ckv_output_dtype in {"int8", "fp8"}:
        if path.mm_ckvkr_output_dtype == "int32":
            ub_fixed += ALIGN_BLOCK_SIZE
        else:
            ub_fixed += case.Hckv * 4
    ub_fixed += (tiling.step_batch_size + 7) * ALIGN_BLOCK_SIZE
    if path.enable_dequant_opt:
        ub_fixed += 2 * case.Dr * 4 * ceil_div(tiling.step_batch_size, max(1, tiling.cv_ratio))
    else:
        ub_fixed += 2 * case.Dr * 4 * tiling.cur_vec_token_max
    if case.actual_seq_mode == ACTUAL_SEQ_EN_Q_LEN:
        ub_fixed += tiling.step_batch_size * 8

    ub_stage_peak = max(
        _share_buffer_rmsnorm_cq(case, path, tiling),
        _share_buffer_rmsnorm_ckv_scatter(case, path),
        _share_buffer_rope_kr(case, path),
        _share_buffer_rope_qr(case, path),
        _share_buffer_dequant_or_cast(case, path, tiling),
        _share_buffer_dynamic_quant(case, tiling) if path.need_dynamic_quant_qn_mul_qr else 0,
    )
    ub_used = ub_fixed + ub_stage_peak
    ub_available = _kernel_ub_cap(mode, hw)

    mm1_a_l1 = tiling.step_batch_size * _mm_k_l1_step(tiling.mm1_base_k, tiling.mm1_step_k) * dtype_size(path.mm_input_dtype)
    mm1_b_l1 = tiling.mm1_base_n * _mm_k_l1_step(tiling.mm1_base_k, tiling.mm1_step_k) * dtype_size(path.mm_input_dtype)
    mm2_a_l1 = tiling.step_batch_size * _mm_k_l1_step(tiling.mm2_base_k, tiling.mm2_step_k) * dtype_size(path.mm_input_dtype)
    mm2_b_l1 = tiling.mm2_base_n * _mm_k_l1_step(tiling.mm2_base_k, tiling.mm2_step_k) * dtype_size(path.mm_input_dtype)
    mm3_m_limit = INT8_AFULLLOAD_MAX_MSIZE if path.mm_qcqr_input_dtype in {"int8", "fp8"} else BF16_AFULLLOAD_MAX_MSIZE
    if tiling.step_batch_size <= mm3_m_limit:
        mm3_a_l1 = tiling.step_batch_size * case.Hcq * dtype_size(path.rmsnorm_cq_output_dtype)
    else:
        mm3_a_l1 = tiling.step_batch_size * _mm_k_l1_step(tiling.mm3_base_k, tiling.mm3_step_k) * dtype_size(path.rmsnorm_cq_output_dtype)
    mm3_b_l1 = tiling.mm3_base_n * _mm_k_l1_step(tiling.mm3_base_k, tiling.mm3_step_k) * dtype_size(path.mm_qcqr_input_dtype)
    mm4_a_l1 = tiling.step_batch_size * case.D * dtype_size(path.mm_qn_input_dtype)
    mm4_b_l1 = case.Hckv * case.D * dtype_size(path.mm_qn_input_dtype)
    l1_a_required = max(mm1_a_l1, mm2_a_l1, mm3_a_l1, mm4_a_l1)
    l1_b_required = max(mm1_b_l1, mm2_b_l1, mm3_b_l1, mm4_b_l1)

    mm1_m_align = align_up(tiling.step_batch_size, BLOCK_CUBE_SIZE)
    mm1_l0a = mm1_m_align * tiling.mm1_base_k * dtype_size(path.mm_input_dtype)
    mm1_l0b = tiling.mm1_base_n * tiling.mm1_base_k * dtype_size(path.mm_input_dtype)
    mm1_l0c = mm1_m_align * tiling.mm1_base_n * 4
    mm2_l0a = mm1_m_align * tiling.mm2_base_k * dtype_size(path.mm_input_dtype)
    mm2_l0b = tiling.mm2_base_n * tiling.mm2_base_k * dtype_size(path.mm_input_dtype)
    mm2_l0c = mm1_m_align * tiling.mm2_base_n * 4
    mm3_l0a = mm1_m_align * tiling.mm3_base_k * dtype_size(path.rmsnorm_cq_output_dtype)
    mm3_l0b = tiling.mm3_base_n * tiling.mm3_base_k * dtype_size(path.mm_qcqr_input_dtype)
    mm3_l0c = mm1_m_align * tiling.mm3_base_n * 4
    mm4_l0a = mm1_m_align * tiling.mm4_k * dtype_size(path.mm_qn_input_dtype)
    mm4_l0b = case.Hckv * tiling.mm4_k * dtype_size(path.mm_qn_input_dtype)
    mm4_l0c = mm1_m_align * case.Hckv * 4

    l0a_required = max(mm1_l0a, mm2_l0a, mm3_l0a, mm4_l0a)
    l0b_required = max(mm1_l0b, mm2_l0b, mm3_l0b, mm4_l0b)
    l0c_required = max(mm1_l0c, mm2_l0c, mm3_l0c, mm4_l0c)

    l1_a_available = hw.l1_size_bytes // 2 if mode == "exploratory" else KERNEL_L1_A_BYTES
    l1_b_available = hw.l1_size_bytes // 2 if mode == "exploratory" else KERNEL_L1_B_BYTES
    l0a_available = hw.l0a_size_bytes if mode == "exploratory" else KERNEL_L0A_BYTES
    l0b_available = hw.l0b_size_bytes if mode == "exploratory" else KERNEL_L0B_BYTES
    l0c_available = hw.l0c_size_bytes if mode == "exploratory" else KERNEL_L0C_BYTES

    notes: List[str] = []
    if mode != "exploratory" and hw.ub_size_bytes > KERNEL_MAX_UB_BYTES:
        notes.append("kernel uses a hard 192KB vector-UB cap even on larger UB hardware")
    if mode != "exploratory" and hw.l1_size_bytes > (KERNEL_L1_A_BYTES + KERNEL_L1_B_BYTES):
        notes.append("kernel L1 double-buffer constants use 128KB + 128KB, below Ascend950 total L1")

    return BufferFit(
        ub_bytes_used=ub_used,
        ub_bytes_available=ub_available,
        l1_a_bytes_required=l1_a_required,
        l1_b_bytes_required=l1_b_required,
        l1_a_bytes_available=l1_a_available,
        l1_b_bytes_available=l1_b_available,
        l0a_bytes_required=l0a_required,
        l0b_bytes_required=l0b_required,
        l0c_bytes_required=l0c_required,
        l0a_bytes_available=l0a_available,
        l0b_bytes_available=l0b_available,
        l0c_bytes_available=l0c_available,
        ub_ok=ub_used <= ub_available,
        l1_ok=l1_a_required <= l1_a_available and l1_b_required <= l1_b_available,
        l0_ok=l0a_required <= l0a_available and l0b_required <= l0b_available and l0c_required <= l0c_available,
        notes=notes,
    )


def _cube_steady_gm_bytes_mm1(case: CaseConfig) -> float:
    return (case.He * case.Hcq) * 0.0 + (case.He * BYTE_BLOCK * 0.0) + (case.He * 0.0)


def _stage_notes_for_path(path: PathInfo) -> List[str]:
    notes = []
    if path.enable_group_compute_opt:
        notes.append("group-compute path active")
    if path.enable_dequant_opt:
        notes.append("split-N dequant overlap active")
    if path.is_pertile:
        notes.append("per-tile kv quant path active")
    return notes


def _build_stage_metrics(case: CaseConfig, hw: HardwareConfig, tiling: TilingParams, *, steady_state: bool) -> List[StageMetrics]:
    path = derive_path_info(case, hw)
    stages: List[StageMetrics] = []
    eff = _cache_efficiency(case, hw)
    input_dtype_size = dtype_size(path.mm_input_dtype)
    qcqr_input_size = dtype_size(path.mm_qcqr_input_dtype)
    qn_input_size = dtype_size(path.mm_qn_input_dtype)

    if not steady_state:
        copy_global_bytes = (case.Hcq + case.Hckv) * 2
        if path.mm_input_dtype in {"int8", "fp8"}:
            copy_global_bytes += case.Hcq * 4 + (case.Hckv + case.Dr) * 4
        if path.mm_qcqr_input_dtype in {"int8", "fp8"}:
            copy_global_bytes += case.Hcq * (4 if case.smooth_scales_enabled else 0)
        if path.rmsnorm_ckv_output_dtype in {"int8", "fp8"}:
            copy_global_bytes += case.Hckv * (4 if not path.is_pertile else 4)
        if path.kr_cache_dtype == "int8":
            copy_global_bytes += case.Dr * 4
        stages.append(
            make_stage(
                name="copy_global_params",
                engine=ENGINE_DMA,
                scope=SCOPE_ONCE,
                dependencies=[],
                ops=0.0,
                gm_bytes=copy_global_bytes,
                l2_bytes=copy_global_bytes,
                active_aic=0,
                active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
                compute_dtype="fp32",
                memory_efficiency=1.0,
                hw=hw,
                notes=_stage_notes_for_path(path),
            )
        )

    copy_sincos_bytes = case.Dr * case.T * 2 * 2 / max(1, ceil_div(case.T, tiling.step_batch_size))
    stages.append(
        make_stage(
            name="copy_sincos",
            engine=ENGINE_DMA,
            scope=SCOPE_STEP,
            dependencies=["copy_global_params"] if not steady_state else [],
            ops=0.0,
            gm_bytes=copy_sincos_bytes,
            l2_bytes=copy_sincos_bytes,
            active_aic=0,
            active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
            compute_dtype="fp32",
            memory_efficiency=1.0,
            hw=hw,
        )
    )

    mm1_nl1 = _mm_n_l1_loops(tiling.mm1_single_core_n, tiling.mm1_base_n)
    mm1_active = min(tiling.mm1_block_num, tiling.aic_num)
    mm1_ops = 2.0 * tiling.step_batch_size * case.Hcq * case.He
    mm1_gm = (tiling.step_batch_size * case.He * input_dtype_size) + (tiling.step_batch_size * case.Hcq * dtype_size(path.mm_cq_output_dtype))
    if not steady_state:
        mm1_gm += case.He * case.Hcq * input_dtype_size
    mm1_l2 = (mm1_active * mm1_nl1 * tiling.step_batch_size * case.He * input_dtype_size) + (case.He * case.Hcq * input_dtype_size)
    stages.append(
        make_stage(
            name="mm1_cq",
            engine=ENGINE_CUBE,
            scope=SCOPE_STEP,
            dependencies=[],
            ops=mm1_ops,
            gm_bytes=mm1_gm,
            l2_bytes=mm1_l2,
            active_aic=mm1_active,
            active_aiv=0,
            compute_dtype=path.mm_input_dtype,
            memory_efficiency=1.0,
            hw=hw,
            chunk_count=mm1_nl1,
        )
    )

    rmsnorm_cq_elems = tiling.step_batch_size * case.Hcq
    rmsnorm_cq_ops = rmsnorm_cq_elems * hw.micro_kernel_costs["rmsnorm_ops_per_elem"]
    if path.rmsnorm_cq_output_dtype in {"int8", "fp8"}:
        rmsnorm_cq_ops += rmsnorm_cq_elems * hw.micro_kernel_costs["dynamic_quant_ops_per_elem"]
    rmsnorm_cq_gm = 0.0
    if case.query_norm_flag and path.dequant_scale_qnorm_dtype:
        rmsnorm_cq_gm += tiling.step_batch_size * BYTE_BLOCK
    rmsnorm_cq_l2 = (
        tiling.step_batch_size * case.Hcq * dtype_size(path.mm_cq_output_dtype)
        + tiling.step_batch_size * case.Hcq * dtype_size(path.rmsnorm_cq_output_dtype)
        + case.Hcq * 2
    )
    stages.append(
        make_stage(
            name="rmsnorm_cq",
            engine=ENGINE_VECTOR,
            scope=SCOPE_STEP,
            dependencies=["copy_sincos", "mm1_cq"] + (["copy_global_params"] if not steady_state else []),
            ops=rmsnorm_cq_ops,
            gm_bytes=rmsnorm_cq_gm,
            l2_bytes=rmsnorm_cq_l2,
            active_aic=0,
            active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
            compute_dtype="fp32",
            memory_efficiency=1.0,
            hw=hw,
        )
    )

    mm2_nl1 = _mm_n_l1_loops(tiling.mm2_single_core_n, tiling.mm2_base_n)
    mm2_active = min(tiling.mm2_block_num, tiling.aic_num)
    mm2_ops = 2.0 * tiling.step_batch_size * (case.Hckv + case.Dr) * case.He
    mm2_gm = (tiling.step_batch_size * case.He * input_dtype_size)
    if not steady_state:
        mm2_gm += case.He * (case.Hckv + case.Dr) * input_dtype_size
    mm2_l2 = (mm2_active * mm2_nl1 * tiling.step_batch_size * case.He * input_dtype_size) + (case.He * (case.Hckv + case.Dr) * input_dtype_size)
    stages.append(
        make_stage(
            name="mm2_ckvkr",
            engine=ENGINE_CUBE,
            scope=SCOPE_STEP,
            dependencies=[],
            ops=mm2_ops,
            gm_bytes=mm2_gm,
            l2_bytes=mm2_l2,
            active_aic=mm2_active,
            active_aiv=0,
            compute_dtype=path.mm_input_dtype,
            memory_efficiency=1.0,
            hw=hw,
            chunk_count=mm2_nl1,
        )
    )

    ckv_elems = tiling.step_batch_size * case.Hckv
    ckv_ops = ckv_elems * hw.micro_kernel_costs["rmsnorm_ops_per_elem"]
    if case.kv_quant_mode != KV_QUANT_NO:
        if case.kv_quant_mode == KV_QUANT_PER_CHANNEL:
            ckv_ops += ckv_elems * hw.micro_kernel_costs["cast_ops_per_elem"]
        else:
            ckv_ops += ckv_elems * hw.micro_kernel_costs["dynamic_quant_ops_per_elem"]
    ckv_ops += ckv_elems * hw.micro_kernel_costs["scatter_compute_ops_per_elem"]
    ckv_gm = tiling.step_batch_size * case.Hckv * dtype_size(path.kv_cache_dtype)
    if path.is_pertile and case.quant_scale_repo_mode == 1:
        ckv_gm += tiling.step_batch_size * ceil_div(case.Hckv, case.tile_size) * 4
    ckv_l2 = tiling.step_batch_size * case.Hckv * dtype_size(path.mm_ckvkr_output_dtype)
    stages.append(
        make_stage(
            name="rmsnorm_ckv_scatter",
            engine=ENGINE_VECTOR,
            scope=SCOPE_STEP,
            dependencies=["rmsnorm_cq", "mm2_ckvkr"] + (["copy_global_params"] if not steady_state else []),
            ops=ckv_ops,
            gm_bytes=ckv_gm,
            l2_bytes=ckv_l2,
            active_aic=0,
            active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
            compute_dtype="fp32",
            memory_efficiency=eff,
            hw=hw,
        )
    )

    kr_elems = tiling.step_batch_size * case.Dr
    kr_ops = kr_elems * (hw.micro_kernel_costs["rope_ops_per_elem"] + hw.micro_kernel_costs["scatter_compute_ops_per_elem"])
    if path.kr_cache_dtype == "int8":
        kr_ops += kr_elems * hw.micro_kernel_costs["cast_ops_per_elem"]
    kr_gm = tiling.step_batch_size * case.Dr * dtype_size(path.kr_cache_dtype)
    kr_l2 = tiling.step_batch_size * case.Dr * dtype_size(path.mm_ckvkr_output_dtype)
    stages.append(
        make_stage(
            name="rope_kr_scatter",
            engine=ENGINE_VECTOR,
            scope=SCOPE_STEP,
            dependencies=["rmsnorm_ckv_scatter"],
            ops=kr_ops,
            gm_bytes=kr_gm,
            l2_bytes=kr_l2,
            active_aic=0,
            active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
            compute_dtype="fp32",
            memory_efficiency=eff,
            hw=hw,
        )
    )

    mm3_nl1 = _mm_n_l1_loops(tiling.mm3_single_core_n, tiling.mm3_base_n)
    mm3_active = min(tiling.mm3_block_num, tiling.aic_num)
    mm3_ops = 2.0 * tiling.step_batch_size * case.Hcq * (case.N * (case.D + case.Dr))
    mm3_gm = 0.0 if steady_state else case.Hcq * case.N * (case.D + case.Dr) * qcqr_input_size * 0.0
    mm3_l2 = (
        tiling.step_batch_size * case.Hcq * dtype_size(path.rmsnorm_cq_output_dtype)
        + case.Hcq * case.N * (case.D + case.Dr) * qcqr_input_size
        + tiling.step_batch_size * case.N * (case.D + case.Dr) * dtype_size(path.mm_qcqr_output_dtype)
    )
    stages.append(
        make_stage(
            name="mm3_qcqr",
            engine=ENGINE_CUBE,
            scope=SCOPE_STEP,
            dependencies=["rmsnorm_cq", "mm2_ckvkr"],
            ops=mm3_ops,
            gm_bytes=mm3_gm,
            l2_bytes=mm3_l2,
            active_aic=mm3_active,
            active_aiv=0,
            compute_dtype=path.mm_qcqr_input_dtype,
            memory_efficiency=1.0,
            hw=hw,
            chunk_count=mm3_nl1,
        )
    )

    if path.need_dequant_or_cast_qc:
        qc_elems = tiling.step_batch_size * case.N * case.D
        if path.mm_qcqr_input_dtype == "int8":
            qc_ops = qc_elems * hw.micro_kernel_costs["dequant_ops_per_elem"]
        else:
            qc_ops = qc_elems * hw.micro_kernel_costs["cast_ops_per_elem"]
        qc_gm = 0.0
        qc_l2 = (
            tiling.step_batch_size * case.N * case.D * dtype_size(path.mm_qcqr_output_dtype)
            + tiling.step_batch_size * case.N * case.D * dtype_size(path.mm_qn_input_dtype)
        )
        if path.mm_qcqr_input_dtype == "int8":
            qc_l2 += case.N * case.D * 4
        stages.append(
            make_stage(
                name="dequant_or_cast_qc",
                engine=ENGINE_VECTOR,
                scope=SCOPE_STEP,
                dependencies=["mm3_qcqr", "rope_kr_scatter"],
                ops=qc_ops,
                gm_bytes=qc_gm,
                l2_bytes=qc_l2,
                active_aic=0,
                active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
                compute_dtype="fp32",
                memory_efficiency=1.0,
                hw=hw,
                chunk_count=max(1, mm3_nl1),
                notes=["modeled as split-N producer/consumer overlap"] if path.enable_dequant_opt else [],
            )
        )
    else:
        stages.append(
            make_stage(
                name="dequant_or_cast_qc",
                engine=ENGINE_VECTOR,
                scope=SCOPE_STEP,
                dependencies=["mm3_qcqr", "rope_kr_scatter"],
                ops=0.0,
                gm_bytes=0.0,
                l2_bytes=0.0,
                active_aic=0,
                active_aiv=0,
                compute_dtype="fp32",
                memory_efficiency=1.0,
                hw=hw,
                notes=["skipped on bf16 path"],
            )
        )

    rope_qr_elems = tiling.step_batch_size * case.N * case.Dr
    rope_qr_ops = rope_qr_elems * hw.micro_kernel_costs["rope_ops_per_elem"]
    rope_qr_gm = tiling.step_batch_size * case.N * case.Dr * 2
    rope_qr_l2 = tiling.step_batch_size * case.N * case.Dr * dtype_size(path.mm_qcqr_output_dtype)
    stages.append(
        make_stage(
            name="rope_qr",
            engine=ENGINE_VECTOR,
            scope=SCOPE_STEP,
            dependencies=["mm3_qcqr", "rope_kr_scatter"],
            ops=rope_qr_ops,
            gm_bytes=rope_qr_gm,
            l2_bytes=rope_qr_l2,
            active_aic=0,
            active_aiv=max(1, min(tiling.vector_block_num, hw.aiv_num)),
            compute_dtype="fp32",
            memory_efficiency=1.0,
            hw=hw,
            chunk_count=max(1, mm3_nl1 if path.enable_dequant_opt else 1),
        )
    )

    mm4_active = min(tiling.mm4_block_num, tiling.aic_num)
    mm4_ops = 2.0 * tiling.step_batch_size * case.N * case.D * case.Hckv
    mm4_gm = 0.0
    if not path.need_dynamic_quant_qn_mul_qr:
        mm4_gm += tiling.step_batch_size * case.N * case.Hckv * dtype_size(path.mm_qn_output_dtype)
    mm4_l2 = (
        tiling.step_batch_size * case.N * case.D * qn_input_size
        + case.N * case.D * case.Hckv * dtype_size(path.mm_qn_input_dtype)
    )
    if path.need_dynamic_quant_qn_mul_qr:
        mm4_l2 += tiling.step_batch_size * case.N * case.Hckv * dtype_size(path.mm_qn_output_dtype)
    stages.append(
        make_stage(
            name="mm4_qn",
            engine=ENGINE_CUBE,
            scope=SCOPE_STEP,
            dependencies=["dequant_or_cast_qc"] if path.need_dequant_or_cast_qc else ["mm3_qcqr"],
            ops=mm4_ops,
            gm_bytes=mm4_gm,
            l2_bytes=mm4_l2,
            active_aic=mm4_active,
            active_aiv=0,
            compute_dtype=path.mm_qn_input_dtype,
            memory_efficiency=1.0,
            hw=hw,
            chunk_count=max(1, tiling.mm4_single_core_batch),
            notes=["mm4 waits on split-N QC chunks"] if path.enable_dequant_opt and path.need_dequant_or_cast_qc else [],
        )
    )

    if path.need_dynamic_quant_qn_mul_qr:
        qn_elems = tiling.step_batch_size * case.N * case.Hckv
        qr_elems = tiling.step_batch_size * case.N * case.Dr
        dyn_ops = qn_elems * hw.micro_kernel_costs["dynamic_quant_ops_per_elem"] + qr_elems * hw.micro_kernel_costs["rope_ops_per_elem"]
        dyn_gm = (
            tiling.step_batch_size * case.N * case.Hckv * dtype_size(path.query_output_dtype)
            + tiling.step_batch_size * case.N * dtype_size(path.dequant_scale_qnope_dtype)
            + tiling.step_batch_size * case.N * case.Dr * 2
        )
        dyn_l2 = (
            tiling.step_batch_size * case.N * case.Hckv * dtype_size(path.mm_qn_output_dtype)
            + tiling.step_batch_size * case.N * case.Dr * 2
        )
        dyn_active = min(tiling.mm4_block_num * tiling.cv_ratio, hw.aiv_num)
        stages.append(
            make_stage(
                name="dynamic_quant_qn_mul_qr",
                engine=ENGINE_VECTOR,
                scope=SCOPE_STEP,
                dependencies=["mm4_qn", "rope_qr"] + (["copy_global_params"] if not steady_state else []),
                ops=dyn_ops,
                gm_bytes=dyn_gm,
                l2_bytes=dyn_l2,
                active_aic=0,
                active_aiv=max(1, dyn_active),
                compute_dtype="fp32",
                memory_efficiency=1.0,
                hw=hw,
            )
        )
    else:
        stages.append(
            make_stage(
                name="dynamic_quant_qn_mul_qr",
                engine=ENGINE_VECTOR,
                scope=SCOPE_STEP,
                dependencies=["mm4_qn", "rope_qr"],
                ops=0.0,
                gm_bytes=0.0,
                l2_bytes=0.0,
                active_aic=0,
                active_aiv=0,
                compute_dtype="fp32",
                memory_efficiency=1.0,
                hw=hw,
                notes=["skipped when query dynamic quant is disabled"],
            )
        )

    return stages


def _stage_map(stages: Sequence[StageMetrics]) -> Dict[str, StageMetrics]:
    return {stage.name: stage for stage in stages}


def _schedule_standard(stages: Sequence[StageMetrics], *, include_once: bool) -> Tuple[List[ScheduleEntry], float]:
    stage_by_name = _stage_map(stages)
    order = [
        "copy_global_params",
        "copy_sincos",
        "mm1_cq",
        "rmsnorm_cq",
        "mm2_ckvkr",
        "rmsnorm_ckv_scatter",
        "rope_kr_scatter",
        "mm3_qcqr",
        "dequant_or_cast_qc",
        "rope_qr",
        "mm4_qn",
        "dynamic_quant_qn_mul_qr",
    ]
    end_times: Dict[str, float] = {}
    schedule: List[ScheduleEntry] = []
    for name in order:
        if name not in stage_by_name:
            continue
        stage = stage_by_name[name]
        if stage.scope == SCOPE_ONCE and not include_once:
            continue
        start = 0.0
        for dep in stage.dependencies:
            start = max(start, end_times.get(dep, 0.0))
        end = start + stage.duration_us
        end_times[name] = end
        schedule.append(ScheduleEntry(name=name, start_us=start, end_us=end))
    return schedule, max((entry.end_us for entry in schedule), default=0.0)


def _split_n_pipeline_time(mm3: StageMetrics, qc: StageMetrics, mm4: StageMetrics) -> float:
    chunk_count = max(mm3.chunk_count, qc.chunk_count, mm4.chunk_count, 1)
    mm3_chunk = mm3.duration_us / max(mm3.chunk_count, 1)
    qc_chunk = qc.duration_us / max(qc.chunk_count, 1)
    mm4_chunk = mm4.duration_us / max(mm4.chunk_count, 1)
    warmup = mm3_chunk + qc_chunk + mm4_chunk
    if chunk_count == 1:
        return warmup
    steady = (chunk_count - 1) * max(mm3_chunk, qc_chunk, mm4_chunk)
    return warmup + steady


def _schedule_with_split_n(stages: Sequence[StageMetrics], *, include_once: bool) -> Tuple[List[ScheduleEntry], float]:
    stage_by_name = _stage_map(stages)
    copy_global = stage_by_name.get("copy_global_params")
    copy_sincos = stage_by_name["copy_sincos"]
    mm1 = stage_by_name["mm1_cq"]
    rmsnorm_cq = stage_by_name["rmsnorm_cq"]
    mm2 = stage_by_name["mm2_ckvkr"]
    ckv = stage_by_name["rmsnorm_ckv_scatter"]
    kr = stage_by_name["rope_kr_scatter"]
    mm3 = stage_by_name["mm3_qcqr"]
    qc = stage_by_name["dequant_or_cast_qc"]
    rope_qr = stage_by_name["rope_qr"]
    mm4 = stage_by_name["mm4_qn"]
    dyn = stage_by_name["dynamic_quant_qn_mul_qr"]

    entries: List[ScheduleEntry] = []
    t_copy_global = copy_global.duration_us if include_once and copy_global is not None else 0.0
    if include_once and copy_global is not None:
        entries.append(ScheduleEntry("copy_global_params", 0.0, t_copy_global))

    copy_sincos_start = t_copy_global
    copy_sincos_end = copy_sincos_start + copy_sincos.duration_us
    entries.append(ScheduleEntry("copy_sincos", copy_sincos_start, copy_sincos_end))

    mm1_start = 0.0
    mm1_end = mm1.duration_us
    mm2_start = 0.0
    mm2_end = mm2.duration_us
    entries.append(ScheduleEntry("mm1_cq", mm1_start, mm1_end))
    entries.append(ScheduleEntry("mm2_ckvkr", mm2_start, mm2_end))

    rmsnorm_start = max(copy_sincos_end, mm1_end, t_copy_global)
    rmsnorm_end = rmsnorm_start + rmsnorm_cq.duration_us
    entries.append(ScheduleEntry("rmsnorm_cq", rmsnorm_start, rmsnorm_end))

    ckv_start = max(rmsnorm_end, mm2_end, t_copy_global)
    ckv_end = ckv_start + ckv.duration_us
    kr_start = ckv_end
    kr_end = kr_start + kr.duration_us
    entries.append(ScheduleEntry("rmsnorm_ckv_scatter", ckv_start, ckv_end))
    entries.append(ScheduleEntry("rope_kr_scatter", kr_start, kr_end))

    mm3_start = max(rmsnorm_end, mm2_end)
    mm3_end = mm3_start + mm3.duration_us
    entries.append(ScheduleEntry("mm3_qcqr", mm3_start, mm3_end))

    vector_split_start = max(mm3_start, kr_end)
    qc_end = vector_split_start + qc.duration_us
    rope_qr_end = vector_split_start + rope_qr.duration_us
    entries.append(ScheduleEntry("dequant_or_cast_qc", vector_split_start, qc_end))
    entries.append(ScheduleEntry("rope_qr", vector_split_start, rope_qr_end))

    mm4_start = max(kr_end, mm3_start)
    mm4_pipeline = _split_n_pipeline_time(mm3, qc, mm4)
    mm4_end = mm4_start + mm4_pipeline
    entries.append(ScheduleEntry("mm4_qn", mm4_start, mm4_end))

    dyn_start = max(mm4_end, rope_qr_end, t_copy_global)
    dyn_end = dyn_start + dyn.duration_us
    entries.append(ScheduleEntry("dynamic_quant_qn_mul_qr", dyn_start, dyn_end))
    return entries, max((entry.end_us for entry in entries), default=0.0)


def schedule_stages(stages: Sequence[StageMetrics], *, split_n_overlap: bool, include_once: bool) -> Tuple[List[ScheduleEntry], float]:
    if split_n_overlap:
        return _schedule_with_split_n(stages, include_once=include_once)
    return _schedule_standard(stages, include_once=include_once)


def analyze_case(case: CaseConfig, hw: HardwareConfig, *, tiling: Optional[TilingParams] = None, mode: str = "host_legal") -> AnalysisResult:
    tiling = tiling or current_host_tiling(case, hw)
    path = derive_path_info(case, hw)
    buffer_fit = estimate_buffer_fit(case, hw, tiling, mode=mode)
    first_step_stages = _build_stage_metrics(case, hw, tiling, steady_state=False)
    steady_step_stages = _build_stage_metrics(case, hw, tiling, steady_state=True)
    split_n_overlap = path.enable_dequant_opt and path.need_dequant_or_cast_qc
    first_schedule, first_time = schedule_stages(first_step_stages, split_n_overlap=split_n_overlap, include_once=True)
    steady_schedule, steady_time = schedule_stages(steady_step_stages, split_n_overlap=split_n_overlap, include_once=False)

    tail_tokens = case.T % tiling.step_batch_size
    if case.T <= tiling.step_batch_size:
        step_loop_count = 1
        tail_time = 0.0
    else:
        step_loop_count = case.T // tiling.step_batch_size
        if tail_tokens:
            tail_case = CaseConfig(
                **{
                    **asdict(case),
                    "T": tail_tokens,
                    "S": tail_tokens if case.B == 1 else case.S,
                }
            )
            tail_tiling = override_tiling(tail_case, hw, step_batch_size=tail_tokens)
            tail_stages = _build_stage_metrics(tail_case, hw, tail_tiling, steady_state=True)
            _, tail_time = schedule_stages(tail_stages, split_n_overlap=split_n_overlap, include_once=False)
        else:
            tail_time = 0.0

    if case.T <= tiling.step_batch_size:
        total_kernel_time = first_time
    else:
        total_kernel_time = first_time + max(step_loop_count - 1, 0) * steady_time + tail_time

    all_steady = sorted(
        (stage for stage in steady_step_stages if stage.duration_us > 0.0),
        key=lambda stage: stage.duration_us,
        reverse=True,
    )
    top_bottlenecks = all_steady[:5]
    cube_sum = sum(stage.duration_us for stage in steady_step_stages if stage.engine == ENGINE_CUBE)
    vector_sum = sum(stage.duration_us for stage in steady_step_stages if stage.engine == ENGINE_VECTOR)
    critical_path_type = "cube_side" if cube_sum >= vector_sum else "vector_side"

    return AnalysisResult(
        case=case,
        path=path,
        tiling=tiling,
        first_step_stages=first_step_stages,
        steady_step_stages=steady_step_stages,
        first_step_schedule=first_schedule,
        steady_step_schedule=steady_schedule,
        first_step_time_us=first_time,
        steady_step_time_us=steady_time,
        tail_time_us=tail_time,
        step_loop_count=step_loop_count,
        tail_tokens=tail_tokens,
        total_kernel_time_us=total_kernel_time,
        buffer_fit=buffer_fit,
        top_bottlenecks=top_bottlenecks,
        critical_path_type=critical_path_type,
    )


def parse_sweep_items(sweep_items: Sequence[str]) -> Dict[str, List[Any]]:
    sweep: Dict[str, List[Any]] = {}
    for item in sweep_items:
        if "=" not in item:
            raise ValueError(f"invalid sweep item {item!r}, expected key=value1,value2")
        key, raw_values = item.split("=", 1)
        values: List[Any] = []
        for value in raw_values.split(","):
            value = value.strip()
            if value == "":
                continue
            if value.upper() in CACHE_MODES or value.upper() in {ACTUAL_SEQ_DISABLED, ACTUAL_SEQ_EN_Q_LEN}:
                values.append(value.upper())
                continue
            if "." in value:
                values.append(float(value))
            else:
                values.append(int(value))
        sweep[key.strip()] = values
    return sweep


def apply_case_updates(case: CaseConfig, updates: Mapping[str, Any]) -> CaseConfig:
    payload = asdict(case)
    payload.update(updates)
    return case_from_mapping(payload)


def sweep_cases(base_case: CaseConfig, sweep: Mapping[str, Sequence[Any]]) -> List[CaseConfig]:
    items = list(sweep.items())
    if not items:
        return [base_case]
    cases = [base_case]
    for key, values in items:
        next_cases: List[CaseConfig] = []
        for case in cases:
            for value in values:
                next_cases.append(apply_case_updates(case, {key: value}))
        cases = next_cases
    return cases


def analysis_rows_for_csv(result: AnalysisResult, *, steady: bool = True) -> List[Dict[str, Any]]:
    stages = result.steady_step_stages if steady else result.first_step_stages
    rows: List[Dict[str, Any]] = []
    for stage in stages:
        row = stage.to_dict()
        row["steady_state"] = steady
        row["B"] = result.case.B
        row["S"] = result.case.S
        row["T"] = result.case.T
        row["cache_mode"] = result.case.cache_mode
        row["quant_mode"] = result.path.quant_mode
        row["critical_path_type"] = result.critical_path_type
        row["step_batch_size"] = result.tiling.step_batch_size
        rows.append(row)
    return rows


def best_candidates(
    case: CaseConfig,
    hw: HardwareConfig,
    *,
    mode: str,
    top_k: int = 5,
    objective: str = "steady_step",
) -> List[SearchCandidate]:
    base = current_host_tiling(case, hw)
    candidates: List[TilingParams] = []
    if mode == "host_legal":
        candidates = [base]
    else:
        step_choices = [value for value in (16, 32, 48, 64, 96, 128) if value <= case.T]
        if base.step_batch_size not in step_choices:
            step_choices.append(base.step_batch_size)
        step_choices = sorted(set(step_choices))

        vector_choices = sorted(
            {
                base.vector_block_num,
                *[
                    value
                    for value in range(1, min(case.T, hw.aiv_num) + 1)
                    if value in {1, 2, 4, 8, 16, 32, 48, 64} and min(case.T, hw.aiv_num) % value == 0
                ],
            }
        )
        dequant_choices = sorted({base.step_num_head_dequant, *[value for value in (16, 32, 64) if value <= case.N]})

        path = derive_path_info(case, hw)
        mm1_align = BYTE_BLOCK // dtype_size(path.mm_input_dtype)
        mm3_align = case.D + case.Dr if path.enable_dequant_opt else BYTE_BLOCK // dtype_size(path.mm_qcqr_input_dtype)
        mm1_candidates = sorted(
            {
                base.mm1_single_core_n,
                *[
                    calc_single_core_n(case.Hcq, active_cores, mm1_align)
                    for active_cores in (4, 8, 12, 16, 20, 24, 32)
                ],
                64,
            }
        )
        mm2_candidates = sorted(
            {
                base.mm2_single_core_n,
                *[
                    align_up(ceil_div(case.Hckv + case.Dr, active_cores), 32)
                    for active_cores in (4, 6, 8, 9, 12, 16, 24, 32)
                ],
                64,
            }
        )
        mm3_candidates = sorted(
            {
                base.mm3_single_core_n,
                *[
                    calc_single_core_n(case.N * (case.D + case.Dr), active_cores, mm3_align)
                    for active_cores in (4, 8, 12, 16, 20, 24, 32)
                ],
            }
        )
        mm4_candidates = sorted({base.mm4_single_core_batch, *[max(1, ceil_div(case.N, active)) for active in (4, 8, 16, 32)]})

        base_n_choices = {base.mm3_base_n}
        if path.mm_qcqr_input_dtype != "fp8":
            base_n_choices.update({128, 256})

        for step_batch_size in step_choices:
            for vector_block_num in vector_choices:
                if vector_block_num > step_batch_size:
                    continue
                for step_num_head_dequant in dequant_choices:
                    for mm1_single_core_n in mm1_candidates:
                        for mm2_single_core_n in mm2_candidates:
                            for mm3_single_core_n in mm3_candidates:
                                for mm4_single_core_batch in mm4_candidates:
                                    for mm3_base_n in base_n_choices:
                                        candidate = override_tiling(
                                            case,
                                            hw,
                                            step_batch_size=step_batch_size,
                                            vector_block_num=vector_block_num,
                                            step_num_head_dequant=step_num_head_dequant,
                                            mm1_single_core_n=mm1_single_core_n,
                                            mm2_single_core_n=mm2_single_core_n,
                                            mm3_single_core_n=mm3_single_core_n,
                                            mm4_single_core_batch=mm4_single_core_batch,
                                            mm3_base_n=mm3_base_n,
                                        )
                                        candidates.append(candidate)

    unique_candidates: Dict[Tuple[Any, ...], TilingParams] = {}
    for candidate in candidates:
        key = (
            candidate.step_batch_size,
            candidate.vector_block_num,
            candidate.step_num_head_dequant,
            candidate.mm1_single_core_n,
            candidate.mm2_single_core_n,
            candidate.mm3_single_core_n,
            candidate.mm4_single_core_batch,
            candidate.mm3_base_n,
        )
        unique_candidates[key] = candidate

    ranked: List[SearchCandidate] = []
    for candidate in unique_candidates.values():
        fit = estimate_buffer_fit(case, hw, candidate, mode=mode)
        # L0 usage is kept as an advisory signal in this first-pass model because the
        # matmul micro-kernel performs additional packing/reuse that is not modeled here.
        if not (fit.ub_ok and fit.l1_ok):
            continue
        analysis = analyze_case(case, hw, tiling=candidate, mode=mode)
        objective_value = analysis.steady_step_time_us if objective == "steady_step" else analysis.total_kernel_time_us
        requires: List[str] = []
        if mode == "exploratory":
            if candidate.step_batch_size != base.step_batch_size:
                requires.append("stepBatchSize host rule")
            if candidate.vector_block_num != base.vector_block_num:
                requires.append("vectorBlockNum host rule")
            if candidate.step_num_head_dequant != base.step_num_head_dequant:
                requires.append("stepNumHeadDequant host rule")
            if candidate.mm1_single_core_n != base.mm1_single_core_n:
                requires.append("mm1 single-core N heuristic")
            if candidate.mm2_single_core_n != base.mm2_single_core_n:
                requires.append("mm2 single-core N heuristic")
            if candidate.mm3_single_core_n != base.mm3_single_core_n:
                requires.append("mm3 single-core N heuristic")
            if candidate.mm4_single_core_batch != base.mm4_single_core_batch:
                requires.append("mm4 head split heuristic")
            if candidate.mm3_base_n != base.mm3_base_n:
                requires.append("mmQcQr.baseN kernel heuristic")
        ranked.append(
            SearchCandidate(
                mode=mode,
                tiling=candidate,
                analysis=analysis,
                requires_host_changes=requires,
                search_objective_us=objective_value,
            )
        )

    ranked.sort(key=lambda item: (item.search_objective_us, item.analysis.total_kernel_time_us))
    return ranked[:top_k]


def _format_number(value: float, digits: int = 3) -> str:
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def render_stage_table(stages: Sequence[StageMetrics]) -> str:
    lines = [
        "| Stage | Engine | Scope | Bound | Ops | GM Bytes | L2 Bytes | Duration (us) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        lines.append(
            "| {name} | {engine} | {scope} | {bound} | {ops} | {gm} | {l2} | {dur} |".format(
                name=stage.name,
                engine=stage.engine,
                scope=stage.scope,
                bound=stage.bound_label,
                ops=_format_number(stage.ops, 2),
                gm=_format_number(stage.gm_bytes, 2),
                l2=_format_number(stage.l2_bytes, 2),
                dur=_format_number(stage.duration_us, 3),
            )
        )
    return "\n".join(lines)


def render_analysis_text(result: AnalysisResult) -> str:
    lines = [
        f"Case: B={result.case.B}, S={result.case.S}, T={result.case.T}, cache_mode={result.case.cache_mode}, quant_mode={result.path.quant_mode}",
        f"Tiling: stepBatchSize={result.tiling.step_batch_size}, vectorBlockNum={result.tiling.vector_block_num}, "
        f"stepNumHeadDequant={result.tiling.step_num_head_dequant}, mm1/mm2/mm3/mm4 blocks="
        f"{result.tiling.mm1_block_num}/{result.tiling.mm2_block_num}/{result.tiling.mm3_block_num}/{result.tiling.mm4_block_num}",
        f"Critical path type: {result.critical_path_type}",
        f"First-step time: {_format_number(result.first_step_time_us, 3)} us",
        f"Steady-step time: {_format_number(result.steady_step_time_us, 3)} us",
        f"Tail time: {_format_number(result.tail_time_us, 3)} us",
        f"Estimated total kernel time: {_format_number(result.total_kernel_time_us, 3)} us",
        "",
        "Steady-state stage table:",
        render_stage_table(result.steady_step_stages),
    ]
    return "\n".join(lines)


def render_report(hw: HardwareConfig, analyses: Mapping[str, AnalysisResult], searches: Mapping[str, Sequence[SearchCandidate]]) -> str:
    cube_bf16 = hw.cube_bf16_tflops_total
    cube_int8 = hw.cube_int8_tops_total
    cube_fp8 = hw.cube_fp8_tops_total
    vector_bf16 = hw.vector_bf16_tflops_total
    vector_fp32 = hw.vector_fp32_tflops_total
    vector_int8 = hw.vector_int8_tops_total
    vector_fp8 = hw.vector_fp8_tops_total

    def top_line(key: str) -> str:
        result = analyses[key]
        stage = result.top_bottlenecks[0] if result.top_bottlenecks else None
        if stage is None:
            return f"- `{key}`: no active stages"
        return (
            f"- `{key}`: steady-step `{_format_number(result.steady_step_time_us, 3)} us`, "
            f"critical side `{result.critical_path_type}`, top stage `{stage.name}` ({stage.bound_label}, "
            f"{_format_number(stage.duration_us, 3)} us)"
        )

    report = [
        "# MlaPrologV3 理论性能分析报告",
        "",
        "## 1. Kernel 分解与分析边界",
        "",
        "本报告基于 `kernel_mla_prolog_split_n.h` 与 `mla_prolog_tiling.cpp` 的当前实现，"
        "按真实的 CV 同步关系拆成 `copy_global_params`、`copy_sincos`、`mm1_cq`、`rmsnorm_cq`、"
        "`mm2_ckvkr`、`rmsnorm_ckv_scatter`、`rope_kr_scatter`、`mm3_qcqr`、`dequant_or_cast_qc`、"
        "`rope_qr`、`mm4_qn`、`dynamic_quant_qn_mul_qr` 等阶段。",
        "",
        "分析假设:",
        "- 外部输入、最终输出与 cache scatter 走 GM 带宽模型。",
        "- kernel 内部 workspace 与 weight 复用优先按 L2 带宽模型估算。",
        "- `enableDequantOpt` 场景下，`mm3_qcqr -> dequant_or_cast_qc -> mm4_qn` 按 split-N producer/consumer pipeline 估算暴露时长。",
        "- 这是理论模型，不包含 runtime profile、DMA 冲突、bank conflict 与指令级泡沫的实测修正。",
        "",
        "## 2. Ascend950 硬件模型",
        "",
        f"- SoC: `{hw.soc_name}`, AIC={hw.aic_num}, AIV={hw.aiv_num}, CV 模式=`{hw.cv_mode}`",
        f"- Cube 峰值: BF16={cube_bf16:.3f} TFLOPS, INT8={cube_int8:.3f} TOPS, FP8={cube_fp8:.3f} TOPS",
        f"- Vector 峰值: FP32={vector_fp32:.3f} TFLOPS, BF16={vector_bf16:.3f} TFLOPS, INT8={vector_int8:.3f} TOPS, FP8={vector_fp8:.3f} TOPS",
        f"- 带宽: GM={hw.gm_bandwidth_gbps / 1000.0:.3f} TB/s, L2={hw.l2_bandwidth_gbps / 1000.0:.3f} TB/s",
        f"- 本地存储: UB={hw.ub_size_bytes / 1024:.1f} KB, L1={hw.l1_size_bytes / 1024:.1f} KB, "
        f"L0A/L0B/L0C={hw.l0a_size_bytes / 1024:.1f}/{hw.l0b_size_bytes / 1024:.1f}/{hw.l0c_size_bytes / 1024:.1f} KB",
        "",
        "需要特别指出的当前 kernel 约束:",
        "- `stepBatchSize` 仍然被 host 固定为 `min(128, T)`。",
        "- vector path 仍然使用 `MAX_UB_SIZE = 192KB` 的静态上限，而不是 Ascend950 的 248KB。",
        "- cube path 仍然按 `L1_A_SIZE = 128KB`, `L1_B_SIZE = 128KB`, `L0A/L0B/L0C = 32/32/64KB` 的旧静态常量组织双 buffer。",
        "",
        "## 3. 当前 Host Tiling 特征",
        "",
        "以 canonical 形状 `He=7168, Hcq=1536, Hckv=512, D=128, Dr=64, N=128` 为例，Ascend950 当前 host 规则得到:",
        f"- `stepBatchSize = {analyses['bf16_large'].tiling.step_batch_size}`, `vectorBlockNum = {analyses['bf16_large'].tiling.vector_block_num}`",
        f"- `mm1BlockNum = {analyses['bf16_large'].tiling.mm1_block_num}`，说明 `MatmulCq` 只激活 24 个 cube block",
        f"- `mm2BlockNum = {analyses['bf16_large'].tiling.mm2_block_num}`，说明 `MatmulCkvKr` 只激活 9 个 cube block",
        f"- `mm3BlockNum = {analyses['bf16_large'].tiling.mm3_block_num}`, `mm4BlockNum = {analyses['bf16_large'].tiling.mm4_block_num}`，"
        "`MatmulQcQr` 与 `MatmulQn` 才能基本吃满 32 个 AIC",
        "",
        "这意味着当前 tiling 在 Ascend950 上的主要问题不是 `mm3/mm4`，而是前两段 cube matmul 天然欠并行。",
        "",
        "## 4. 分场景瓶颈结论",
        "",
        top_line("bf16_small"),
        top_line("bf16_large"),
        top_line("int8_tensor"),
        top_line("fp8_tensor"),
        top_line("pa_blk_stress"),
        "",
        "从模型结果归纳:",
        "- BF16 / 非量化路径: 主瓶颈仍落在 cube 侧，核心压力集中在 `mm1_cq` 与 `mm3_qcqr`。其中 `mm1_cq` 经常先撞到 L2 供数，而不是 vector 侧。",
        "- INT8 全量化 + per-tensor kv + query quant: 在当前 canonical 大形状上仍是 cube-side bound，但 `dequant_or_cast_qc`、`rope_qr`、`dynamic_quant_qn_mul_qr` 与 cache/update 使 vector 暴露时长明显上升，最容易先把瓶颈推向 vector + memory。",
        "- MXFP8-like 路径: 当前 canonical 大形状下仍由 `mm3_qcqr` 主导，但 `float` 累加输出、split-N dequant/cast、rope 和最终 query post-process 使后半段的 vector/L2 压力更突出。",
        "- PA/PA_BLK cache 模式: scatter efficiency 降低后，GM/L2 在 `rmsnorm_ckv_scatter` 与 `rope_kr_scatter` 上更容易主导总时长，尤其是 `PA_BLK_*`。",
        "",
        "### Regime Map",
        "",
        "- BF16 canonical path: 预计为 cube-side bound，热点集中在 `MatmulCq` 与 `MatmulQcQr`。",
        "- INT8 / FP8 full-quant path: 在当前 canonical `T=144` case 上仍是 cube-side bound，但最脆弱的部分已经转到 `dequant/cast + rope + scatter + dynamic quant`，cache 更差或 shape 更偏 vector 时更容易变成 vector + memory bound。",
        "- PA-style cache writes: 最终瓶颈取决于 scatter memory efficiency，可能是 vector 算子本身，也可能直接退化成 GM 带宽瓶颈。",
        "",
        "## 5. `stepBatchSize = 128` 何时不优",
        "",
        "当前 host 固定 `stepBatchSize = 128` 有两个问题:",
        "- 当 `T` 远大于 128 且 path 已经被 vector/scatter 主导时，更大的 step 会放大 `rope_qr`、`dynamic_quant_qn_mul_qr` 与 cache scatter 的单步尾延迟。",
        "- 当 `mm3` 允许 `baseN = 256` 的 BF16/INT8 非 FP8 场景，`stepBatchSize <= 64` 会触发更激进的 `mmQcQr.baseN`，有机会改善 `mm3` 的 L1/L0 利用率和 steady-step time。",
        "",
        "模型上的结论不是“永远把 128 改小”，而是:",
        "- cube-bound 且 scatter 不重时，128 仍然合理，因为它提升了 `mm1/mm3/mm4` 的 M 维利用率。",
        "- vector/scatter-bound 时，128 往往开始拖累后半段，应该允许更小的 `stepBatchSize` 进入候选集。",
        "",
        "## 6. 优化建议",
        "",
        "- 优先改 `mm2SingleCoreN = 64` 的固定经验值。Ascend950 上这会把 `MatmulCkvKr` 长期锁死在 9 个 cube block，明显浪费 32 AIC。",
        "- 重新审视 `mm1SingleCoreN >= 64` 的下界。对 `Hcq = 1536` 这类常见形状，它让 `MatmulCq` 只能启用 24 个 cube block。",
        "- 给 host 增加可搜索的 `stepBatchSize` 与 `vectorBlockNum`，而不是把它们写死为 `min(128, T)` / `min(stepBatchSize, aiv_num)`。",
        "- 允许 Ascend950 使用更大的 vector UB 预算。当前 `MAX_UB_SIZE = 192KB` 会直接吃掉新芯片的额外 56KB 空间。",
        "- 让 `mmQcQr.baseN`、`mm1/mm2/mm3/mm4` split heuristic 与 Ascend950 分离配置，不再沿用旧 NPU 的经验常量。",
        "- 对 PA / PA_BLK 路径单独做 memory-oriented tiling，重点降低 `rmsnorm_ckv_scatter` 与 `rope_kr_scatter` 的 GM 暴露时长。",
        "",
        "## 7. Search 结果摘要",
        "",
    ]
    for key, candidates in searches.items():
        if not candidates:
            report.append(f"- `{key}`: no legal candidate found")
            continue
        best = candidates[0]
        report.append(
            f"- `{key}`: best objective `{_format_number(best.search_objective_us, 3)} us`, "
            f"`stepBatchSize={best.tiling.step_batch_size}`, `vectorBlockNum={best.tiling.vector_block_num}`, "
            f"`mm1/mm2/mm3/mm4={best.tiling.mm1_single_core_n}/{best.tiling.mm2_single_core_n}/"
            f"{best.tiling.mm3_single_core_n}/{best.tiling.mm4_single_core_batch}`"
        )
        if best.requires_host_changes:
            report.append(f"  requires host/kernel changes: {', '.join(best.requires_host_changes)}")
        else:
            report.append("  no better host-legal change was found under current C++ tiling policy")
    report.append("")
    report.append("## 8. BF16 大形状稳态阶段表")
    report.append("")
    report.append(render_stage_table(analyses["bf16_large"].steady_step_stages))
    report.append("")
    report.append("## 9. INT8 全量化稳态阶段表")
    report.append("")
    report.append(render_stage_table(analyses["int8_tensor"].steady_step_stages))
    report.append("")
    return "\n".join(report)
