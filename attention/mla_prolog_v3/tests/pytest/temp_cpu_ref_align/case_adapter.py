import sys
from pathlib import Path

import numpy as np
import torch


PYTEST_DIR = Path(__file__).resolve().parents[1]
if str(PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTEST_DIR))

import prologv3_generalized


def _dtype_to_old_name(dtype):
    hif8_dtype = prologv3_generalized._get_hif8_dtype()
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.int8:
        return "int8"
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    if hif8_dtype is not None and dtype == hif8_dtype:
        return "hifloat8"
    return str(dtype).replace("torch.", "")


def _tensor_or_empty(tensor, dtype=torch.float32):
    if tensor is None:
        return torch.empty((0,), dtype=dtype)
    return tensor.detach().cpu()


def _shape_of(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def build_old_ref_case(case_payload, generalized_result):
    named = case_payload["named_params"]
    runtime_inputs = case_payload["runtime_inputs"]
    outputs = generalized_result["outputs"]
    inplace = generalized_result["inplace"]

    enable_quant_output = bool(
        named["query_quant_mode"] == 1 and named["weight_quant_mode"] in (2, 3, 4, 5)
    )
    flaglist = [0] * 25
    flaglist[19] = int(named["weight_quant_mode"] == 1 and named["kv_quant_mode"] == 2)
    flaglist[20] = int(runtime_inputs["smooth_scales_cq"] is not None)
    flaglist[21] = int(enable_quant_output)
    flaglist[22] = int(named["query_norm_flag"])
    flaglist[23] = int(named["query_norm_flag"])
    flaglist[24] = int(runtime_inputs["actual_seq_len"] is not None)

    actual_seq_value = None
    if runtime_inputs["actual_seq_len"] is not None:
        actual_seq_value = runtime_inputs["actual_seq_len"].detach().cpu().numpy()

    tensor_list = [None] * 21
    tensor_list[0] = _tensor_or_empty(runtime_inputs["token_x"], torch.bfloat16)
    tensor_list[1] = _tensor_or_empty(runtime_inputs["w_dq"], torch.bfloat16)
    tensor_list[2] = _tensor_or_empty(runtime_inputs["w_uq_qr"], torch.bfloat16)
    tensor_list[3] = _tensor_or_empty(runtime_inputs["w_uk"], torch.bfloat16)
    tensor_list[4] = _tensor_or_empty(runtime_inputs["w_dkv_kr"], torch.bfloat16)
    tensor_list[5] = _tensor_or_empty(runtime_inputs["rmsnorm_gamma_cq"], torch.bfloat16)
    tensor_list[6] = _tensor_or_empty(runtime_inputs["rmsnorm_gamma_ckv"], torch.bfloat16)
    tensor_list[7] = _tensor_or_empty(runtime_inputs["rope_sin"], torch.bfloat16)
    tensor_list[8] = _tensor_or_empty(runtime_inputs["rope_cos"], torch.bfloat16)
    tensor_list[9] = _tensor_or_empty(runtime_inputs["cache_index"], torch.int64)
    tensor_list[10] = _tensor_or_empty(runtime_inputs["kv_cache"], torch.bfloat16)
    tensor_list[11] = _tensor_or_empty(runtime_inputs["kr_cache"], torch.bfloat16)
    tensor_list[12] = _tensor_or_empty(runtime_inputs["dequant_scale_x"], torch.float32)
    tensor_list[13] = _tensor_or_empty(runtime_inputs["dequant_scale_w_dq"], torch.float32)
    tensor_list[14] = _tensor_or_empty(runtime_inputs["dequant_scale_w_uq_qr"], torch.float32)
    tensor_list[15] = _tensor_or_empty(runtime_inputs["dequant_scale_w_dkv_kr"], torch.float32)
    tensor_list[16] = _tensor_or_empty(runtime_inputs["quant_scale_ckv"], torch.float32)
    tensor_list[17] = _tensor_or_empty(runtime_inputs["quant_scale_ckr"], torch.float32)
    tensor_list[18] = _tensor_or_empty(runtime_inputs["smooth_scales_cq"], torch.float32)
    tensor_list[19] = _tensor_or_empty(runtime_inputs["actual_seq_len"], torch.int32)
    tensor_list[20] = _tensor_or_empty(runtime_inputs["k_nope_clip_alpha"], torch.float32)

    shape_input = [_shape_of(tensor) for tensor in tensor_list]
    dtype_input = [_dtype_to_old_name(tensor.dtype) for tensor in tensor_list]
    if named["weight_quant_mode"] == prologv3_generalized.WEIGHT_QUANT_MODE_FULL_HIF8:
        for index in (0, 1, 2, 4):
            dtype_input[index] = "hifloat8"
        if named["kv_quant_mode"] in (1, 3):
            dtype_input[10] = "hifloat8"

    shape_output = [
        _shape_of(outputs[0]),
        _shape_of(outputs[1]),
        _shape_of(inplace[0]),
        _shape_of(inplace[1]),
    ]
    dtype_output = [
        _dtype_to_old_name(outputs[0].dtype),
        _dtype_to_old_name(outputs[1].dtype),
        _dtype_to_old_name(inplace[0].dtype),
        _dtype_to_old_name(inplace[1].dtype),
    ]
    if named["weight_quant_mode"] == prologv3_generalized.WEIGHT_QUANT_MODE_FULL_HIF8:
        if enable_quant_output:
            dtype_output[0] = "hifloat8"
        if named["kv_quant_mode"] in (1, 3):
            dtype_output[2] = "hifloat8"
    if flaglist[21]:
        shape_output.append(_shape_of(outputs[2]))
        dtype_output.append(_dtype_to_old_name(outputs[2].dtype))
    if flaglist[22]:
        shape_output.append(_shape_of(outputs[3]))
        dtype_output.append(_dtype_to_old_name(outputs[3].dtype))
        if named["weight_quant_mode"] == prologv3_generalized.WEIGHT_QUANT_MODE_FULL_HIF8:
            dtype_output[-1] = "hifloat8"
    if flaglist[23]:
        shape_output.append(_shape_of(outputs[4]))
        dtype_output.append(_dtype_to_old_name(outputs[4].dtype))

    params = {
        "action_type": "bm_output",
        "flaglist": flaglist,
        "b": named["batch_size"],
        "n1": named["q_head_num"],
        "n2": named["kv_head_num"],
        "d": named["head_dim"],
        "dr": named["rope_head_dim"],
        "s1": named["q_seq"],
        "s2": named["q_seq"],
        "he": named["He"],
        "hcq": named["Hcq"],
        "hckv": named["Hckv"],
        "t": named["batch_size"] * named["q_seq"],
        "block_size": named["block_size"],
        "cache_mode": named["cache_mode"],
        "qnorm_flag": named["query_norm_flag"],
        "shape_input": shape_input,
        "dtype_input": dtype_input,
        "shape_output": shape_output,
        "dtype_output": dtype_output,
        "epsilon_cq": named["cq_epsilon"],
        "epsilon_ckv": named["ckv_epsilon"],
        "weight_quant_mode": named["weight_quant_mode"],
        "kv_quant_mode": named["kv_quant_mode"],
        "query_quant_mode": named["query_quant_mode"],
        "ckvkr_repo_mode": named["ckvkr_repo_mode"],
        "quant_scale_repo_mode": named["quant_scale_repo_mode"],
        "tile_size": named["tile_size"],
        "qc_qr_scale": named["qc_qr_scale"],
        "kc_scale": named["kc_scale"],
        "actual_seq_value": actual_seq_value if actual_seq_value is not None else np.empty((0,), dtype=np.int32),
    }
    return params, tensor_list
