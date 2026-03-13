import sys
from pathlib import Path

import pytest
import torch


PYTEST_DIR = Path(__file__).resolve().parents[1]
if str(PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTEST_DIR))

import aclnnMlaPrologV3Ref
import check_valid_param
import prologv3_generalized

from .case_adapter import build_old_ref_case, build_trusted_refnew_case
from . import trusted_refnew_bridge


def _canonicalize_old_result(old_raw_result, case_payload, generalized_result):
    out1, out2, out3, out4, deq_scale_q_nope, query_norm, deq_scale_q_norm = old_raw_result
    outputs = generalized_result["outputs"]
    named = case_payload["named_params"]
    t_flag = bool(named["bs_fused_flag"]) or named["cache_mode"] == "TND"
    batch_size = named["batch_size"]
    q_seq = named["q_seq"]
    hcq = named["Hcq"]

    if deq_scale_q_nope is None:
        deq_scale_q_nope = torch.empty_like(outputs[2])
    if query_norm is None:
        query_norm = torch.empty_like(outputs[3])
    elif query_norm.numel() != 0 and not t_flag and query_norm.ndim == 2:
        query_norm = query_norm.reshape(batch_size, q_seq, hcq)
    if deq_scale_q_norm is None:
        deq_scale_q_norm = torch.empty_like(outputs[4])

    return {
        "outputs": [out1, out2, deq_scale_q_nope, query_norm, deq_scale_q_norm],
        "inplace": [out3, out4],
    }


def run_cpu_ref_alignment_case(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    case_payload, generalized_result = prologv3_generalized.run_prologv3_cpu_only(
        params,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
        validate_quant_combo=validate_quant_combo,
    )
    old_params, torch_tensor_list = build_old_ref_case(case_payload, generalized_result)
    old_mla_param = aclnnMlaPrologV3Ref.get_param(torch_tensor_list, old_params)
    old_mla_param["device"] = "cpu"
    old_raw_result = aclnnMlaPrologV3Ref.cal_mlaprolog(old_mla_param)
    old_result = _canonicalize_old_result(old_raw_result, case_payload, generalized_result)
    return old_result, generalized_result, case_payload


def run_triple_ref_overlap_case(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    case_payload, generalized_result = prologv3_generalized.run_prologv3_cpu_only(
        params,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
        validate_quant_combo=validate_quant_combo,
    )

    old_params, old_tensor_list = build_old_ref_case(case_payload, generalized_result)
    old_mla_param = aclnnMlaPrologV3Ref.get_param(old_tensor_list, old_params)
    old_mla_param["device"] = "cpu"
    old_raw_result = aclnnMlaPrologV3Ref.cal_mlaprolog(old_mla_param)
    old_result = _canonicalize_old_result(old_raw_result, case_payload, generalized_result)

    trusted_params, trusted_tensor_list = build_trusted_refnew_case(case_payload, generalized_result)
    trusted_mla_param = trusted_refnew_bridge.get_param(trusted_tensor_list, trusted_params)
    trusted_mla_param["device"] = "cpu"
    trusted_raw_result = trusted_refnew_bridge.cal_mlaprolog(trusted_mla_param)
    trusted_result = _canonicalize_old_result(trusted_raw_result, case_payload, generalized_result)

    return old_result, trusted_result, generalized_result, case_payload


def run_trusted_refnew_overlap_case(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    case_payload, generalized_result = prologv3_generalized.run_prologv3_cpu_only(
        params,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
        validate_quant_combo=validate_quant_combo,
    )
    trusted_params, torch_tensor_list = build_trusted_refnew_case(case_payload, generalized_result)
    trusted_mla_param = trusted_refnew_bridge.get_param(torch_tensor_list, trusted_params)
    trusted_mla_param["device"] = "cpu"
    trusted_raw_result = trusted_refnew_bridge.cal_mlaprolog(trusted_mla_param)
    trusted_result = _canonicalize_old_result(trusted_raw_result, case_payload, generalized_result)
    return trusted_result, generalized_result, case_payload


def assert_cpu_refs_aligned(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    old_result, generalized_result, case_payload = run_cpu_ref_alignment_case(
        params,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
        validate_quant_combo=validate_quant_combo,
    )
    check_valid_param.check_result(old_result, generalized_result)
    return case_payload


def assert_trusted_refnew_aligned(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    trusted_result, generalized_result, case_payload = run_trusted_refnew_overlap_case(
        params,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
        validate_quant_combo=validate_quant_combo,
    )
    check_valid_param.check_result(trusted_result, generalized_result)
    return case_payload


def assert_overlap_refs_match_trusted(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    old_result, trusted_result, generalized_result, case_payload = run_triple_ref_overlap_case(
        params,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
        validate_quant_combo=validate_quant_combo,
    )
    check_valid_param.check_result(old_result, generalized_result)
    check_valid_param.check_result(trusted_result, generalized_result)
    check_valid_param.check_result(old_result, trusted_result)
    return case_payload


def skip_if_case_unsupported(param_dict):
    weight_quant_mode = param_dict["weight_quant_mode"]
    if weight_quant_mode == prologv3_generalized.WEIGHT_QUANT_MODE_MXFP8_FULL and \
            not prologv3_generalized.is_mxfp8_runtime_supported():
        pytest.skip("mxfp8 CPU alignment requires float8_e8m0 + ml_dtypes support")
