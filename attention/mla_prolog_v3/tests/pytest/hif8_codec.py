import math
from functools import lru_cache

import numpy as np
import torch


HIF8_DTYPE_MAX = 32768.0
HIF8_NAN_CODE = np.uint8(128)
HIF8_POS_INF_CODE = np.uint8(111)
HIF8_NEG_INF_CODE = np.uint8(239)
_HIF8_POS_INF_ROUND_VALUE = np.float32(1.5 * (2 ** 15))


def _get_hif8_fraction_bits_number(exponent):
    if exponent < -22:
        return -1, 3, 0
    if -22 <= exponent < -15:
        return 0, 3, 0
    if exponent == 0:
        return 1, 0, 3
    if abs(exponent) == 1:
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        return 12, 4, 1
    if exponent > 15:
        return 12, 4, -1
    raise ValueError(f"unsupported exponent for hif8: {exponent}")


def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    if hif8_value_tmp == 0:
        return False, 0
    if hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    return False, hif8_value_tmp >> 1


def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = 16383
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (23 - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (23 - hif8_bits_num))
        f14_values = f14_t14 >> (23 - hif8_bits_num - 14)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        return False, hif8_value + 1
    return False, hif8_value


def _fp16_ta_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    hif8_value_tmp = fraction16_int >> (10 - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    if hif8_value_tmp == 0:
        return False, 0
    if hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    return False, hif8_value_tmp >> 1


def _fp16_ssr_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
    t2_mask = 1
    t2_values = (fraction16_int & t2_mask) * 2 + 1
    if exponent == -23:
        f2_values = 2 + (fraction16_int >> 9)
        hif8_value = 0
    else:
        hif8_value = fraction16_int >> (10 - hif8_bits_num)
        f2_t2 = fraction16_int - (hif8_value << (10 - hif8_bits_num))
        f2_values = f2_t2 >> (10 - hif8_bits_num - 2)
    if f2_values >= t2_values:
        if hif8_value == pow(2, hif8_bits_num):
            return True, 0
        return False, hif8_value + 1
    return False, hif8_value


def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    if x == 128:
        return np.nan if over_mode else float(0)
    if x == 239:
        return -np.inf if over_mode else -32768.0
    if x == 111:
        return np.inf if over_mode else 32768.0

    sign = -1.0 if x >= 128 else 1.0
    dot_4_value = (x & 120) >> 3
    if dot_4_value >= 12:
        exponent_int = (x & 30) >> 1
        exponent_value = -exponent_int if exponent_int >= 8 else exponent_int + 8
        m_value = 1.0 + (x & 1) * 0.5
    elif dot_4_value >= 8:
        exponent_int = (x & 28) >> 2
        exponent_value = -exponent_int if exponent_int >= 4 else exponent_int + 4
        m_value = 1.0 + (x & 3) * 0.25
    elif dot_4_value >= 4:
        exponent_int = (x & 24) >> 3
        exponent_value = -exponent_int if exponent_int >= 2 else exponent_int + 2
        m_value = 1.0 + (x & 7) * 0.125
    elif dot_4_value >= 2:
        exponent_value = -1 if ((x & 8) >> 3) >= 1 else 1
        m_value = 1.0 + (x & 7) * 0.125
    elif dot_4_value == 1:
        exponent_value = 0
        m_value = 1.0 + (x & 7) * 0.125
    elif dot_4_value == 0:
        exponent_value = (x & 7) - 23
        m_value = 1.0
    else:
        return float(0)
    return sign * pow(2.0, exponent_value) * m_value


@lru_cache(maxsize=2)
def _decode_table(over_mode):
    return np.array([cvt_hifuint8_to_float(i, over_mode=over_mode) for i in range(256)], dtype=np.float32)


@lru_cache(maxsize=1)
def _positive_round_values_and_codes():
    decode_table = _decode_table(True)
    positive_codes = []
    positive_values = []
    for code in range(128):
        value = decode_table[code]
        if np.isfinite(value) and value >= 0:
            positive_codes.append(code)
            positive_values.append(float(value))
    positive_codes.append(int(HIF8_POS_INF_CODE))
    positive_values.append(float(_HIF8_POS_INF_ROUND_VALUE))
    order = np.argsort(np.asarray(positive_values, dtype=np.float64))
    values = np.asarray(positive_values, dtype=np.float32)[order]
    codes = np.asarray(positive_codes, dtype=np.uint8)[order]
    return values, codes


def decode_hif8_uint8_tensor_to_float32(in_tensor, over_mode=True):
    if torch.is_tensor(in_tensor):
        array = in_tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        array = np.asarray(in_tensor, dtype=np.uint8)
    out = _decode_table(over_mode)[array.reshape(-1)].reshape(array.shape).astype(np.float32, copy=False)
    return out


def _encode_float_array_to_hif8_uint8(array, round_mode="hybrid", over_mode=True):
    if round_mode != "hybrid":
        raise ValueError(f"unsupported hif8 round_mode: {round_mode}")

    values = np.asarray(array, dtype=np.float32).reshape(-1)
    out = np.zeros(values.shape, dtype=np.uint8)

    nan_mask = np.isnan(values)
    pos_inf_mask = np.isposinf(values)
    neg_inf_mask = np.isneginf(values)
    finite_mask = np.isfinite(values)

    out[nan_mask] = HIF8_NAN_CODE
    out[pos_inf_mask] = HIF8_POS_INF_CODE
    out[neg_inf_mask] = HIF8_NEG_INF_CODE

    if not np.any(finite_mask):
        return out.reshape(array.shape)

    finite_values = values[finite_mask]
    sign_mask = np.signbit(finite_values)
    abs_values = np.abs(finite_values).astype(np.float32, copy=False)

    positive_round_values, positive_codes = _positive_round_values_and_codes()
    idx = np.searchsorted(positive_round_values, abs_values, side="left")
    idx = np.clip(idx, 0, positive_round_values.shape[0] - 1)
    left_idx = np.clip(idx - 1, 0, positive_round_values.shape[0] - 1)
    right_idx = idx

    left_values = positive_round_values[left_idx]
    right_values = positive_round_values[right_idx]
    choose_right = (abs_values - left_values) >= (right_values - abs_values)
    chosen_idx = np.where(choose_right, right_idx, left_idx)
    chosen_codes = positive_codes[chosen_idx].astype(np.uint8, copy=False)

    negative_codes = np.where(chosen_codes == 0, 0, chosen_codes | 0x80).astype(np.uint8, copy=False)
    out[finite_mask] = np.where(sign_mask, negative_codes, chosen_codes)

    if not over_mode:
        out = np.where(out == HIF8_POS_INF_CODE, np.uint8(110), out)
        out = np.where(out == HIF8_NEG_INF_CODE, np.uint8(238), out)

    return out.reshape(array.shape)


def encode_float_tensor_to_hif8_uint8(in_tensor, round_mode="hybrid", over_mode=True):
    if torch.is_tensor(in_tensor):
        array = in_tensor.detach().cpu().to(torch.float32).numpy()
    else:
        array = np.asarray(in_tensor, dtype=np.float32)
    return _encode_float_array_to_hif8_uint8(array, round_mode=round_mode, over_mode=over_mode)


def quantize_tensor_to_hif8_native_float32(in_tensor, round_mode="hybrid", over_mode=True):
    encoded = encode_float_tensor_to_hif8_uint8(in_tensor, round_mode=round_mode, over_mode=over_mode)
    decoded = decode_hif8_uint8_tensor_to_float32(encoded, over_mode=over_mode)
    if torch.is_tensor(in_tensor):
        return torch.from_numpy(decoded).to(torch.float32)
    return decoded


def ensure_hif8_native_float32_tensor(in_tensor, round_mode="hybrid", over_mode=True):
    if in_tensor is None:
        return None
    return quantize_tensor_to_hif8_native_float32(in_tensor, round_mode=round_mode, over_mode=over_mode)
