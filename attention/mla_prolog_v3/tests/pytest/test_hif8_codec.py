import math
import sys
from pathlib import Path

import pytest
import torch


PYTEST_DIR = Path(__file__).resolve().parent
if str(PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTEST_DIR))

import hif8_codec


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, 0.0),
        (1, 2.0 ** -22),
        (8, 1.0),
        (15, 1.875),
        (16, 2.0),
    ],
)
def test_hif8_decode_representative_values(code, expected):
    decoded = hif8_codec.cvt_hifuint8_to_float(code, over_mode=True)
    assert decoded == pytest.approx(expected)


def test_hif8_decode_special_values():
    assert math.isinf(hif8_codec.cvt_hifuint8_to_float(hif8_codec.HIF8_POS_INF_CODE, over_mode=True))
    assert hif8_codec.cvt_hifuint8_to_float(hif8_codec.HIF8_POS_INF_CODE, over_mode=True) > 0
    assert math.isinf(hif8_codec.cvt_hifuint8_to_float(hif8_codec.HIF8_NEG_INF_CODE, over_mode=True))
    assert hif8_codec.cvt_hifuint8_to_float(hif8_codec.HIF8_NEG_INF_CODE, over_mode=True) < 0
    assert math.isnan(hif8_codec.cvt_hifuint8_to_float(hif8_codec.HIF8_NAN_CODE, over_mode=True))


def test_hif8_native_roundtrip_exact_values():
    codes = torch.tensor([0, 1, 8, 15, 16, 23, 31, 47, 79, 143, 151, 159, 175], dtype=torch.uint8)
    decoded = torch.from_numpy(hif8_codec.decode_hif8_uint8_tensor_to_float32(codes, over_mode=True))
    roundtrip = hif8_codec.quantize_tensor_to_hif8_native_float32(decoded, round_mode="hybrid", over_mode=True)
    assert torch.allclose(roundtrip, decoded.to(torch.float32), equal_nan=True)


def test_hif8_overflow_maps_to_inf():
    values = torch.tensor([-1.0e9, -50000.0, 50000.0, 1.0e9], dtype=torch.float32)
    quantized = hif8_codec.quantize_tensor_to_hif8_native_float32(values, round_mode="hybrid", over_mode=True)
    assert torch.isneginf(quantized[:2]).all()
    assert torch.isposinf(quantized[2:]).all()


def test_hif8_nan_roundtrip():
    values = torch.tensor([0.0, float("nan"), 1.0], dtype=torch.float32)
    quantized = hif8_codec.quantize_tensor_to_hif8_native_float32(values, round_mode="hybrid", over_mode=True)
    assert quantized[0].item() == pytest.approx(0.0)
    assert math.isnan(quantized[1].item())
    assert quantized[2].item() == pytest.approx(1.0)
