"""Flash Attention 2 AscendC Python 接口

加载编译好的 libfa2_ops.so，提供与 PyTorch + torch_npu 集成的接口。

使用方式:
    from fa2_ops import flash_attention2
    output = flash_attention2(q, k, v, causal=True)
"""
import os
import ctypes
import math
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "build", "lib", "libfa2_ops.so")

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise RuntimeError(
            f"libfa2_ops.so not found at {_LIB_PATH}\n"
            f"Please build first:\n"
            f"  cd {_HERE} && mkdir -p build && cd build && "
            f"source /usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux/bin/setenv.bash && "
            f"cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"
        )
    _lib = ctypes.CDLL(_LIB_PATH)
    _lib.fa2_forward.restype = ctypes.c_int
    _lib.fa2_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
    ]
    _lib.fa2_cleanup.restype = None
    _lib.fa2_cleanup.argtypes = []
    return _lib


def _get_npu_stream():
    """Get the raw stream pointer from the current NPU stream."""
    stream = torch.npu.current_stream()
    # torch_npu exposes stream pointer via .npu_stream or stream_id
    if hasattr(stream, 'npu_stream'):
        return stream.npu_stream
    # Fallback: use the stream's cuda_stream (torch_npu mirrors CUDA API)
    return stream.stream_id


def flash_attention2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float = 0.0,
) -> torch.Tensor:
    """
    Flash Attention 2 forward pass on Ascend NPU.

    Args:
        q: [batch, heads, seq_q,  head_dim]  float16, on NPU
        k: [batch, heads, seq_kv, head_dim]  float16, on NPU
        v: [batch, heads, seq_kv, head_dim]  float16, on NPU
        causal: 是否启用因果遮罩
        softmax_scale: softmax 缩放因子，0 表示自动 1/sqrt(d)

    Returns:
        output: [batch, heads, seq_q, head_dim]  float16, on NPU
    """
    lib = _load_lib()

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), \
        "Q, K, V must be contiguous"
    assert q.dtype == torch.float16, f"Expected float16, got {q.dtype}"
    assert q.device.type == 'npu', "Tensors must be on NPU"
    assert q.dim() == 4, "Expected [B, H, S, D] layout"

    B, H, Sq, D = q.shape
    _, _, Skv, _ = k.shape

    output = torch.empty_like(q)

    stream_ptr = _get_npu_stream()

    ret = lib.fa2_forward(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        B, H, Sq, Skv, D,
        ctypes.c_float(softmax_scale),
        1 if causal else 0,
        ctypes.c_void_p(stream_ptr),
    )
    if ret != 0:
        raise RuntimeError(f"fa2_forward failed with ACL error code {ret}")

    return output


def cleanup():
    """Release device-side tiling/workspace buffers."""
    if _lib is not None:
        _lib.fa2_cleanup()
