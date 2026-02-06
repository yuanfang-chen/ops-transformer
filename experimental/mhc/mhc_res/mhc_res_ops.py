#!/usr/bin/env python3
"""mhc_res operator: Stream mixing via learned weight matrix.

This module provides PyTorch interface to the AscendC mhc_res kernel.

Usage:
    import mhc_res_ops
    output = mhc_res_ops.mhc_res(input_tensor, h_res_matrix)
"""

import torch
import torch_npu

# Try to import C++ extension (built via setup.py)
try:
    import mhc_res_ext
    _USE_CPP_EXT = True
except ImportError:
    _USE_CPP_EXT = False
    print("Warning: mhc_res_ext not found. Run 'python setup.py build_ext --inplace' to build.")


def mhc_res(x: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """Mix streams via learned weight matrix (residual path).
    
    Mathematical operation:
        out[b*N + t, s, d] = sum_r(h_res[r, t] * x[b*N + r, s, d])
    
    Equivalent einsum:
        x_4d = x.view(batch, num_streams, seq_len, dim)
        out = torch.einsum('brsd,rt->btsd', x_4d, h_res).reshape(B*N, S, D)
    
    Args:
        x: Input tensor [batch * num_streams, seq_len, dim]
        h_res: Weight matrix [num_streams, num_streams]
        
    Returns:
        Output tensor [batch * num_streams, seq_len, dim]
    """
    if not _USE_CPP_EXT:
        raise RuntimeError("mhc_res_ext not available. Build with setup.py first.")
    
    return mhc_res_ext.forward(x.contiguous(), h_res.contiguous())


def mhc_res_einsum(x: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.einsum."""
    num_streams = h_res.size(0)
    batch = x.size(0) // num_streams
    seq_len = x.size(1)
    dim = x.size(2)
    x_4d = x.view(batch, num_streams, seq_len, dim)
    return torch.einsum('brsd,rt->btsd', x_4d, h_res).reshape(batch * num_streams, seq_len, dim)


if __name__ == '__main__':
    # Quick test
    torch.npu.set_device(0)
    
    batch, seq_len, dim, num_streams = 2, 16, 32, 4
    x = torch.randn(batch * num_streams, seq_len, dim, dtype=torch.float32).npu()
    h = torch.randn(num_streams, num_streams, dtype=torch.float32).npu()
    
    out_npu = mhc_res(x, h)
    out_ref = mhc_res_einsum(x, h)
    
    print(f"Input: {x.shape}, h_res: {h.shape}")
    print(f"Output: {out_npu.shape}")
    print(f"Match: {torch.allclose(out_npu, out_ref, atol=1e-5)}")
    print(f"Max diff: {(out_npu - out_ref).abs().max().item():.2e}")
