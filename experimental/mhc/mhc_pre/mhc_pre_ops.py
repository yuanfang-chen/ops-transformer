#!/usr/bin/env python3
"""mhc_pre operator: Reduce N streams to 1 via weighted sum.

This module provides PyTorch interface to the AscendC mhc_pre kernel.

Usage:
    import mhc_pre_ops
    output = mhc_pre_ops.mhc_pre(input_tensor, h_pre_weights)
"""

import torch
import torch_npu

# Try to import C++ extension (built via setup.py)
try:
    import mhc_pre_ext
    _USE_CPP_EXT = True
except ImportError:
    _USE_CPP_EXT = False
    print("Warning: mhc_pre_ext not found. Run 'python setup.py build_ext --inplace' to build.")


def mhc_pre(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Reduce N streams to 1 via weighted sum.
    
    Mathematical operation:
        out[b, s, d] = sum_n(x[b*N + n, s, d] * h_pre[n])
    
    Equivalent einsum:
        x_4d = x.view(batch, num_streams, seq_len, dim)
        out = torch.einsum('bnsd,n->bsd', x_4d, h_pre)
    
    Args:
        x: Input tensor [batch * num_streams, seq_len, dim]
        h_pre: Weight tensor [num_streams]
        
    Returns:
        Output tensor [batch, seq_len, dim]
    """
    if not _USE_CPP_EXT:
        raise RuntimeError("mhc_pre_ext not available. Build with setup.py first.")
    
    return mhc_pre_ext.forward(x.contiguous(), h_pre.contiguous())


def mhc_pre_einsum(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.einsum."""
    num_streams = h_pre.size(0)
    batch = x.size(0) // num_streams
    seq_len = x.size(1)
    dim = x.size(2)
    x_4d = x.view(batch, num_streams, seq_len, dim)
    return torch.einsum('bnsd,n->bsd', x_4d, h_pre)


if __name__ == '__main__':
    # Quick test
    torch.npu.set_device(0)
    
    batch, seq_len, dim, num_streams = 2, 16, 32, 4
    x = torch.randn(batch * num_streams, seq_len, dim, dtype=torch.float32).npu()
    h = torch.randn(num_streams, dtype=torch.float32).npu()
    
    out_npu = mhc_pre(x, h)
    out_ref = mhc_pre_einsum(x, h)
    
    print(f"Input: {x.shape}, h_pre: {h.shape}")
    print(f"Output: {out_npu.shape}")
    print(f"Match: {torch.allclose(out_npu, out_ref, atol=1e-5)}")
    print(f"Max diff: {(out_npu - out_ref).abs().max().item():.2e}")
