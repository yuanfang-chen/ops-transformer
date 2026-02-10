#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""mhc_res operator: Stream mixing via learned weight matrix.

This module provides PyTorch interface to the AscendC mhc_res kernel.

Usage:
    import mhc_res_ops
    output = mhc_res_ops.mhc_res(input_tensor, h_res_matrix)
"""

import logging

import torch
import torch_npu

logger = logging.getLogger(__name__)

try:
    import mhc_res_ext
    _USE_CPP_EXT = True
except ImportError:
    _USE_CPP_EXT = False
    logger.warning("mhc_res_ext not found. Run 'python setup.py build_ext --inplace' to build.")


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
    if num_streams == 0:
        raise ValueError("num_streams must be > 0, got 0")
    batch = x.size(0) // num_streams
    seq_len = x.size(1)
    dim = x.size(2)
    x_4d = x.view(batch, num_streams, seq_len, dim)
    return torch.einsum('brsd,rt->btsd', x_4d, h_res).reshape(batch * num_streams, seq_len, dim)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.npu.set_device(0)

    batch, seq_len, dim, num_streams = 2, 16, 32, 4
    x = torch.randn(batch * num_streams, seq_len, dim, dtype=torch.float32).npu()
    h = torch.randn(num_streams, num_streams, dtype=torch.float32).npu()

    out_npu = mhc_res(x, h)
    out_ref = mhc_res_einsum(x, h)

    logger.info("Input: %s, h_res: %s", x.shape, h.shape)
    logger.info("Output: %s", out_npu.shape)
    logger.info("Match: %s", torch.allclose(out_npu, out_ref, atol=1e-5))
    logger.info("Max diff: %.2e", (out_npu - out_ref).abs().max().item())
