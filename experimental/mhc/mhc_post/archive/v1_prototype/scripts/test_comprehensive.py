#!/usr/bin/env python3
# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
"""
mhc_post 综合验证: 对照论文公式 + PyTorch参考实现 + 多种shape测试
"""

import numpy as np
import logging
import struct
import os
import subprocess

logger = logging.getLogger(__name__)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def mhc_post_numpy(branch_output, h_post):
    """
    NumPy参考实现
    等价于PyTorch: einsum(branch_output, h_post, "b ... d, s -> b ... s d")
                     rearrange(output, "b ... s d -> (b s) ... d")
    """
    batch, seq_len, dim = branch_output.shape
    num_streams = len(h_post)
    output = np.zeros((batch * num_streams, seq_len, dim), dtype=np.float32)
    
    for b in range(batch):
        for s in range(num_streams):
            out_idx = b * num_streams + s
            output[out_idx] = branch_output[b] * h_post[s]
    
    return output

def mhc_post_einsum(branch_output, h_post):
    """NumPy einsum实现 - 模拟PyTorch参考代码"""
    batch, seq_len, dim = branch_output.shape
    num_streams = len(h_post)
    
    # einsum: "b seq d, s -> b seq s d"
    tmp = np.einsum('bsd,n->bsnd', branch_output, h_post)
    # rearrange: "b seq s d -> (b s) seq d"
    output = tmp.transpose(0, 2, 1, 3).reshape(batch * num_streams, seq_len, dim)
    
    return output

def run_npu_kernel(branch_output, h_post, batch, seq_len, dim, num_streams):
    """NPU kernel调用包装"""
    # 保存输入
    branch_output.astype(np.float32).tofile('/tmp/mhc_input.bin')
    h_post.astype(np.float32).tofile('/tmp/mhc_hpost.bin')
    
    # 调用NPU kernel
    cpp_code = f'''
#include <iostream>
#include <fstream>
#include <vector>
#include "acl/acl.h"

extern "C" void mhc_post_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t, int64_t, int64_t);

int main() {{
    int64_t batch={batch}, seq_len={seq_len}, dim={dim}, num_streams={num_streams};
    int64_t in_size = batch * seq_len * dim;
    int64_t out_size = batch * num_streams * seq_len * dim;
    
    std::vector<float> input(in_size), hpost(num_streams), output(out_size);
    
    std::ifstream f1("/tmp/mhc_input.bin", std::ios::binary);
    f1.read((char*)input.data(), in_size * sizeof(float));
    std::ifstream f2("/tmp/mhc_hpost.bin", std::ios::binary);
    f2.read((char*)hpost.data(), num_streams * sizeof(float));
    
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream stream; aclrtCreateStream(&stream);
    
    void *d_in, *d_hp, *d_out;
    aclrtMalloc(&d_in, in_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_hp, num_streams * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_out, out_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtMemcpy(d_in, in_size*4, input.data(), in_size*4, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_hp, num_streams*4, hpost.data(), num_streams*4, ACL_MEMCPY_HOST_TO_DEVICE);
    
    mhc_post_do(batch * num_streams, stream, (uint8_t*)d_in, (uint8_t*)d_hp, (uint8_t*)d_out,
                batch, seq_len, dim, num_streams);
    aclrtSynchronizeStream(stream);
    
    aclrtMemcpy(output.data(), out_size*4, d_out, out_size*4, ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::ofstream fout("/tmp/mhc_output.bin", std::ios::binary);
    fout.write((char*)output.data(), out_size * sizeof(float));
    
    aclrtFree(d_in); aclrtFree(d_hp); aclrtFree(d_out);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}}
'''
    
    with open('/tmp/test_mhc_run.cpp', 'w') as f:
        f.write(cpp_code)
    
    # 编译和运行
    compile_cmd = '''
    compile_cmd = [
        'g++', '-o', '/tmp/test_mhc_run', '/tmp/test_mhc_run.cpp',
        '-I/usr/local/Ascend/ascend-toolkit/latest/include',
        '-L/usr/local/Ascend/ascend-toolkit/latest/lib64',
        '-L/root/tst/mhc_post/build/lib',
        '-lmhc_post_kernel', '-lascendcl', '-lruntime',
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True,
                            cwd='/root/tst/mhc_post/build')
    if result.returncode != 0:
        logger.error("Compile error: %s", result.stderr)
        return None

    run_cmd = ['/tmp/test_mhc_run']
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Run error: %s", result.stderr)
        return None
    
    # 读取输出
    output = np.fromfile('/tmp/mhc_output.bin', dtype=np.float32)
    return output.reshape(batch * num_streams, seq_len, dim)

def test_case(name, batch, seq_len, dim, num_streams):
    """单个测试用例"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"  batch={batch}, seq_len={seq_len}, dim={dim}, num_streams={num_streams}")
    print(f"{'='*60}")
    
    # 生成测试数据
    np.random.seed(42)
    branch_output = np.random.randn(batch, seq_len, dim).astype(np.float32)
    h_post_raw = np.random.randn(num_streams).astype(np.float32)
    h_post = softmax(h_post_raw)
    
    print(f"\nh_post (softmax归一化): {h_post}")
    print(f"h_post sum = {np.sum(h_post):.6f}")
    
    # 计算参考结果
    ref_loop = mhc_post_numpy(branch_output, h_post)
    ref_einsum = mhc_post_einsum(branch_output, h_post)
    
    # 验证两种参考实现一致
    diff_ref = np.max(np.abs(ref_loop - ref_einsum))
    print(f"\n参考实现一致性检查: max_diff={diff_ref:.2e} {'✓' if diff_ref < 1e-6 else '✗'}")
    
    # 运行NPU kernel
    npu_output = run_npu_kernel(branch_output, h_post, batch, seq_len, dim, num_streams)
    
    if npu_output is None:
        print("NPU执行失败")
        return False
    
    # 精度对比
    diff_npu = np.abs(ref_loop - npu_output)
    max_abs_err = np.max(diff_npu)
    max_rel_err = np.max(diff_npu / (np.abs(ref_loop) + 1e-8))
    avg_err = np.mean(diff_npu)
    
    print(f"\nNPU vs 参考实现:")
    print(f"  最大绝对误差: {max_abs_err:.2e}")
    print(f"  最大相对误差: {max_rel_err:.2e}")
    print(f"  平均绝对误差: {avg_err:.2e}")
    
    # 抽样对比
    print(f"\n抽样对比 (前4个输出batch):")
    for s in range(min(4, num_streams)):
        print(f"  stream {s}: ref={ref_loop[s,0,:4]}, npu={npu_output[s,0,:4]}")
    
    passed = max_abs_err < 1e-4 and max_rel_err < 1e-3
    print(f"\n测试结果: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed

def main():
    print("\n" + "#"*60)
    print("# mhc_post 算子综合验证")
    print("# 论文: mHC: Manifold-Constrained Hyper-Connections")
    print("# 公式: output[b*s+i] = branch_output[b] * h_post[i]")
    print("#"*60)
    
    results = []
    
    # 测试用例
    test_cases = [
        ("基础测试", 2, 4, 64, 4),
        ("单batch", 1, 8, 128, 4),
        ("大batch", 8, 4, 64, 4),
        ("长序列", 2, 32, 64, 4),
        ("大dim", 2, 4, 256, 4),
        ("多streams", 2, 4, 64, 8),
        ("典型配置", 4, 16, 128, 4),
    ]
    
    for name, batch, seq_len, dim, num_streams in test_cases:
        passed = test_case(name, batch, seq_len, dim, num_streams)
        results.append((name, passed))
    
    # 汇总
    print("\n" + "#"*60)
    print("# 测试汇总")
    print("#"*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n总体结果: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())
