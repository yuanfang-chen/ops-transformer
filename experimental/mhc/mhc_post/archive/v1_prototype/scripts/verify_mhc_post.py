#!/usr/bin/env python3
"""
mhc_post 算子验证脚本

论文公式 (mHC Equation 3):
  x_{l+1} = H_l^{res} * x_l + H_l^{post}^T * F(H_l^{pre} * x_l, W_l)
  
mhc_post 对应的是: H_l^{post}^T * branch_output
  - H_post: [num_streams] 经过softmax归一化的权重向量
  - branch_output: [batch, seq_len, dim] 分支模块输出 F(...)
  - output: [batch * num_streams, seq_len, dim]
  
参考实现 (tokenbender/mHC):
  output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
  output = rearrange(output, "b ... s d -> (b s) ... d")
  
即: output[b*num_streams + s, seq, d] = branch_output[b, seq, d] * h_post[s]
"""

import numpy as np

def softmax(x):
    """Softmax归一化，与论文中H_post的约束一致"""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def mhc_post_reference(branch_output, h_post):
    """
    mhc_post 参考实现 (对应论文公式和PyTorch参考代码)
    
    等价于:
      output = einsum(branch_output, h_post, "b seq d, s -> b seq s d")
      output = rearrange(output, "b seq s d -> (b s) seq d")
    """
    batch, seq_len, dim = branch_output.shape
    num_streams = len(h_post)
    
    # 输出shape: [batch * num_streams, seq_len, dim]
    output = np.zeros((batch * num_streams, seq_len, dim), dtype=np.float32)
    
    for b in range(batch):
        for s in range(num_streams):
            # 论文公式: H_post^T * branch_output
            # 具体计算: output[b*s+i] = branch_output[b] * h_post[i]
            out_idx = b * num_streams + s
            output[out_idx] = branch_output[b] * h_post[s]
    
    return output

def verify_formula():
    """验证公式正确性"""
    print("=" * 60)
    print("mhc_post 公式验证")
    print("=" * 60)
    
    # 小规模数据便于手工验证
    batch, seq_len, dim = 2, 2, 4
    num_streams = 3
    
    # 构造简单输入
    branch_output = np.arange(batch * seq_len * dim, dtype=np.float32).reshape(batch, seq_len, dim)
    h_post_raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    h_post = softmax(h_post_raw)
    
    print(f"\n输入参数:")
    print(f"  batch={batch}, seq_len={seq_len}, dim={dim}, num_streams={num_streams}")
    print(f"\nbranch_output shape: {branch_output.shape}")
    print(f"branch_output[0,0,:] = {branch_output[0,0,:]}")
    print(f"branch_output[1,0,:] = {branch_output[1,0,:]}")
    print(f"\nh_post (softmax归一化): {h_post}")
    print(f"h_post sum = {np.sum(h_post):.6f} (应为1.0)")
    
    # 计算参考结果
    output = mhc_post_reference(branch_output, h_post)
    
    print(f"\n输出 shape: {output.shape}")
    print(f"  期望: ({batch * num_streams}, {seq_len}, {dim})")
    
    # 手工验证几个点
    print(f"\n手工验证:")
    for b in range(batch):
        for s in range(num_streams):
            out_idx = b * num_streams + s
            expected = branch_output[b, 0, 0] * h_post[s]
            actual = output[out_idx, 0, 0]
            match = "✓" if np.isclose(expected, actual) else "✗"
            print(f"  output[{out_idx},0,0] = branch_output[{b},0,0] * h_post[{s}]")
            print(f"    = {branch_output[b,0,0]:.4f} * {h_post[s]:.4f} = {expected:.4f}")
            print(f"    实际值: {actual:.4f} {match}")
    
    return True

def compare_with_einsum():
    """与einsum实现对比验证"""
    print("\n" + "=" * 60)
    print("与 einsum 参考实现对比")
    print("=" * 60)
    
    batch, seq_len, dim = 4, 8, 64
    num_streams = 4
    
    np.random.seed(42)
    branch_output = np.random.randn(batch, seq_len, dim).astype(np.float32)
    h_post = softmax(np.random.randn(num_streams).astype(np.float32))
    
    # 方法1: 参考实现
    output1 = mhc_post_reference(branch_output, h_post)
    
    # 方法2: 使用numpy einsum (等价于PyTorch einsum)
    # "b seq d, s -> b seq s d" 然后 reshape
    output2_tmp = np.einsum('bsd,n->bsnd', branch_output, h_post)
    output2 = output2_tmp.transpose(0, 2, 1, 3).reshape(batch * num_streams, seq_len, dim)
    
    # 对比
    max_diff = np.max(np.abs(output1 - output2))
    print(f"\n最大差异: {max_diff:.2e}")
    print(f"是否一致: {'✓ PASS' if max_diff < 1e-6 else '✗ FAIL'}")
    
    return max_diff < 1e-6

def generate_test_vectors():
    """生成用于NPU算子验证的测试向量"""
    print("\n" + "=" * 60)
    print("生成NPU验证测试向量")
    print("=" * 60)
    
    # 使用与C++测试相同的参数
    batch, seq_len, dim = 2, 4, 64
    num_streams = 4
    
    # 相同的初始化方式
    branch_output = np.array([(i % 100) / 100.0 for i in range(batch * seq_len * dim)], 
                             dtype=np.float32).reshape(batch, seq_len, dim)
    h_post_raw = np.array([float(i + 1) for i in range(num_streams)], dtype=np.float32)
    h_post = softmax(h_post_raw)
    
    print(f"\nh_post weights: {h_post}")
    
    # 计算期望输出
    expected_output = mhc_post_reference(branch_output, h_post)
    
    # 打印部分结果用于对比
    print(f"\n期望输出 (前几个值):")
    for s in range(num_streams):
        print(f"  stream {s}: output[{s},0,:4] = {expected_output[s, 0, :4]}")
    
    # 保存为二进制文件供C++读取
    branch_output.tofile('/root/tst/mhc_post/branch_output.bin')
    h_post.tofile('/root/tst/mhc_post/h_post.bin')
    expected_output.tofile('/root/tst/mhc_post/expected_output.bin')
    
    print(f"\n测试向量已保存到:")
    print(f"  branch_output.bin: {branch_output.shape}")
    print(f"  h_post.bin: {h_post.shape}")
    print(f"  expected_output.bin: {expected_output.shape}")
    
    return h_post, expected_output

if __name__ == '__main__':
    print("\nmhc_post 算子原理验证\n")
    print("论文: mHC: Manifold-Constrained Hyper-Connections (DeepSeek 2024.12.31)")
    print("公式: output = H_post^T ⊗ branch_output")
    print("      output[b*s+i, seq, d] = branch_output[b, seq, d] * h_post[i]")
    
    verify_formula()
    compare_with_einsum()
    generate_test_vectors()
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
