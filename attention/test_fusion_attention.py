# -*- coding: utf-8 -*-
"""
Flash Attention Fusion 算子测试脚本

测试 torch_npu.npu_fusion_attention 算子的功能：
1. 基础功能测试 - 验证算子能否正确调用
2. 精度测试 - 与PyTorch原生实现对比
3. 不同配置测试 - 测试不同sparse_mode、数据类型等
"""

import math
import torch
import torch_npu
import numpy as np
from typing import List, Tuple, Optional


class FusionAttentionTester:
    """Flash Attention Fusion 算子测试类"""
    
    def __init__(self, device: str = "npu"):
        self.device = device
        self.prec = 1e-3  # float32精度阈值
        self.prec16 = 1e-2  # float16精度阈值
        
    def cpu_attention(self, query: torch.Tensor, key: torch.Tensor, 
                      value: torch.Tensor, atten_mask: Optional[torch.Tensor] = None,
                      scale: float = 1.0) -> torch.Tensor:
        """
        CPU上的标准Attention实现，用于对比验证
        
        Args:
            query: [B, N, S, D] 或 [B, S, N, D]
            key: [B, N, S, D] 或 [B, S, N, D]  
            value: [B, N, S, D] 或 [B, S, N, D]
            atten_mask: 注意力掩码
            scale: 缩放系数
            
        Returns:
            attention输出
        """
        # QK^T
        qk = torch.matmul(query, key.transpose(-2, -1)).mul(scale)
        
        # 添加mask
        if atten_mask is not None:
            qk = qk + atten_mask * (-10000.0)
            
        # Softmax
        softmax_res = torch.nn.functional.softmax(qk, dim=-1)
        
        # attention * V
        attention_out = torch.matmul(softmax_res, value)
        
        return attention_out
    
    def get_atten_mask(self, shape: List[int], sparse_mode: int = 0,
                       pre_tokens: int = 65536, next_tokens: int = 65536) -> torch.Tensor:
        """
        生成注意力掩码
        
        Args:
            shape: 掩码shape [B, N, S, S] 或 [S, S]
            sparse_mode: 稀疏模式
            pre_tokens: 前向token数
            next_tokens: 后向token数
            
        Returns:
            atten_mask tensor
        """
        if len(shape) == 4:
            B, N, S1, S2 = shape
        else:
            S1, S2 = shape
            B, N = 1, 1
            
        atten_mask_np = None
        
        if sparse_mode == 0:
            # defaultMask模式
            mask = np.ones((S1, S2), dtype=np.float32)
            atten_mask_u = np.triu(mask, k=next_tokens + 1)
            atten_mask_l = np.tril(mask, k=-pre_tokens - 1)
            atten_mask_np = atten_mask_u + atten_mask_l
            
        elif sparse_mode == 1:
            # allMask模式 - 全0表示全计算
            atten_mask_np = np.zeros((S1, S2), dtype=np.float32)
            
        elif sparse_mode == 2:
            # leftUpCausal模式
            atten_mask_np = np.triu(np.ones((S1, S2), dtype=np.float32), k=1)
            
        elif sparse_mode == 3:
            # rightDownCausal模式
            atten_mask_np = np.triu(np.ones((S1, S2), dtype=np.float32), k=1)
            
        elif sparse_mode == 4:
            # band模式
            mask = np.ones((S1, S2), dtype=np.float32)
            atten_mask_u = np.triu(mask, k=next_tokens + 1)
            atten_mask_l = np.tril(mask, k=-pre_tokens - 1)
            atten_mask_np = atten_mask_u + atten_mask_l
            
        else:
            atten_mask_np = np.zeros((S1, S2), dtype=np.float32)
            
        # 转换为tensor并扩展到4维
        atten_mask = torch.from_numpy(atten_mask_np).to(self.device)
        
        if len(shape) == 4:
            atten_mask = atten_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            
        return atten_mask
    
    def test_basic_functionality(self) -> bool:
        """测试基础功能 - 验证算子能否正确调用"""
        print("\n" + "="*60)
        print("测试1: 基础功能测试")
        print("="*60)
        
        try:
            # 基本参数配置
            B, S, N, D = 1, 256, 8, 32  # batch, seq, head, dim
            scale = 1.0 / math.sqrt(D)
            
            # 创建输入数据
            query = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
            key = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
            value = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
            
            print(f"输入shape: query={query.shape}, key={key.shape}, value={value.shape}")
            print(f"scale={scale}, head_num={N}, input_layout=BNSD")
            
            # 调用fusion_attention算子
            result = torch_npu.npu_fusion_attention(
                query, key, value,
                head_num=N,
                input_layout="BNSD",
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0
            )
            
            # 解析返回值
            attention_out, softmax_max, softmax_sum, _, seed, offset, mask_len = result
            
            print(f"\n输出shape: {attention_out.shape}")
            print(f"softmax_max shape: {softmax_max.shape}")
            print(f"softmax_sum shape: {softmax_sum.shape}")
            print(f"seed={seed}, offset={offset}, mask_len={mask_len}")
            
            # 检查输出是否包含有效数据
            assert attention_out.shape == (B, N, S, D), f"输出shape不匹配: {attention_out.shape}"
            assert not torch.isnan(attention_out).any(), "输出包含NaN"
            assert not torch.isinf(attention_out).any(), "输出包含Inf"
            
            print("\n[✓] 基础功能测试通过!")
            return True
            
        except Exception as e:
            print(f"\n[✗] 基础功能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_precision(self) -> bool:
        """测试精度 - 与CPU实现对比"""
        print("\n" + "="*60)
        print("测试2: 精度测试")
        print("="*60)
        
        try:
            B, S, N, D = 1, 128, 8, 32
            scale = 1.0 / math.sqrt(D)
            
            # 创建相同种子以保证数据一致
            torch.manual_seed(42)
            query_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
            key_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
            value_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
            
            # CPU参考实现
            atten_mask = self.get_atten_mask([B, N, S, S], sparse_mode=0)
            cpu_output = self.cpu_attention(query_cpu, key_cpu, value_cpu, atten_mask, scale)
            
            # NPU实现
            query_npu = query_cpu.half().to(self.device)
            key_npu = key_cpu.half().to(self.device)
            value_npu = value_cpu.half().to(self.device)
            atten_mask_npu = atten_mask.to(self.device)
            
            result = torch_npu.npu_fusion_attention(
                query_npu, key_npu, value_npu,
                head_num=N,
                input_layout="BNSD",
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0,
                atten_mask=atten_mask_npu
            )
            
            npu_output = result[0].float()  # 转换为float32进行比较
            
            # 计算相对误差
            diff = torch.abs(cpu_output - npu_output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"最大绝对误差: {max_diff:.6f}")
            print(f"平均绝对误差: {mean_diff:.6f}")
            
            # 检查精度
            if max_diff < self.prec16:
                print(f"\n[✓] 精度测试通过! (max_diff={max_diff:.6f} < {self.prec16})")
                return True
            else:
                print(f"\n[✗] 精度测试失败! (max_diff={max_diff:.6f} >= {self.prec16})")
                return False
                
        except Exception as e:
            print(f"\n[✗] 精度测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_different_sparse_modes(self) -> bool:
        """测试不同sparse_mode"""
        print("\n" + "="*60)
        print("测试3: 不同sparse_mode测试")
        print("="*60)
        
        B, S, N, D = 1, 256, 8, 32
        scale = 1.0 / math.sqrt(D)
        
        sparse_modes = [0, 1, 2, 3, 4]
        results = []
        
        for sparse_mode in sparse_modes:
            try:
                print(f"\n测试 sparse_mode={sparse_mode}...", end=" ")
                
                query = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
                key = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
                value = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
                
                atten_mask = self.get_atten_mask([B, N, S, S], sparse_mode=sparse_mode)
                
                result = torch_npu.npu_fusion_attention(
                    query, key, value,
                    head_num=N,
                    input_layout="BNSD",
                    scale=scale,
                    keep_prob=1.0,
                    sparse_mode=sparse_mode,
                    atten_mask=atten_mask
                )
                
                attention_out = result[0]
                
                assert not torch.isnan(attention_out).any(), f"sparse_mode={sparse_mode} 输出包含NaN"
                assert not torch.isinf(attention_out).any(), f"sparse_mode={sparse_mode} 输出包含Inf"
                
                print(f"✓ shape={attention_out.shape}")
                results.append((sparse_mode, True, ""))
                
            except Exception as e:
                print(f"✗ 错误: {e}")
                results.append((sparse_mode, False, str(e)))
                
        # 汇总结果
        success_count = sum(1 for _, success, _ in results if success)
        print(f"\n成功: {success_count}/{len(sparse_modes)}")
        
        if success_count == len(sparse_modes):
            print("[✓] 所有sparse_mode测试通过!")
            return True
        else:
            print("[✗] 部分sparse_mode测试失败:")
            for mode, success, error in results:
                if not success:
                    print(f"  - sparse_mode={mode}: {error}")
            return False
    
    def test_different_dtypes(self) -> bool:
        """测试不同数据类型"""
        print("\n" + "="*60)
        print("测试4: 不同数据类型测试")
        print("="*60)
        
        B, S, N, D = 1, 128, 8, 32
        scale = 1.0 / math.sqrt(D)
        
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        results = []
        
        for dtype in dtypes:
            try:
                print(f"\n测试 dtype={dtype}...", end=" ")
                
                query = torch.randn(B, N, S, D, dtype=dtype).to(self.device)
                key = torch.randn(B, N, S, D, dtype=dtype).to(self.device)
                value = torch.randn(B, N, S, D, dtype=dtype).to(self.device)
                
                result = torch_npu.npu_fusion_attention(
                    query, key, value,
                    head_num=N,
                    input_layout="BNSD",
                    scale=scale,
                    keep_prob=1.0,
                    sparse_mode=0
                )
                
                attention_out = result[0]
                
                assert not torch.isnan(attention_out).any(), f"dtype={dtype} 输出包含NaN"
                assert not torch.isinf(attention_out).any(), f"dtype={dtype} 输出包含Inf"
                
                print(f"✓ shape={attention_out.shape}, dtype={attention_out.dtype}")
                results.append((dtype, True, ""))
                
            except Exception as e:
                print(f"✗ 错误: {e}")
                results.append((dtype, False, str(e)))
                
        success_count = sum(1 for _, success, _ in results if success)
        print(f"\n成功: {success_count}/{len(dtypes)}")
        
        if success_count == len(dtypes):
            print("[✓] 所有数据类型测试通过!")
            return True
        else:
            print("[✗] 部分数据类型测试失败:")
            for dtype, success, error in results:
                if not success:
                    print(f"  - dtype={dtype}: {error}")
            return False
    
    def test_backward(self) -> bool:
        """测试反向传播"""
        print("\n" + "="*60)
        print("测试5: 反向传播测试")
        print("="*60)
        
        try:
            B, S, N, D = 1, 128, 8, 32
            scale = 1.0 / math.sqrt(D)
            
            # 创建可训练的输入
            query = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(self.device)
            key = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(self.device)
            value = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(self.device)
            
            # 调用fusion_attention
            result = torch_npu.npu_fusion_attention(
                query, key, value,
                head_num=N,
                input_layout="BNSD",
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0
            )
            
            attention_out = result[0]
            
            # 反向传播
            loss = attention_out.sum()
            loss.backward()
            
            # 检查梯度是否计算
            assert query.grad is not None, "query梯度为None"
            assert key.grad is not None, "key梯度为None"
            assert value.grad is not None, "value梯度为None"
            
            assert not torch.isnan(query.grad).any(), "query梯度包含NaN"
            assert not torch.isnan(key.grad).any(), "key梯度包含NaN"
            assert not torch.isnan(value.grad).any(), "value梯度包含NaN"
            
            print(f"query梯度shape: {query.grad.shape}")
            print(f"key梯度shape: {key.grad.shape}")
            print(f"value梯度shape: {value.grad.shape}")
            
            print("\n[✓] 反向传播测试通过!")
            return True
            
        except Exception as e:
            print(f"\n[✗] 反向传播测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        print("\n" + "="*60)
        print("  Fusion Attention 算子测试套件")
        print("="*60)
        
        tests = [
            ("基础功能测试", self.test_basic_functionality),
            ("精度测试", self.test_precision),
            ("不同sparse_mode测试", self.test_different_sparse_modes),
            ("不同数据类型测试", self.test_different_dtypes),
            ("反向传播测试", self.test_backward),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
            except Exception as e:
                print(f"\n[✗] {name} 异常: {e}")
                results.append((name, False))
        
        # 汇总结果
        print("\n" + "="*60)
        print("  测试结果汇总")
        print("="*60)
        
        for name, result in results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {name}: {status}")
            
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        print(f"\n总计: {passed}/{total} 测试通过")
        
        if passed == total:
            print("\n[✓] 所有测试通过!")
            return True
        else:
            print(f"\n[✗] {total - passed} 个测试失败")
            return False


def main():
    """主函数"""
    # 检查NPU设备
    if not torch_npu.npu.is_available():
        print("[警告] NPU不可用，将使用CPU进行测试")
        device = "cpu"
    else:
        device = "npu"
        print(f"[信息] 使用NPU设备: {torch_npu.npu.get_device_name(0)}")
    
    # 创建测试实例
    tester = FusionAttentionTester(device=device)
    
    # 运行测试
    success = tester.run_all_tests()
    
    # 返回状态码
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
