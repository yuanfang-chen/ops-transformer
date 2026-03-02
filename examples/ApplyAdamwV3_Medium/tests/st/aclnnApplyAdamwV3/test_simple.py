#!/usr/bin/env python3
import torch
import torch_npu
import math
import numpy as np

def adamw_golden(var, m, v, grad, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps, amsgrad=False, maximize=False, max_grad_norm=None):
    if maximize:
        grad = -grad
    m_out = beta1 * m + (1 - beta1) * grad
    v_out = beta2 * v + (1 - beta2) * grad * grad
    if amsgrad and max_grad_norm is not None:
        v_out = torch.maximum(v_out, max_grad_norm)
    bias_correction1 = 1 - beta1_power
    bias_correction2 = 1 - beta2_power
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = torch.sqrt(v_out) / bias_correction2_sqrt + eps
    var_out = var - step_size * m_out / denom
    var_out = var_out - lr * weight_decay * var
    return var_out, m_out, v_out

def compare_tensor(a, b, rtol, atol, name):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = torch.abs(a - b)
        max_diff = diff.max().item()
        print(f"  [FAIL] {name}: max_diff={max_diff:.6e}")
        return False
    print(f"  [PASS] {name}")
    return True

def test_apply_adamw_v3():
    torch.npu.set_device(0)
    
    tests = [
        {"shape": [1024], "dtype": torch.float32, "amsgrad": False, "maximize": False},
        {"shape": [1024], "dtype": torch.float16, "amsgrad": False, "maximize": False},
        {"shape": [1024], "dtype": torch.bfloat16, "amsgrad": False, "maximize": False},
        {"shape": [1024], "dtype": torch.float32, "amsgrad": True, "maximize": False},
        {"shape": [1024], "dtype": torch.float32, "amsgrad": False, "maximize": True},
        {"shape": [64, 64], "dtype": torch.float32, "amsgrad": False, "maximize": False},
    ]
    
    for i, t in enumerate(tests):
        print(f"\n=== Test {i}: shape={t['shape']}, dtype={t['dtype']}, amsgrad={t['amsgrad']}, maximize={t['maximize']} ===")
        
        shape = t["shape"]
        dtype = t["dtype"]
        amsgrad = t["amsgrad"]
        maximize = t["maximize"]
        
        var = torch.randn(shape, dtype=torch.float32).npu()
        m = torch.randn(shape, dtype=torch.float32).npu()
        v = torch.abs(torch.randn(shape, dtype=torch.float32)).npu() + 0.1
        grad = torch.randn(shape, dtype=torch.float32).npu()
        
        beta1 = 0.9
        beta2 = 0.999
        lr = 0.001
        weight_decay = 0.01
        eps = 1e-8
        beta1_power = 0.9
        beta2_power = 0.999
        
        if dtype != torch.float32:
            var = var.to(dtype)
            m = m.to(dtype)
            v = v.to(dtype)
            grad = grad.to(dtype)
        
        var_golden = var.clone().float()
        m_golden = m.clone().float()
        v_golden = v.clone().float()
        grad_golden = grad.clone().float()
        
        var_out, m_out, v_out = adamw_golden(
            var_golden, m_golden, v_golden, grad_golden,
            beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps, amsgrad, maximize
        )
        
        try:
            from torch_npu.utils import aclnn_apply_adamw_v3
            var_npu = var.clone()
            m_npu = m.clone()
            v_npu = v.clone()
            
            aclnn_apply_adamw_v3(var_npu, m_npu, v_npu, grad, beta1_power, beta2_power, 
                                   lr, weight_decay, beta1, beta2, eps, amsgrad, maximize)
            
            rtol = 1e-3 if dtype == torch.float16 else 1e-4
            atol = 1e-3 if dtype == torch.float16 else 1e-4
            
            passed = True
            passed &= compare_tensor(var_npu.float(), var_out, rtol, atol, "var")
            passed &= compare_tensor(m_npu.float(), m_out, rtol, atol, "m")
            passed &= compare_tensor(v_npu.float(), v_out, rtol, atol, "v")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False
    
    print("\n=== All tests passed ===")
    return True

if __name__ == "__main__":
    test_apply_adamw_v3()
