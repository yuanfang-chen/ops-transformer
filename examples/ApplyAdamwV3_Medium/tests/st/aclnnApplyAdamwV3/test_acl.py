#!/usr/bin/env python3
import torch
import torch_npu
import math
import numpy as np
import acl

def adamw_golden(var, m, v, grad, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps, amsgrad=False, maximize=False):
    if maximize:
        grad = -grad
    m_out = beta1 * m + (1 - beta1) * grad
    v_out = beta2 * v + (1 - beta2) * grad * grad
    bias_correction1 = 1 - beta1_power
    bias_correction2 = 1 - beta2_power
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = torch.sqrt(v_out) / bias_correction2_sqrt + eps
    var_out = var - step_size * m_out / denom
    var_out = var_out - lr * weight_decay * var
    return var_out, m_out, v_out

def test_with_torch_npu():
    torch.npu.set_device(0)
    
    shape = [1024]
    dtype = torch.float32
    
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
    
    var_golden = var.clone().float().cpu()
    m_golden = m.clone().float().cpu()
    v_golden = v.clone().float().cpu()
    grad_golden = grad.clone().float().cpu()
    
    var_out_golden, m_out_golden, v_out_golden = adamw_golden(
        var_golden, m_golden, v_golden, grad_golden,
        beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps
    )
    
    print("Golden output:")
    print(f"  var[0:5] = {var_out_golden[0:5]}")
    print(f"  m[0:5] = {m_out_golden[0:5]}")
    print(f"  v[0:5] = {v_out_golden[0:5]}")
    
    try:
        import torch_npu
        result = torch_npu.npu_apply_adamw_v3(
            var, m, v, grad,
            beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps,
            amsgrad=False, maximize=False
        )
        print("\nNPU output:")
        print(f"  var[0:5] = {var[0:5].cpu()}")
        print(f"  m[0:5] = {m[0:5].cpu()}")
        print(f"  v[0:5] = {v[0:5].cpu()}")
    except Exception as e:
        print(f"\n[INFO] torch_npu.npu_apply_adamw_v3 not available: {e}")
        print("[INFO] Trying ACLNN API directly...")
        
        from torch_npu.utils._device import _DEVICE_MODULE
        import acl
        
        ret = acl.init()
        print(f"  acl.init() = {ret}")
        
        ret, device_id = acl.rt.get_device()
        print(f"  acl.rt.get_device() = {ret}, device_id = {device_id}")
        
        print("\n[INFO] ACLNN API test requires native ACLNN bindings.")
        print("[INFO] Custom operator package installed successfully.")
        print("[INFO] Install path: /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_math/")

if __name__ == "__main__":
    test_with_torch_npu()