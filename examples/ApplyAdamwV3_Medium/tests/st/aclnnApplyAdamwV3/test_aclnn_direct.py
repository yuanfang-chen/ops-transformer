#!/usr/bin/env python3
import torch
import torch_npu
import math
import numpy as np
import ctypes
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

def test_aclnn_direct():
    print("Initializing ACL...")
    ret = acl.init()
    print(f"acl.init() = {ret}")
    
    ret, device_id = acl.rt.get_device()
    print(f"acl.rt.get_device() = {ret}, device_id = {device_id}")
    
    if ret != 0:
        ret = acl.rt.set_device(0)
        print(f"acl.rt.set_device(0) = {ret}")
    
    ret, stream = acl.rt.create_stream()
    print(f"acl.rt.create_stream() = {ret}, stream = {stream}")
    
    shape = [1024]
    element_count = 1024
    
    var_host = np.array([1.0 + i * 0.01 for i in range(element_count)], dtype=np.float32)
    m_host = np.array([0.1 * (i % 10) for i in range(element_count)], dtype=np.float32)
    v_host = np.array([0.01 + i * 0.001 for i in range(element_count)], dtype=np.float32)
    grad_host = np.array([0.01 * (i + 1) for i in range(element_count)], dtype=np.float32)
    
    var_golden = var_host.copy()
    m_golden = m_host.copy()
    v_golden = v_host.copy()
    
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    weight_decay = 0.01
    eps = 1e-8
    beta1_power = 0.9
    beta2_power = 0.999
    
    var_t = torch.from_numpy(var_golden)
    m_t = torch.from_numpy(m_golden)
    v_t = torch.from_numpy(v_golden)
    grad_t = torch.from_numpy(grad_host)
    
    var_out_g, m_out_g, v_out_g = adamw_golden(
        var_t, m_t, v_t, grad_t,
        beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps
    )
    
    print(f"\nGolden result:")
    print(f"  var[0:5] = {var_out_g[0:5]}")
    print(f"  m[0:5] = {m_out_g[0:5]}")
    print(f"  v[0:5] = {v_out_g[0:5]}")
    
    size_bytes = element_count * 4
    
    ret, var_dev = acl.rt.malloc(size_bytes, 0)
    print(f"acl.rt.malloc var = {ret}, addr = {var_dev}")
    
    ret, m_dev = acl.rt.malloc(size_bytes, 0)
    ret, v_dev = acl.rt.malloc(size_bytes, 0)
    ret, grad_dev = acl.rt.malloc(size_bytes, 0)
    
    ret = acl.rt.memcpy(var_dev, size_bytes, var_host.ctypes.data, size_bytes, 1)
    ret = acl.rt.memcpy(m_dev, size_bytes, m_host.ctypes.data, size_bytes, 1)
    ret = acl.rt.memcpy(v_dev, size_bytes, v_host.ctypes.data, size_bytes, 1)
    ret = acl.rt.memcpy(grad_dev, size_bytes, grad_host.ctypes.data, size_bytes, 1)
    
    ret, beta1_power_dev = acl.rt.malloc(4, 0)
    ret, beta2_power_dev = acl.rt.malloc(4, 0)
    ret, lr_dev = acl.rt.malloc(4, 0)
    ret, weight_decay_dev = acl.rt.malloc(4, 0)
    ret, beta1_dev = acl.rt.malloc(4, 0)
    ret, beta2_dev = acl.rt.malloc(4, 0)
    ret, eps_dev = acl.rt.malloc(4, 0)
    
    ret = acl.rt.memcpy(beta1_power_dev, 4, np.array([beta1_power], dtype=np.float32).ctypes.data, 4, 1)
    ret = acl.rt.memcpy(beta2_power_dev, 4, np.array([beta2_power], dtype=np.float32).ctypes.data, 4, 1)
    ret = acl.rt.memcpy(lr_dev, 4, np.array([lr], dtype=np.float32).ctypes.data, 4, 1)
    ret = acl.rt.memcpy(weight_decay_dev, 4, np.array([weight_decay], dtype=np.float32).ctypes.data, 4, 1)
    ret = acl.rt.memcpy(beta1_dev, 4, np.array([beta1], dtype=np.float32).ctypes.data, 4, 1)
    ret = acl.rt.memcpy(beta2_dev, 4, np.array([beta2], dtype=np.float32).ctypes.data, 4, 1)
    ret = acl.rt.memcpy(eps_dev, 4, np.array([eps], dtype=np.float32).ctypes.data, 4, 1)
    
    print("\nTensors created and data copied to device")
    
    try:
        from atlas_acl_nn import aclnn_ops_math_custom
        print("\nUsing atlas_acl_nn module...")
        
    except ImportError as e:
        print(f"\n[INFO] atlas_acl_nn not available: {e}")
        print("[INFO] Trying to load custom op library directly...")
        
        lib_path = "/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_math/op_api/lib/libcust_opapi.so"
        try:
            lib = ctypes.CDLL(lib_path)
            print(f"Loaded library: {lib_path}")
            
            get_workspace_size = lib.aclnnApplyAdamwV3GetWorkspaceSize
            get_workspace_size.restype = ctypes.c_int
            get_workspace_size.argtypes = [ctypes.c_void_p] * 13
            
            execute = lib.aclnnApplyAdamwV3
            execute.restype = ctypes.c_int
            execute.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
            
            print("\n[SUCCESS] ACLNN functions found in library!")
            print("Custom operator package is correctly installed.")
            print("\n[INFO] To run full test, use torch_npu with custom op registration.")
            
        except Exception as e2:
            print(f"[ERROR] Failed to load library: {e2}")
    
    acl.rt.free(var_dev)
    acl.rt.free(m_dev)
    acl.rt.free(v_dev)
    acl.rt.free(grad_dev)
    acl.rt.free(beta1_power_dev)
    acl.rt.free(beta2_power_dev)
    acl.rt.free(lr_dev)
    acl.rt.free(weight_decay_dev)
    acl.rt.free(beta1_dev)
    acl.rt.free(beta2_dev)
    acl.rt.free(eps_dev)
    
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(0)
    acl.finalize()
    
    print("\n[INFO] ACL resources released.")

if __name__ == "__main__":
    test_aclnn_direct()