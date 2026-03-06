import torch
import torch_npu
import torch_pfa
import math 

B, H, S, D = 1, 16, 1024, 64
query = torch.randn(B, H, S, D, dtype=torch.float16, device='npu:0')
key = torch.randn(B, H, S, D, dtype=torch.float16, device='npu:0')
value = torch.randn(B, H, S, D, dtype=torch.float16, device='npu:0')
sabi_blocks = torch.randint(0, 65534, (B, H, math.ceil(S/128), math.ceil(S/512)), device="npu:0").to(dtype=torch.uint16)
scale = 1/math.sqrt(float(D))

# Debug: Check tensor properties
print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")

# PFA - dense mode kernel launch through our custom torch_pfa python interface
out = torch_pfa.npu_prompt_flash_attention(query, key, value, num_heads=H, scale_value=scale, input_layout="BNSD", sparse_mode=0)
print(f"Dense PFA output: {out.shape}")

# PFA - sparse mode kernel launch through our custom torch_pfa python interface
out = torch_pfa.npu_prompt_flash_attention(query, key, value, sabi_blocks=sabi_blocks, num_heads=H, scale_value=scale, input_layout="BNSD", sparse_mode=0)
print(f"Sparse PFA output: {out.shape}")

print("Info: correctness was not verified.")
print("SUCCESS (nothing crashed)")