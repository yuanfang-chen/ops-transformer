# import torch
# import torch_npu
# from torch_pfa import npu_prompt_flash_attention
# help(npu_prompt_flash_attention)

import torch
import torch_npu
import torch_pfa
import math 

B, H, S, D = 1, 16, 1024, 64
query = torch.randn(B, H, S, D, dtype=torch.float16, device='npu:0')
key = torch.randn(B, H, S, D, dtype=torch.float16, device='npu:0')
value = torch.randn(B, H, S, D, dtype=torch.float16, device='npu:0')
scale = 1/math.sqrt(float(D))

# Debug: Check tensor properties
print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")

# PFA
out = torch_pfa.npu_prompt_flash_attention(query, key, value, actual_seq_lengths=[S], actual_seq_lengths_kv=[S], num_heads=H, scale_value=scale, input_layout="BNSD", sparse_mode=0)
print(out.shape)

# # FIA - doesn't crash, although report "Warning: Version: 8.5.0.alpha001 is invalid"
# out_tuple = torch_npu.npu_fused_infer_attention_score(query, key, value, num_heads=H, scale=scale, input_layout="BNSD")
# print(out_tuple[0].shape)   
# print(out_tuple[1].shape)   

print("Success!")