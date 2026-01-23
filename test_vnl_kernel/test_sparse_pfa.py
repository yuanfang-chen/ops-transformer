import os
import sys
sys.path.append("..")
import torch
import torch_npu
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# import sabi_attention
from benchmark.benchmark import * 

PRINT_BLOCK_EQUALITY = True
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

def get_block_mask(q, sabi, block_size_Q, block_size_K):
    B, H, N, D = q.shape

    block_num_Q = math.ceil(N / block_size_Q)
    block_num_K = math.ceil(N / block_size_K)

    block_mask = torch.full((B, H, block_num_Q, block_num_K), True, device=q.device, dtype=torch.bool)
    valid_mask = sabi > -1
    b, h, r, t = torch.where(valid_mask)
    col_indices = sabi[valid_mask].long()

    block_mask[b, h, r, col_indices] = False
    
    # print(block_mask)
    return block_mask

def get_token_mask(block_mask, q, k, block_size_Q=128, block_size_K=128):
    B, H, N, D = q.shape

    repeat_block_mask = torch.repeat_interleave(block_mask, block_size_K, dim = 3)
    repeat_block_mask = torch.repeat_interleave(repeat_block_mask, block_size_Q, dim = 2)
    repeat_block_mask = repeat_block_mask[:,:,:N, :N]
    return repeat_block_mask

def load_data(dir_path, H, device):
    
    q = torch.load(f"{dir_path}/real_data/10_0_q.pt", map_location=device)[:,:H]
    k = torch.load(f"{dir_path}/real_data/10_0_k.pt", map_location=device)[:,:H]
    v = torch.load(f"{dir_path}/real_data/10_0_v.pt", map_location=device)[:,:H]
    actseqlen, actseqlenkv = [q.shape[2]] * B, [k.shape[2]] * B
    # sparsity = torch.load(f"{dir_path}/sparsity/320x480x65/v3/step-10/sparsity_of_RE_0.9_only_img.pt", map_location=device)[0,:H]
    sabi_path = f"{dir_path}/mask/sabi_tensor.pt"
    sabi_tensor = torch.load(sabi_path, map_location=device)

    return q, k, v, sabi_tensor, actseqlen, actseqlenkv



if __name__ == "__main__":
    txt_len = 6
    B, H, S_q, S_kv, D = 1, 3, 10200+txt_len, 10200+txt_len, 128
    sink_frame_len = 600
    scale = 1.0 / math.sqrt(float(D))
    device = torch.device("npu:5")
    dtype = torch.bfloat16
    block_size_q = 128
    block_size_kv = 512
    dir_path = "."
    q, k, v, sabi_tensor, actseqlen, actseqlenkv = load_data(dir_path, H, device)
    block_mask = get_block_mask(q, sabi_tensor, block_size_q, block_size_kv)
    token_mask = get_token_mask(block_mask, q, k, block_size_q, block_size_kv)

    npu_atten_mask = None
    sm=0
    pre_tok = 2147483647     # default pre-token value
    post_tok = 0     # default post-token value
    out_vnl = prompt_flash_attention_npu(q, 
                k, 
                v,
                sabi_blocks=sabi_tensor,
                actual_seq_lengths=actseqlen,
                actual_seq_lengths_kv=actseqlenkv,
                num_heads=H,
                num_key_value_heads=H,
                input_layout="BNSD",
                scale_value = scale,
                atten_mask=npu_atten_mask,
                sparse_mode=sm,
                pre_tokens=pre_tok,
                next_tokens=post_tok,
                )
    out_ref = ref_prompt_flash_attention_bf16(q, k, v, scale, atten_mask=token_mask)
    # Compare on CPU for convenience
    out_our_cpu = out_vnl.cpu()
    out_ref_cpu = out_ref.cpu()
    print(out_vnl.shape)
    print(out_ref.shape)


    equal_ref = torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.02, atol=0.02)

    if not equal_ref and PRINT_BLOCK_EQUALITY:
        block_allclose_map(out_our_cpu, out_ref_cpu,
                        block_h=PRINT_HEIGHT, block_w=PRINT_WIDTH, rtol=0.02, atol=0.02,
                        print_map=True)
    print(f"Is equal: {equal_ref}")

