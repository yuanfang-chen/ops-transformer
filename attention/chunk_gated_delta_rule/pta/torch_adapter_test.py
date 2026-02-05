import torch
import torch_npu
import torch.nn.functional as F
 
if __name__ == "__main__":
    """
    bs: 一个批量中的语句个数;
    seqlen: 语句的Token数量, 即长度;
    nk, nv: QK和V对应的attention头数;
    dk, dv: QK和V对应的词和位置空间嵌入维度。
    """
    bs, seqlen = 2, 10
    nk, nv = 2, 4
    dk, dv = 16, 16
    actual_seq_lengths = (torch.ones(bs) * seqlen).npu().to(torch.int32)
    T = torch.sum(actual_seq_lengths)

    print(f"Input Info:\n{bs=}, {seqlen=}, {nk=}, {nv=}, {dk=}, {dv=}, {actual_seq_lengths=}, {T=}")

    # 初始化输入张量
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16, device='npu')
    initial_state = torch.rand((bs, nv, dk, dv), dtype=torch.bfloat16, device='npu')
    beta = torch.rand((T, nv), dtype=torch.bfloat16, device='npu')
    gama = torch.rand((T, nv), dtype=torch.float32, device='npu')
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0).npu().to(torch.int32)
    scale = 1.0

    print(f"Input Shape:\n Q:{query.shape}, K:{key.shape}, V:{value.shape},"
          f"\nstate:{initial_state.shape}, beta:{beta.shape}, gama:{gama.shape},"
          f"\nactual_seq_lengths:{initial_state.shape}, beta:{beta.shape}, gama:{gama.shape},"
          f"\ncu_seqlens:{cu_seqlens.shape}, scale:{scale}")

    print("\nTest Torch Adapter...")
    o_chunk, state_chunk = torch_npu.npu_chunk_gated_delta_rule_functional(
        query, key, value, initial_state,
        beta = beta,
        scale = scale,
        actual_seq_lengths = cu_seqlens,
        g = gama
    )

    print("\nRuning Torch Adapter Successfully!!!")
