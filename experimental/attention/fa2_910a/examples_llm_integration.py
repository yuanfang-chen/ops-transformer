"""Flash Attention 2 AscendC 算子 — 大模型推理集成示例

本文件展示如何将自定义 FA2 算子集成到主流 LLM 推理代码中。
包含 4 种典型场景：
  1. 直接替换 Transformer Attention
  2. 集成到 HuggingFace transformers
  3. 集成到 vLLM
  4. 通用 monkey-patch 方式
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# =====================================================================
# 核心: 导入自定义 FA2 算子
# =====================================================================
from fa2_ops import flash_attention2


# =====================================================================
# 示例 1：直接替换 Transformer 中的 Attention 计算
# =====================================================================
class AscendFlashAttention(nn.Module):
    """直接使用自定义 FA2 kernel 的 Attention 模块。

    可直接替代任何标准 Multi-Head Attention 的前向计算部分。
    """
    def __init__(self, hidden_size, num_heads, head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)

        # [B, S, H, D] -> [B, H, S, D]  (FA2 要求 BHSD 布局)
        q = q.transpose(1, 2).contiguous().half()
        k = k.transpose(1, 2).contiguous().half()
        v = v.transpose(1, 2).contiguous().half()

        # 调用自定义 FA2 kernel
        attn_output = flash_attention2(q, k, v, causal=True)

        # [B, H, S, D] -> [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, S, -1).float()

        return self.o_proj(attn_output)


# =====================================================================
# 示例 2：集成到 HuggingFace transformers (Llama 为例)
# =====================================================================
def patch_llama_attention():
    """通过 monkey-patch 将 Llama 的 attention 替换为自定义 FA2。

    使用方式:
        patch_llama_attention()
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = model.half().npu()
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError:
        print("transformers not installed, skip Llama patch")
        return

    _original_forward = LlamaAttention.forward

    def _patched_forward(self, hidden_states, attention_mask=None,
                         position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False,
                         cache_position=None, **kwargs):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 处理 GQA：将 KV heads 扩展到与 Q heads 相同
        if self.num_key_value_heads != self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # KV cache 处理
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # 调用自定义 FA2
        q_fa = q.contiguous().half()
        k_fa = k.contiguous().half()
        v_fa = v.contiguous().half()
        attn_output = flash_attention2(q_fa, k_fa, v_fa, causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, S, -1).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, new_kv

    LlamaAttention.forward = _patched_forward
    print("[FA2] Llama attention patched with AscendC FA2 kernel")


# =====================================================================
# 示例 3：集成到 vLLM (PagedAttention 替换)
# =====================================================================
def patch_vllm_attention():
    """将 vLLM 的 prefill attention 替换为自定义 FA2。

    使用方式:
        patch_vllm_attention()
        # 然后正常启动 vLLM
    """
    try:
        from vllm.attention.backends.ascend import AscendAttentionBackend
    except ImportError:
        print("vLLM not installed or no Ascend backend, skip")
        return

    _original_forward = AscendAttentionBackend.forward

    def _patched_forward(self, query, key, value, kv_cache, attn_metadata, **kwargs):
        if attn_metadata.is_prompt:
            B = 1
            H = query.shape[-2] if query.dim() == 4 else self.num_heads
            S = query.shape[-3] if query.dim() == 4 else query.shape[0]
            D = query.shape[-1]

            q = query.view(B, H, S, D).contiguous().half()
            k = key.view(B, H, S, D).contiguous().half()
            v = value.view(B, H, S, D).contiguous().half()

            return flash_attention2(q, k, v, causal=True).view_as(query)
        else:
            return _original_forward(self, query, key, value,
                                     kv_cache, attn_metadata, **kwargs)

    AscendAttentionBackend.forward = _patched_forward
    print("[FA2] vLLM Ascend attention patched with AscendC FA2 kernel")


# =====================================================================
# 示例 4：通用 monkey-patch — 替换任意模型的 scaled_dot_product_attention
# =====================================================================
def patch_torch_sdpa():
    """替换 torch.nn.functional.scaled_dot_product_attention。

    这是最通用的方式，任何使用 F.scaled_dot_product_attention 的模型
    都会自动使用自定义 FA2 kernel。
    """
    import torch.nn.functional as F

    _original_sdpa = F.scaled_dot_product_attention

    def _fa2_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
        if (query.device.type == 'npu' and
            query.dtype == torch.float16 and
            query.dim() == 4 and
            attn_mask is None and
            dropout_p == 0.0):
            q = query.contiguous()
            k = key.contiguous()
            v = value.contiguous()
            sc = scale if scale is not None else 0.0
            return flash_attention2(q, k, v, causal=is_causal, softmax_scale=sc)
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale)

    F.scaled_dot_product_attention = _fa2_sdpa
    print("[FA2] torch.nn.functional.scaled_dot_product_attention patched")


# =====================================================================
# 完整推理示例
# =====================================================================
def demo_inference():
    """演示完整的推理流程。"""
    import torch_npu

    device = "npu:4"
    torch.npu.set_device(device)

    print("\n" + "=" * 60)
    print("示例 1：直接使用 AscendFlashAttention 模块")
    print("=" * 60)

    hidden_size = 512
    num_heads = 8
    seq_len = 128
    batch = 1

    model = AscendFlashAttention(hidden_size, num_heads).half().to(device)
    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.float16)

    with torch.no_grad():
        out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  ✅ 运行成功")

    print("\n" + "=" * 60)
    print("示例 2：通用 SDPA Patch")
    print("=" * 60)

    patch_torch_sdpa()

    import torch.nn.functional as F
    q = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 64, 64, device=device, dtype=torch.float16)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output: {out.shape}")
    print(f"  ✅ SDPA 已透明替换为 FA2 kernel")

    from fa2_ops import cleanup
    cleanup()
    print("\n清理完成")


if __name__ == "__main__":
    demo_inference()
