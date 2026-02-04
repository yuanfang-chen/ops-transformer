# KvQuantSparseAttnSharedkvMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 推理系列产品</term>   | √  |

## 功能说明
- API功能：`KvQuantSparseAttentionSharedKVMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`KvQuantSparseAttentionSharedKV`算子使用。
- KvQuantSparseAttentionSharedKv计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$。


## 参数说明

>- 参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）注意力头的维度，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。

<table style="undefined;table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 100px">
  <col style="width: 120px">
  <col style="width: 500px">
  <col style="width: 80px">
  <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>num_heads_q</td>
      <td>属性</td>
      <td>公式中的Q的多头数，目前仅支持64</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>公式中的K和V的多头数，目前仅支持1</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>注意力头的维度，目前仅支持512</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kv_quant_mode</td>
      <td>属性</td>
      <td>表示kv nope的量化模式，仅支持1，表示K、V nope为per-tile量化，量化后的KV数据类型为`float8_e4m3`</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>当layout_query为TND时，表示不同Batch中q的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>当layout_kv为TND时，表示不同Batch中ori_kv的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。目前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>当layout_kv为TND时，表示不同Batch中cmp_kv的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。目前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的token数，维度为B。目前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的token数，维度为B。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>输入样本批量大小。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>当layout_query为BSND时，表示每个Batch中的q的有效token数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_kv</td>
      <td>可选属性</td>
      <td>当layout_kv为BSND时，表示每个Batch中的ori_kv的有效token数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>可选属性</td>
      <td>表示通过QLI算法从ori_kv中筛选出的关键稀疏token的个数。目前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示通过QLI算法从cmp_kv中筛选出的关键稀疏token的个数，目前仅支持512。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tile_size</td>
      <td>可选属性</td>
      <td>表示量化粒度，必须能被rope_head_dim整除，默认值为None，目前仅支持64。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rope_head_dim</td>
      <td>可选属性</td>
      <td>表示rope的多头数，默认值为0，目前仅支持64。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示对ori_kv的压缩率，数据范围支持4/128，默认值为None。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量，仅支持默认值127。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，仅支持默认值0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>用于标识输入q的数据排布格式。</td>
      <td>STRING</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>用于标识输入ori_kv和cmp_kv的数据排布格式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>可选属性</td>
      <td>是否含有ori_kv。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>可选属性</td>
      <td>是否含有cmp_kv。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>device</td>
      <td>可选属性</td>
      <td>用于获取设备信息，当输入tensor均没有传入时，此字段必填</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引的列表，shape固定为1024。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。

## 调用说明
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import custom_ops
    import numpy as np
    import torch.nn as nn

    layout_q="TND"
    layout_kv="PA_ND"
    B = 1
    S1 = 1
    S2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True,
        device = "npu:0")
    ```

- aclgraph模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import custom_ops
    import numpy as np
    import torch.nn as nn
    from torchair.configs.compiler_config import CompilerConfig

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, num_heads_q, num_heads_kv, head_dim, kv_quant_mode, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, 
                    seqused_q, seqused_kv, batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk, tile_size, rope_head_dim, 
                    cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv, has_ori_kv, has_cmp_kv, device):

            metadata = torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
                num_heads_q = num_heads_q,
                num_heads_kv = num_heads_kv,
                head_dim = head_dim,
                kv_quant_mode = kv_quant_mode,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_ori_kv = cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv = cu_seqlens_cmp_kv,
                seqused_q = seqused_q,
                seqused_kv = seqused_kv,
                batch_size = batch_size,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_kv = max_seqlen_kv,
                ori_topk = ori_topk,
                cmp_topk = cmp_topk,
                tile_size = tile_size,
                rope_head_dim = rope_head_dim,
                cmp_ratio = cmp_ratio,
                ori_mask_mode = ori_mask_mode,
                cmp_mask_mode = cmp_mask_mode,
                ori_win_left = ori_win_left,
                ori_win_right = ori_win_right,
                layout_q = layout_q,
                layout_kv = layout_kv,
                has_ori_kv = has_ori_kv,
                has_cmp_kv = has_cmp_kv,
                device=device
                )
            return metadata

    npu_mode = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)

    layout_q="TND"
    layout_kv="PA_ND"
    B = 4
    S1 = 1
    S2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    metadata = npu_mode(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        ori_topk = 0,
        cmp_topk=K,
        tile_size = 0,
        rope_head_dim = 0,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True,
        device = "npu:0")
    ```