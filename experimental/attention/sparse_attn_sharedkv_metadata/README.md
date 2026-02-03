# SparseAttnSharedkvMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明
- API功能：`SparseAttentionSharedKVMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`SparseAttentionSharedKV`算子使用。
- `SparseAttentionSharedKV`计算公式：

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
      <td>输入</td>
      <td>公式中的Q的多头数</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>输入</td>
      <td>公式中的K和V的多头数</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>输入</td>
      <td>注意力头的维度</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>当layout_query为TND时，表示不同Batch中q的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>当layout_kv为TND时，表示不同Batch中ori_kv的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>当layout_kv为TND时，表示不同Batch中cmp_kv的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的token数，维度为B</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的token数，维度为B</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>输入样本批量大小</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>当layout_query为BSND时，表示每个Batch中的q的有效token数</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_kv</td>
      <td>可选属性</td>
      <td>当layout_kv为BSND时，表示每个Batch中的ori_kv的有效token数</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>可选属性</td>
      <td>表示选取ori_topk的K个token</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示选取cmp_topk的K个token</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示对ori_kv的压缩率
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>用于标识输入q的数据排布格式</td>
      <td>STRING</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>用于标识输入ori_kv和cmp_kv的数据排布格式</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>可选属性</td>
      <td>是否含有ori_kv</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>可选属性</td>
      <td>是否含有cmp_kv</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>device</td>
      <td>可选属性</td>
      <td>npu的ID</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引的列表</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。

## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math

    data_type = torch.bfloat16
    softmax_scale = 0.041666666666666664
    b = 4
    s1 = 128
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    k = 512
    ori_block_size = 128
    cmp_block_size = 128
    s2_act = 4096
    cmp_ratio = 4
    ori_win_left = 127
    ori_win_right = 0
    layout_q = 'TND'
    layout_kv = 'PA_ND'
    ori_mask_mode = 4
    cmp_mask_mode = 3
    q = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, dn))).to(data_type).npu()

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1).to(torch.int).npu()
    seqused_kv = torch.tensor([s2]*b).to(torch.int).npu()

    cmp_kv_len = s2_act // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - s1 + 1),  k)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, k). \
        to(torch.int).npu()
        
    ori_block_num =  math.ceil(s2_act/ori_block_size) * b
    ori_block_table = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int).reshape(b, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, n2, dn))).to(data_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/ori_block_size) * b
    cmp_block_table = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int).reshape(b, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, cmp_block_size, n2, dn))).to(data_type).npu()
    sinks = torch.rand(n1).to(torch.float32).npu()
    metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        topk=k,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True
    )
    attn_out, softmax_lse = torch_npu.npu_sparse_attn_sharedkv(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=None,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=False)
    ```

- aclgraph模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import torchair

    data_type = torch.bfloat16
    softmax_scale = 0.041666666666666664
    b = 4
    s1 = 128
    s2 = 8192
    n1 = 64
    n2 = 1
    dn = 512
    k = 512
    ori_block_size = 128
    cmp_block_size = 128
    s2_act = 4096
    cmp_ratio = 4
    ori_win_left = 127
    ori_win_right = 0
    layout_q = 'TND'
    layout_kv = 'PA_ND'
    ori_mask_mode = 4
    cmp_mask_mode = 3
    q = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, dn))).to(data_type).npu()

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1).to(torch.int).npu()
    seqused_kv = torch.tensor([s2]*b).to(torch.int).npu()

    cmp_kv_len = s2_act // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - s1 + 1),  k)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, k). \
        to(torch.int).npu()
        
    ori_block_num =  math.ceil(s2_act/ori_block_size) * b
    ori_block_table = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int).reshape(b, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, n2, dn))).to(data_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/ori_block_size) * b
    cmp_block_table = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int).reshape(b, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, cmp_block_size, n2, dn))).to(data_type).npu()
    sinks = torch.rand(n1).to(torch.float32).npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_kv,
            topk, has_ori_kv, has_cmp_kv, q, ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table, 
            cmp_block_table, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, sinks,
            ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv):
            metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                topk=topk,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=has_ori_kv,
                has_cmp_kv=has_cmp_kv
            )
            npu_out = torch_npu.npu_sparse_attn_sharedkv(
                q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=None,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                return_softmax_lse=False)
            return npu_out

    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)
    attn_out, softmax_lse = mod(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        topk=k,
        has_ori_kv=True,
        has_cmp_kv=True,
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        sinks=sinks,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv)
    ```
