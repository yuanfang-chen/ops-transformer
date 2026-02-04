# SparseAttnSharedkvMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明
- API功能：`SparseAttnSharedkvMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`SparseAttnSharedkv`算子使用。
- `SparseAttnSharedkvMetadata`计算公式：

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
      <td>公式中的Q的多头数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>公式中的K和V的多头数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>注意力头的维度。</td>
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
      <td>表示选取ori_topk的K个token。目前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示选取cmp_topk的K个token，目前仅支持4或128。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示对ori_kv的压缩率。</td>
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
      <td>npu的ID。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引的列表。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。

## 调用示例
- 支持单算子模式调用和aclgraph模式调用，作为SparseAttnSharedkv算子的前序算子，调用示例见[SparseAttnSharedkv调用示例](../sparse_attn_sharedkv/README.md)。
