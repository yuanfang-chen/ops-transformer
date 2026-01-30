# Compressor

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A5 推理系列产品</term>   | √  |

## 功能说明

-   API功能：Compressor将每4或128个token的 KV cache 压缩成一个，然后每个token与这些压缩的 KV cache进行 DSA 计算。在长序列的情况下，Compressor可以有效地减少计算开销。

-   计算公式：
    压缩阶段：
    1. 计算矩阵乘法：
        - C4A: $\left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right], \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right];$ 
        - C128A: $\left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]$
    2. 计算分组加法：
        - C4A: $score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};$ 
        - C128A: $score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};$
    3. 计算分组Softmax：
        - C4A: $S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};$ 
        - C128A: $S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};$
    4. 计算Hadamard乘积:
        - C4A: $(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ; kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $S_H = S_i^\prime \odot kv\_state;$
    5. 沿着压缩轴分组求和：
        - C4A: $kv\_state_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $kv\_state_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};$

    后处理阶段：
    6. 计算RMSNorm：
    7. 计算Rope：
   
-   主要计算过程为：
    1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算后再与$Ape$做Add运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 对$kv\_state$和$score\_state$进行数据重排，再对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行Reducesum运算。
    3. 根据输入数据norm_weight、rope_sin、rope_cos，进行RMSNorm和Rope运算，得到$cmp\_kv$结果输出。

## 函数原型

```
torch_npu.compressor(x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos, rope_head_dim, cmp_ratio, *,kv_block_table = None, score_block_table = None, cu_seqlens = None, seqused = None, start_pos = None, coff = 1, norm_eps = 1e-6, rotary_mode = 1, enabled_grad = false) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明
>**说明：**<br> 
>
>- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层最小的单元、T表示所有Batch输入样本序列长度的累加和。

-   **x**（`Tensor`）：必选参数，表示原始不经压缩的数据，对应公式中的$X$。不支持非连续，数据格式支持ND，数据类型支持`bfolat16`、`folat16`。支持输入shape[B,S,H]、[T,H]。
    
-   **wkv**（`Tensor`）：必选参数，表示KV和压缩权重的权重参数，对应公式中的$W^{KV}$。不支持非连续，数据格式支持ND，数据类型支持`bfolat16`、`folat16`。支持输入shape[coff* D,H]。
    
-   **wgate**（`Tensor`）：必选参数，表示KV和压缩权重的权重参数，对应公式中的$W^{Gate}$。不支持非连续，数据格式支持ND，数据类型支持`bfolat16`、`folat16`。支持输入shape[coff* D,H]。

-   **kv\_state**（`Tensor`）：必选参数，表示kv\_state的历史数据，对应公式中的$kv\_state$。不支持非连续，数据格式支持ND，数据类型支持`folat32`。支持输入shape[block_num,block_size,coff*D]。

-   **score\_state**（`Tensor`）：必选参数，表示score\_state中的历史数据, 对应公式中的$score\_state$。不支持非连续，数据格式支持ND，数据类型支持`folat32`。支持输入shape[block_num,block_size,coff*D]。

-   **ape**（`Tensor`）：必选参数，表示输入的positional biases，对应公式中的$Ape$。不支持非连续，数据格式支持ND，数据类型支持`folat32`。支持输入shape[cmp_ratio,coff* D]。

-   **norm\_weight**（`Tensor`）：必选参数，表示计算RmsNorm时的权重系数。数据类型支持`bfolat16`、`folat16`。支持输入shape[D,]。

-   **rope\_sin**（`Tensor`）：必选参数，表示Rope计算的权重系数。数据类型支持`bfolat16`、`folat16`。支持输入shape[B,ceil(S/cmp_ratio),rope_head_dim]、[min(T,T//cmp_ratio+B),rope_head_dim]。

-   **rope\_cos**（`Tensor`）：必选参数，表示Rope计算的权重系数。数据类型支持`bfolat16`、`folat16`。支持输入shape[B,ceil(S/cmp_ratio),rope_head_dim]、[min(T,T//cmp_ratio+B),rope_head_dim]。

-   **rope\_head\_dim**（`int`）：必选参数，表示rope_cos和rope_sin的hidden层最小单元。目前仅支持64。

-   **cmp\_ratio**（`int`）：必选参数，表示数据压缩率。支持2/4/8/16/32/64/128。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入；之后的参数是可选参数，位置无关，不赋值会使用默认值。

-   **kv\_blcok\_table**（`Tensor`）：可选参数，表示kv\_state存储使用的block映射表。不支持非连续，数据格式支持ND，数据类型支持`int32`。支持输入shape[B,Smax/block_size]。当输入为0时，表示无需进行更新kv_state操作。

-   **score\_blcok\_table**（`Tensor`）：可选参数，表示score\_state存储使用的block映射表。不支持非连续，数据格式支持ND，数据类型支持`int32`。支持输入shape[B,Smax/block_size]。当输入为0时，表示无需进行更新score_state操作。

-   **cu\_seqlens**（`Tensor`）：可选参数，表示不同Batch上的有效token数。不支持非连续，数据格式支持ND，数据类型支持`int32`。支持输入shape[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **seqused**（`Tensor`）：可选参数，表示不同Batch中实际参与运算的token数。不支持非连续，数据格式支持ND，数据类型支持`int32`。支持输入shape[B,]。如果不指定为None时，表示和每个Batch上的Sequence Length长度相同。该入参中每个Batch的有效token数要求小于等于对应cu\_seqlens的值，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。支持长度为B的一维tensor。

-   **start\_pos**（`Tensor`）：可选参数，表示计算起始位置。不支持非连续，数据格式支持ND，数据类型支持`int32`。支持输入shape[B,]。当输入为None时，表示从0开始进行计算。

-   **coff**（`int`）：可选参数，表示headDim的数据份数。默认值1，支持1/2。当coff=1时，表示仅1份headDim数据，无需进行overlap数据重排。当coff=2时，表示2份相互关联的headDim数据，后续需要进行overlap数据重排。

-   **norm\_eps**（`float`）：可选参数，表示RmsNorm计算的权重系数。仅支持默认值1e-6。

-   **ratary\_mode**（`int`）：可选参数，表示Rop计算的模式。默认值1，支持1/2。ratary\_mode为1时，代表half模式。ratary\_mode为2时，代表interleave模式。

-   **enabled\_grad**（`bool`）：可选参数，训练场景使用，表示是否参与反向更新。默认值false，支持false/true。

## 返回值说明
-   **cmp\_kv**（`Tensor`）：必选输出，表示压缩后的数据。不支持非连续，数据格式支持ND。数据类型支持`bfolat16`、`folat16`。支持输出shape[B,ceil(S/cmp_ratio),D]、[min(T,T//cmp_ratio+B),D]。

-   **wkv\_proj**（`Tensor`）：可选输出，训练反向使用，表示wkv权重Matmul的计算结果。不支持非连续，数据格式支持ND。数据类型支持`bfolat16`、`folat16`。支持输出shape[B,S,coff* D]、[T,coff* D]。

-   **softmax\_res**（`Tensor`）：可选输出，训练反向使用，表示Softmax计算结果。不支持非连续，数据格式支持ND。数据类型支持`bfolat16`、`folat16`。支持输出shape[B,ceil(S/cmp_ratio),coff* cmp_ratio,D]、[min(T,T//cmp_ratio+B),coff* cmp_ratio,D]。

-   **norm\_x**（`Tensor`）：可选输出，训练反向使用，表示Rms计算的输入。不支持非连续，数据格式支持ND。数据类型支持`bfolat16`、`folat16`。支持输出shape[B,ceil(S/cmp_ratio),D]、[min(T,T//cmp_ratio+B),D]。

-   **norm\_rstd**（`Tensor`）：可选输出，训练反向使用，表示Rms计算的中间结果，rms(x)。不支持非连续，数据格式支持ND。数据类型支持`bfolat16`、`folat16`。支持输出shape[B,ceil(S/cmp_ratio)]、[min(T,T//cmp_ratio+B)]。

## 约束说明

-   该接口支持B、S泛化, 当layout为(B,S,H)时B,S过大可能出现异常，当layout为(T,H)时T过大可能出现异常。//（注意事项）
-   D支持128/512。
-   H支持1K~10K，512对齐。
-   block_size, 128/256/512/1024；泛化支持小于等于1024，16对齐。

## 调用示例

-   单算子模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    data_type = torch.bfloat16
    hidden_size = 4096
    rope_head_dim = 64
    norm_eps = 1e-6
    coff = 1 # 1:no overlap 2:overlap
    cmp_ratio = 128
    rotary_mode = 2
    head_dim = 512
    cu_seqlens = [0, 1]
    # ------------- 
    B = 1
    S_max = 16384
    block_size = 128
    start_pos = [8191] * B # (B,)
    seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

    # BS是否合轴
    bs_combine_flag = True
    update_flag = 1

    if bs_combine_flag:
        if seqused is not None:
            S = max(seqused)
        else:
            S = 0
            for i in range(B):
                if (cu_seqlens[i + 1] - cu_seqlens[i]) > S:
                    S = cu_seqlens[i + 1] - cu_seqlens[i]
    else:
        cu_seqlens = None
        S = 16384 # 作为x的shape[1]
    ### ======================== gen input data start =============================
    # page state
    max_block_num_per_batch = (S_max + block_size - 1) // block_size
    block_num = B * max_block_num_per_batch
    shuffled_indices = torch.randperm(block_num)
    index = torch.arange(1, block_num + 1, 1, dtype=torch.int32)
    index = index[shuffled_indices].reshape(B, max_block_num_per_batch)
    block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
    for i in range(B):
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            block_table[i][j] = index[i][j]
        end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            block_table[i][j] = index[i][j]
    kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
    score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
        rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + B), rope_head_dim)
        rope_cos_shape = rope_sin_shape
    else:
        x_shape = (B, S, hidden_size)
        rope_sin_shape = (B, (S + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
        rope_cos_shape = rope_sin_shape

    x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type)
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type)
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type)
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32)
    norm_weight = torch.tensor(np.random.uniform(-10, 10, (head_dim))).to(data_type)
    rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type)
    rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type)

    npu_out = (
        torch.ops.custom.compressor(
            x,
            wkv,
            wgate,
            kv_state,
            score_state,
            ape,
            norm_weight, 
            rope_sin,
            rope_cos,
            kv_block_table = block_table,
            score_block_table = block_table,
            cu_seqlens = cu_seqlens,
            seqused = seqused,
            start_pos = start_pos,
            rope_head_dim = rope_head_dim,
            cmp_ratio = cmp_ratio,
            coff = coff,
            norm_eps = norm_eps,
            rotary_mode = rotary_mode
        )
    )
    ### ===================

    ```
-   图模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    data_type = torch.bfloat16
    hidden_size = 4096
    rope_head_dim = 64
    norm_eps = 1e-6
    coff = 1 # 1:no overlap 2:overlap
    cmp_ratio = 128
    rotary_mode = 2
    head_dim = 512
    cu_seqlens = [0, 1]
    # ------------- 
    B = 1
    S_max = 16384
    block_size = 128
    start_pos = [8191] * B # (B,)
    seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

    # BS是否合轴
    bs_combine_flag = True
    update_flag = 1

    if bs_combine_flag:
        if seqused is not None:
            S = max(seqused)
        else:
            S = 0
            for i in range(B):
                if (cu_seqlens[i + 1] - cu_seqlens[i]) > S:
                    S = cu_seqlens[i + 1] - cu_seqlens[i]
    else:
        cu_seqlens = None
        S = 16384 # 作为x的shape[1]
    ### ======================== gen input data start =============================
    # page state
    max_block_num_per_batch = (S_max + block_size - 1) // block_size
    block_num = B * max_block_num_per_batch
    shuffled_indices = torch.randperm(block_num)
    index = torch.arange(1, block_num + 1, 1, dtype=torch.int32)
    index = index[shuffled_indices].reshape(B, max_block_num_per_batch)
    block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
    for i in range(B):
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            block_table[i][j] = index[i][j]
        end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            block_table[i][j] = index[i][j]
    kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
    score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
        rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + B), rope_head_dim)
        rope_cos_shape = rope_sin_shape
    else:
        x_shape = (B, S, hidden_size)
        rope_sin_shape = (B, (S + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
        rope_cos_shape = rope_sin_shape

    x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type)
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type)
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type)
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32)
    norm_weight = torch.tensor(np.random.uniform(-10, 10, (head_dim))).to(data_type)
    rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type)
    rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type)

    class CompressorNetwork(nn.Module):
            def __init__(self):
                super(CompressorNetwork, self).__init__()

            def forward(self, x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin,         
                        rope_cos, rope_head_dim, cmp_ratio, *,kv_block_table = None, score_block_table = None, cu_seqlens = None, 
                        seqused = None, start_pos = None, coff = 1, norm_eps = 1e-6, rotary_mode = 1, enabled_grad = false):
                npu_out = (
                    torch.ops.custom.compressor(
                        x,
                        wkv,
                        wgate,
                        kv_state,
                        score_state,
                        ape,
                        norm_weight, 
                        rope_sin,
                        rope_cos,
                        kv_block_table = block_table,
                        score_block_table = block_table,
                        cu_seqlens = cu_seqlens,
                        seqused = seqused,
                        start_pos = start_pos,
                        rope_head_dim = rope_head_dim,
                        cmp_ratio = cmp_ratio,
                        coff = coff,
                        norm_eps = norm_eps,
                        rotary_mode = rotary_mode
                    )
                )
                return npu_out

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(CompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = npu_mode(                    
                    x,
                    wkv,
                    wgate,
                    kv_state,
                    score_state,
                    ape,
                    norm_weight, 
                    rope_sin,
                    rope_cos,
                    kv_block_table = block_table,
                    score_block_table = block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    rope_head_dim = rope_head_dim,
                    cmp_ratio = cmp_ratio,
                    coff = coff,
                    norm_eps = norm_eps,
                    rotary_mode = rotary_mode)
    ```