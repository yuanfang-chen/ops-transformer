# 1. 算子需求

## 1.1 设计约束

运行环境（950 AI处理器）

## 1.2 CasualConv1dFn算子介绍

盘古模型为了增强ffn获取局部信息的能力，增加conv1d计算, 当前token可以获取之前token 信息，从而增强表达能力。

causal_conv1d_fn算子本质：

对连续k个token进行一维卷积操作。

causal_conv1d_fn计算公式：

输入(x)和输出(y)的shape是(S,dim)，(weight)的shape是(K=k,dim)，i分别表示S轴的索引，那么输出将被表示为：

$y[i] = \sum_{k=0}^{K-1}w[k]\cdot  x[i+k]$ .

causal_conv1d_fn算子功能介绍：

对一批变长的 token 序列，执行因果一维卷积（每个特征通道独立），并根据每条序列是否已有历史上下文，决定是否使用缓存的历史状态参与计算；计算完成后，自动用当前输入的最新部分更新其对应的历史状态缓存，同时保证输出符合因果性约束（即每个位置的输出仅依赖于它自身及之前的真实输入）

# 2. 算子设计

## 2.1 接口定义

```c++
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(const aclTensor* x, 
	const aclTensor* weight, 
	aclTensor* cacheState,
	aclTensor* cacheIndices,
	aclTensor* seqStartIndex,
	aclTensor* hasInitialState,
	uint64_t* workspaceSize, 
	aclOpExecutor** executor);
```

## 2.2 参数定义

routingMap，normOut，gradProbs和gradLogits的数据类型必须一致。

| 参数名          | 输入/输出 | 描述                                                                                                            | 使用说明       | 数据类型                                   | 数据格式 | 维度(shape)                                                                                                           | 非连续Tensor |
| --------------- | --------- | --------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------- | ------------ |
| x               | 输入      | 输入序列，采用 CuSeqLen 布局。                                                                                  | 不支持空Tensor | FLOAT16、BFLOAT16                          | ND       | 2维[cu_seq_len, dim]<br />cu_seq_len 为所有变长序列拼接后的总长度，<br />范围是[1, 65536]<br />dim为特征维度,保证为16的倍数，范围是 [64, 16384]。 | √           |
| weight          | 输入      | 因果1维卷积核                                                                                                   | 不支持空Tensor | FLOAT16、BFLOAT16<br />数据类型与输入一致  | ND       | 2维[K, dim]<br /> K是卷积核宽度，K = k(k<=6>)<br />dim为特征维度,与x的dim保持一致，范围是 [64, 16384]。                                              | √           |
| seqStartIndices | 输入      | 序列起始位置索引<br /> 记录各序列在拼接张量 x 中的起始位置：<br />seqStartIndices[i] 表示第 i个序列的起始偏移。 | 不支持空Tensor | INT64                                      | ND       | 1维[batch+1,]<br /> batch 范围[1, 256 ]                                                                               | √           |
| cacheIndices    | 输入      | 缓存索引，<br /> 指定每个序列对应的缓存状态在 cacheState 中的索引                                               | 不支持空Tensor | INT64                                      | ND       | 1维[batch,]                                                                                                           | √           |
| cacheState      | 输入/输出 | 缓存状态张量，存储各序列的历史卷积状态<br />各序列计算完成后原地更新                                            | 不支持空Tensor | FLOAT16、BFLOAT16<br /> 数据类型与输入一致 | ND       | 3维[-1, K-1, dim]<br /> 第0维的大小不固定，且大于batch个数                                                                             | √           |
| hasInitialState  | 可选输入  | 初始状态标志, 表示各序列是否使用缓存数据                                                                        | 不支持空Tensor | BOOL                                       | ND       | 1维[batch,]                                                                                                           | √           |
| y               | 输出      | 输出序列                                                                                                        | -              | FLOAT16、BFLOAT16<br /> 数据类型与输入一致 | ND       | 与x 保持一致                                                                                                          | -            |
| workspaceSize   | 输出      | 返回需要在Device侧申请的workspace大小。                                                                         | -              | -                                          | -        | -                                                                                                                     | -            |
| executor        | 输出      | 返回op执行器，包含了算子计算流程。                                                                              | -              | -                                          | -        | -                                                                                                                     | -            |

## 2.3 逻辑计算流程图

## 2.4 计算公式

# 3. 代码结构

```
ops-transformer\moe\causal_conv1d_fn
├── op_host/
│   └── causal_conv1d_fn_def.cpp     # 算子信息库
│   └── causal_conv1d_tiling_arch35.cpp    # tiling代码
│   └── causal_conv1d_tiling_arch35.h       # tilingData
│
├── op_kernel/
│   └── causal_conv1d.h      # kernel代码
│   └── causal_conv1d_apt.cpp     # kernel入口
```

# 4. 算子信息库

```
namespace ops {
    class CausalConv1dFn : public OpDef {
    public:
        explicit CausalConv1dFn(const char* name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("cacheStates")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("cacheIndices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("seqStartIndex")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("hasInitialState")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_BOOL, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("cacheStates")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->AICore().AddConfig("ascend950");
        }
    };
OP_ADD(CausalConv1dFn);
} // namespace ops
```

# 5. Tiling设计

## 5.1 核间切分

由于输入x的shape为二维，shape为(cu_seq_len, dim)。cu_seq_len是batch和sequence经过乘积之后的结果，以书写便利为例，以BS(batch * sequence)代替cu_seq_len。
前提条件：由于每个batch的sequence长度不同，且差异性较大，因此无法根据batch进行分核。
由于因果卷积需要3个sequence同时参与计算，但是同一个batch中的多个sequence可能出现跨核的情况，因此在切分的时候，需要将每个核第一个处理的sequence的前两个sequence一起载入。所以tiling核间切分时会出现重叠现象。
在上述基础上：核间切分切BS轴，采用核均分方法，BS平均分配到每个核，只有最后一个核是尾核。

## 5.2 核内切分

通过核间切分后，每个核切分得到的shape为（BS.i, dim），即每个核处理的shape大小为（BS.i, dim）
将每个核处理的数据看作一个二维的数据块，有BS.i行，dim列。BS.i是切核后，每个核处理的BS轴的具体大小。
由于dim存在：
1、无法全载的情况（dim=16384，且算卷积的话需要至少一次性载入k个sequence）。
2、每个sequence之间并不连续，sequence和sequence之间的搬运本身就是跳搬。
3、如果以dim全载优先的话，会导致重复搬运的数据变多。
因此，核内切分遵从以下基本规则：
优先保证dim=256B的情况，随后尽可能往BS.i方向切，这样可以在保证burstLen>256B的情况下，有较好的带宽利用率，并尽可能在dim方向上减少重复载入。

> **详细过程：**
> ub内切dim，每次循环满载(n, AlignElement)个元素，其中AlignElement的大小为256B / xDtypeSize。n是BS.i方向能够满载ub的最大值。
> 如果发现n *AlignElement占不满ub（例如总体的bs太小，bs方向全载了。但是dim依旧很大），以256Byte为粒度（取向下对齐）扩展dim，将AlignElement往大方向扩展。

这里需要计算BS方向的循环次数以及tailn的大小，dim方向的循环次数以及tailDim的大小。
要注意，BS方向的循环次数需要把重叠的部分算进去，
例如，一次UB每次载入N*AlignElement个，
BS方向循环次数loopNumBS = ceilDiv（BS - （width - 1）, N - (width - 1)），
Dim方向循环次数loopNumD = ceilDiv(Dim, AlignElement)。

按以下循环方式来处理单个核要处理的元素：

```
//N方向
for i in loopNumBS:
    if (i == loopNumBS - 1):
        i = tilingdata->tailUbFactorN
    else:
        i = tilingdata->ubFactorN
    //dim方向
    for j in loopNumD:
        if(j == loopNumD - 1):
            j = tilingdata->tailubFactorD;
        else:
            j = tilingdata->ubFactorD;
        //对每一个UB块进行处理
        process();
```

## 5.3 TIlingData

```
namespace optiling {

BEGIN_TILING_DATA_DEF(CasualConv1dFnTilingData)
TILING_DATA_FIELD_DEF(uint32_t, loopNumBS);  // 每个核内BS方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, loopNumDim);  // 每个核内Dim方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, ubFactorBS);  // 每个核内BS方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, ubTailFactorBS);  // 每个核内BS方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, ubFactorDim);  // 每个核内Dim方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, ubTailFactorBS);  // 每个核内Dim方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint64_t, blockFactor); //切核的切分因子
TILING_DATA_FIELD_DEF(uint64_t, blockTailFactor); //切核的尾核切分因子
TILING_DATA_FIELD_DEF(uint32_t, tailBlockloopNumBS);  // 每个核内BS方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, tailBlockloopNumDim);  // 每个核内Dim方向的loop循环数
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubFactorBS);  // 每个核内BS方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubTailFactorBS);  // 每个核内BS方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubFactorDim);  // 每个核内Dim方向单次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, tailBlockubTailFactorBS);  // 每个核内Dim方向尾次循环载入的大小
TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);  // 实际使用核数

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CasualConv1dFn, CasualConv1dFnTilingData)

} // namespace optiling
```

# 6. Buffer设计

总体UB空间分配策略如下:
cacheQueue和xQueue使用TQueBind分配buffer（既能VECIN又能VECOUT）。


|  UB                  |  块大小（Byte）               |  BUF_NUM  |  数据类型   |
| ---------------------- | ----------------------------| ----------| ------------|
|  weightInQueue      |  k * sizeof(bf16) * 256        |  1        |  FP16/BF16  |
|  cacheQueue         |  2 * sizeof(bf16) * 256        |  1        |  FP16/BF16  |
|  startLocInQueue    |  64 * sizeof(int64)            |  1        |  INT64  |
|  indicesInQueue     |  64 * sizeof(int64)            |  1        |  INT64  |
|  hasInitialInQueue  |  64 * sizeof(bool)             |  1        |  BOOL   |
|  xQueue(y复用)      |  236 * sizeof(bf16) * 256      |  2        |  FP16/BF16  |

# 7. kernel设计

## 7.1 总体流程

## 7.2 模板具体设计

# 8. golden参考

golden的切分方式与前述Tiling切分机制不同，因此以下golden相关参考代码仅用于流程演示，不具实际部署意义。

```
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
causal_conv1d_fn
"""
import tbetoolkits
import copy
from .registry import register_golden


@register_golden(["causal_conv1d_fn"])
def causal_conv1d_fn(context: "tbetoolkits.UniversalTestcaseStructure"):
    import torch
    import torch.nn.functional as F
    from tbetoolkits.utilities import numpy_to_torch_tensor
    from tbetoolkits.utilities import torch_to_numpy_tensor
    from einops import rearrange
    torch.set_printoptions(threshold=torch.inf)
    
    
    x = context.input_arrays[0]
    x = numpy_to_torch_tensor(x)
    weight = context.input_arrays[1]
    weight = numpy_to_torch_tensor(weight)
    seq_start_indices = context.input_arrays[2]
    seq_start_indices = numpy_to_torch_tensor(seq_start_indices)
    cache_indices = context.input_arrays[3]
    cache_indices = numpy_to_torch_tensor(cache_indices)
    cache_state = context.input_arrays[4]
    cache_state = numpy_to_torch_tensor(cache_state)
    has_inital_state = context.input_arrays[5]
    has_inital_state = numpy_to_torch_tensor(has_inital_state)

    dtype = x.dtypes
    dtype_str = str(dtype)
    # print("dtype_str:", dtype_str)

    cu_seq_len, dim = x.shape
    batch_size = seq_start_indices.shape[0] - 1  # Batch size inferred from query locations

    kernel_width = weight.size(0) # 
    state_len = kernel_width - 1 # 需要缓存的长度

    # Create the output tensor
    out = torch.zeros_like(x) # Shape: (cu_seq_len, dim)

    for batch_idx in range(batch_size):
        # Extract the start and end indices for the current sequence
        start_idx = seq_start_indices[batch_idx].item()
        end_idx = seq_start_indices[batch_idx + 1].item()
        seq_len = end_idx - start_idx # 当前序列的长度
        assert seq_len >= 2 # 要求序列大于2? 为什么要求

        # Extract the sub-tensor for the current sequence
        seq_x = x[start_idx:end_idx]  # Shape: (seq_len, dim)

        # Initialize the state for the sequence
        # 如果缓存了，则从cache_state加载，否则就是0
        if has_inital_state is not None and has_inital_state[batch_idx]:
            cached_state = cache_state[cache_indices[batch_idx]] # Shape: (state_len, dim)
        else:
            cached_state = torch.zeros((state_len, dim), device=x.device, dtype=x.dtype)

        # Concatenate the cached state and the input sequence to create a larger receptive field
	# 类似padding 拼接
        padded_input = torch.cat([cached_state, seq_x], dim=0)  # Shape: (state_len + seq_len, dim)

        # Perform the convolution  仅支持NCL
        result = F.conv1d(
            padded_input.transpose(0, 1).unsqueeze(0),  # Add batch dimension: (1, dim, state_len + seq_len), NCL
            weight.transpose(0, 1).unsqueeze(1),        # Reshape for torch: (dim, 1, kernel_width) 
            bias=None,                  # Optional bias
            stride=1,
            padding=0,                  # Padding already handled via cached states
            groups=dim                  # Grouped convolution to process each feature independently
        ) # Output shape: (1, dim, seq_len)
        result = result.squeeze(0).transpose(0, 1) # Shape: (seq_len, dim)

        # Reset the elements affected by zero padding # pad 0 的token计算结构强制0
        if has_inital_state is None or not has_inital_state[batch_idx]:
            result[:kernel_width-1] = 0

        # Remove unnecessary dimensions and store to the output tensor
        out[start_idx:end_idx] = result  # Shape: (seq_len, dim)

        # Update the cached state for the current sequence
        cache_state[cache_indices[batch_idx]] = padded_input[-state_len:] # (state_len, dim)

    return torch_to_numpy_tensor(out + x), torch_to_numpy_tensor(cache_state) # With residual connection, shape: (cu_seq_len, dim)
```