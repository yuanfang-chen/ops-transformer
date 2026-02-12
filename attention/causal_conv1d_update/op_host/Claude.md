# 1. 算子需求

## 1.1 设计约束

运行环境（950 AI处理器）

## 1.2 causal_conv1d_update 算子

- **参数说明**：
  
| 参数名         | 输入/输出 | 描述                                                                                                                                          | 使用说明       | 数据类型                                   | 数据格式 | 维度(shape)                                                                                               | 非连续Tensor |
| -------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------- | ------------ |
| x              | 输入      | 输入序列                                                                                                                                      | 不支持空Tensor | FLOAT16、BFLOAT16                          | ND       | 3维[batch, m+1, dim]<br />batch 范围1~256，<br />m范围 0，1，2，3，4，5<br />dim为特征维度, [64, 16384]。 | √           |
| filter         | 输入      | 因果1维卷积核                                                                                                                                 | 不支持空Tensor | FLOAT16、BFLOAT16<br />数据类型与输入一致  | ND       | 2维[K, dim]<br /> K是卷积核宽度，K = 3<br />dim为特征维度, [64, 16384]。                                  | √           |
| cacheIndices   | 输入      | 缓存索引，<br /> 指定每个序列对应的缓存状态在 cacheState 中的索引                                                                             | 不支持空Tensor | INT64                                      | ND       | 1维[batch,]                                                                                               | √           |
| cacheState     | 输入/输出 | 缓存状态张量，存储各序列的历史卷积状态<br />各序列计算完成后原地更新                                                                          | 不支持空Tensor | FLOAT16、BFLOAT16<br /> 数据类型与输入一致 | ND       | 3维[-1, K-1+m, dim]                                                                                       | √           |
| acceptTokenNum | 可选输入  | Update场景下接受的 token 数量。<br />指定每个序列实际接受的 token 数（取值范围 [1, m+1]）。<br />若为 None，表示单 token 推理（等价于全 1）。 | 不支持空Tensor | INT64                                      | ND       | 1维[batch,]                                                                                               | √           |
| padSlotId      | 输入      | 不需要参与计算的batch                                                                                                                         |                | INT64                                      |          |                                                                                                           | -            |
| y              | 输出      | 输出序列                                                                                                                                      | -              | FLOAT16、BFLOAT16<br /> 数据类型与输入一致 | ND       | 与x 保持一致                                                                                              | -            |


## 3.3 tiling

核间切分：\
只切batch轴(batch, sequence, dim)则每个核处理（batch.i, sequence, dim）个元素，共batch.o个核参与运算.
通过核间切分后，每个核切分得到的shape为（BS.i, dim），即每个核处理的shape大小为（BS.i, dim）\
核内切分：\
由于dim存在：
①无法全载的情况（dim=16384，且算卷积的话需要至少一次性载入三个sequence）
②每个sequence之间并不连续，sequence和sequence之间的搬运本身就是跳搬
③如果以dim全载优先的话，会导致重复搬运的数据变多
因此，核内切分遵从以下基本规则：
在保证dim=256B的情况下尽可能往sequence方向切，这样可以在保证burstLen>256B的情况下（带宽利用率不错）尽可能少重复载入
ub内切dim，每次循环满载(n, AlignElement)个元素，其中AlignElement的大小为256B / xDtypeSize
如果发现n *AlignElement打不满ub（例如总体的bs太小，bs方向全载了。但是dim依旧很大），以128Byte为粒度（取向下对齐）扩展dim，将AlignElement往大方向扩展
这里需要计算BS方向的循环次数以及tailn的大小，dim方向的循环次数以及tailDim的大小
要注意，BS方向的循环次数需要把重叠的部分算进去
如下图所示，一次UB全载一个红框 每次载入N*AlignElement个
BS方向循环次数loopNumBS = ceilDiv（BS - （width - 1）, N - (width - 1)）
Dim方向循环次数loopNumD = ceilDiv(Dim, AlignElement)

TilingData设计：

|  Key           |  Type      |  Remark                        |
| ---------------- | ------------ | -------------------------------- |
|  loopNumBS             |  int64_t  |  每个核内BS方向的loop循环数           |
|  loopNumDim             |  int64_t  |  每个核内Dim方向的loop循环数           |
|  ubFactorBS             |  int64_t  |  每个核内BS方向单次循环载入的大小           |
|  ubTailFactorBS             |  int64_t  |  每个核内BS方向尾次循环载入的大小         |
|  ubFactorDim      |  int64_t  |  每个核内Dim方向单次循环载入的大小  |
|  ubTailFactorDim            |  int64_t  |  每个核内Dim方向尾次循环载入的大小           |
|  blockFactor     |  int64_t  |  切核的切分因子             |
|  blockTailFactor  |  int64_t  |  切核的尾核切分因子         |
|  tailBlockloopNumBS     |  int64_t  |  每个核内BS方向的loop循环数             |
|  tailBlockloopNumDim  |  int64_t  |  每个核内Dim方向的loop循环数         |
|  tailBlockubFactorBS     |  int64_t  |  每个核内BS方向单次循环载入的大小             |
|  tailBlockubTailFactorBS  |  int64_t  |  每个核内BS方向尾次循环载入的大小         |
|  tailBlockubFactorDim    |  int64_t  |  每个核内Dim方向单次循环载入的大小               |
|  tailBlockubTailFactorDim     |  int64_t  |  每个核内Dim方向尾次循环载入的大小               |

任务：
1.你参考其他算子目录下op_host的tiling代码，校验shape、type、tiling切分等，将代码写入attention\causal_conv1d_update\op_host\causal_conv1d_update_tiling.h和causal_conv1d_update_tiling.cpp
2.完成attention\causal_conv1d_update\tests\utest\op_tiling目录下causal_conv1d_update_tiling.cpp的UT
