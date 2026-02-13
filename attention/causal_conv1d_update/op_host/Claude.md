## Task Description:

你参考其他算子目录下op_host的tiling代码，继承TilingBaseClass，校验shape、type，并进行tiling切分等，对每个tensor的shape写个校验函数，对每个tensor的type写个校验函数.
将代码写入/home/public/ops-transformer/attention/causal_conv1d_update/op_host/arch35/causal_conv1d_update_tiling.cpp和/home/public/ops-transformer/attention/causal_conv1d_update/op_host/arch35/causal_conv1d_update_tiling.h，其他文件不要修改.


# 1. 算子需求

## 1.1 设计约束

运行环境（950 AI处理器）

## 1.2 causal_conv1d_update 算子
对一批固定长度的 token 序列（每条包含一个已接受 token 和若干预测 token），执行因果一维卷积（每个特征通道独立），并根据每条序列是否处于推测解码模式，动态决定使用哪些历史 token 作为上下文；计算完成后，自动用当前输入和有效历史的最新部分更新其对应的状态缓存，确保后续推理能正确延续上下文，同时保证输出满足因果性约束。
- **参数说明**：

算子定义位于/home/public/ops-transformer/attention/causal_conv1d_update/op_host/causal_conv1d_update_def.cpp

| 参数名         | 输入/输出 | 描述                                                                                                                                          | 使用说明       | 数据类型                                   | 数据格式 | 维度(shape)                                                                                               | 非连续Tensor |
| -------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------- | ------------ |
| x              | 输入      | 输入序列                                                                                                                                      | 不支持空Tensor | FLOAT16、BFLOAT16                          | ND       | 3维[batch, m+1, dim]<br />batch 范围1~256，<br />m范围 0，1，2，3，4，5<br />dim为特征维度, [64, 16384]。 | √           |
| filter         | 输入      | 因果1维卷积核                                                                                                                                 | 不支持空Tensor | FLOAT16、BFLOAT16<br />数据类型与输入一致  | ND       | 2维[K, dim]<br /> K是卷积核宽度，K = 3<br />dim为特征维度, [64, 16384]。                                  | √           |
| cacheIndices   | 输入      | 缓存索引，<br /> 指定每个序列对应的缓存状态在 cacheState 中的索引                                                                             | 不支持空Tensor | INT64                                      | ND       | 1维[batch,]                                                                                               | √           |
| cacheState     | 输入/输出 | 缓存状态张量，存储各序列的历史卷积状态<br />各序列计算完成后原地更新                                                                          | 不支持空Tensor | FLOAT16、BFLOAT16<br /> 数据类型与输入一致 | ND       | 3维[-1, K-1+m, dim]                                                                                       | √           |
| acceptTokenNum | 可选输入  | Update场景下接受的 token 数量。<br />指定每个序列实际接受的 token 数（取值范围 [1, m+1]）。<br />若为 None，表示单 token 推理（等价于全 1）。 | 不支持空Tensor | INT64                                      | ND       | 1维[batch,]                                                                                               | √           |
| padSlotId      | 输入      | 不需要参与计算的batch                                                                                                                         |                | INT64                                      |          |                                                                                                           | -            |
| y              | 输出      | 输出序列                                                                                                                                      | -              | FLOAT16、BFLOAT16<br /> 数据类型与输入一致 | ND       | 与x 保持一致                                                                                              | -            |


## 1.3 tiling
核的个数和UB大小分别通过GetCoreNumAiv和GetUbBlockSize获得.

核间切分：

1.只切batch轴(batch, sequence, dim)，则每个核处理（batch.i, sequence, dim）个元素，共batch.o个核参与运算. 

2.Golden实现中，部分batch会跳过，如下:
```
for batch_idx in range(batch):
	# Skip the padding batches in graph mode
	if conv_state_indices[batch_idx] == pad_slot_id:
		continue
```
业务侧保证能跳过的batch一定位于最后，因此可以从后往前遍历conv_state_indices，遇到第一个满足conv_state_indices[batch_idx] == pad_slot_id条件时，从batch_idx开始及后面（右边）直接跳过，不参与计算，也不参与tiling核间切分.在tiling中可以通过GetInputTensor(CONV_STATE_INDICES_INDEX)获取conv_state_indices的具体值.伪代码如下：
```
auto convStateIndicesTensor = context_->GetInputTensor(CONV_STATE_INDICES_INDEX);
if (convStateIndicesTensor != nullptr) {
	const int64_t* dataPtr = convStateIndicesTensor->GetData<int64_t>();
	if (dataPtr != nullptr) {
		for (int64_t i = batchSize_ - 1; i >= 0; i--) {
			if (pad_slot_id_ == dataPtr[i]) {
				inValidBatchNum_++;
			} else {
				break;
			}
		}
	}
}
```
最后参与计算的batch数需要减去inValidBatchNum_.
```
int64_t blockFactor = (validBatch + TOTAL_CORE_NUM - 1) / TOTAL_CORE_NUM;
int64_t usedCoreNum = (validBatch + blockFactor - 1) / blockFactor;
int64_t blockTailFactor = validBatch - (usedCoreNum - 1) * blockFactor;
```

核内切分：

1.在保证dim=256B的情况下尽可能往sequence方向切.

2.ub内切dim，每次循环满载(n, AlignElement)个元素，其中AlignElement的大小为256B( 输入x的类型是fp16或者bf16,其xDtypeSize=2,则AlignElement = 256B / xDtypeSize = 128)。n是sequence方向能够满载ub的最大值。（为了满载ub，需要用总的ub空间大小，减去filterQueue、cacheQueue、indicesQueue和acceptTokenQueue的大小. 此时剩余的空间全部用于xQueue，据此可算出n）如果发现n * AlignElement占不满ub（例如总体的sequence太小，sequence方向全载了。但是dim依旧很大），以256Byte为粒度（取向下对齐）扩展dim，将AlignElement往大方向扩展.
这里需要计算sequence方向的循环次数及其尾次循环的大小，dim方向的循环次数及其尾次循环的大小.


TilingData设计：

|  Key           |  Type      |  Remark                        |
| ---------------- | ------------ | -------------------------------- |
|  blockFactor     |  int64_t  |  切核的切分因子             |
|  blockTailFactor  |  int64_t  |  切核的尾核切分因子         |
|  loopNumBS             |  int64_t  |  每个核内BS方向的loop循环数           |
|  loopNumDim             |  int64_t  |  每个核内Dim方向的loop循环数           |
|  ubFactorBS             |  int64_t  |  每个核内BS方向单次循环载入的大小           |
|  ubTailFactorBS             |  int64_t  |  每个核内BS方向尾次循环载入的大小         |
|  ubFactorDim      |  int64_t  |  每个核内Dim方向单次循环载入的大小  |
|  ubTailFactorDim            |  int64_t  |  每个核内Dim方向尾次循环载入的大小           |
|  tailBlockloopNumBS     |  int64_t  |  每个核内BS方向的loop循环数             |
|  tailBlockloopNumDim  |  int64_t  |  每个核内Dim方向的loop循环数         |
|  tailBlockubFactorBS     |  int64_t  |  每个核内BS方向单次循环载入的大小             |
|  tailBlockubTailFactorBS  |  int64_t  |  每个核内BS方向尾次循环载入的大小         |
|  tailBlockubFactorDim    |  int64_t  |  每个核内Dim方向单次循环载入的大小               |
|  tailBlockubTailFactorDim     |  int64_t  |  每个核内Dim方向尾次循环载入的大小               |

## 1.4 Buffer设计
|  UB                  |  块大小（Byte）               |  BUF_NUM  |  数据类型   |
| ---------------------- | ----------------------------| ----------| ------------|
|  filterQueue      |  3 * 256        |  1        |  FP16/BF16  |
|  cacheQueue         |  7 * 256        |  1        |  FP16/BF16  |
|  indicesQueue     |  256 * sizeof(int64)            |  1        |  INT64  |
|  acceptTokenQueue  |  256 * sizeof(int64)             |  1        |  INT64   |
|  xQueue(y复用)      |  472 * 256      |  2        |  FP16/BF16  |



You are an expert Ascendc developer. Given an operation, please generate an optimized Ascendc tiling code. What you should do are as follows:
**Workflow**:
- Step 1: Read the Basic Knowledge document first using Read tool
- Step 2: During development, search API Reference in need
- Step 3: After implementation, read Performance Optimization document using Read tool
- Step 4: Follow the suggestions listed in the Tips section while generating the code

### AscendC Knowledge Base

#### Reference Documentation

**IMPORTANT**: Before developing any AscendC kernel, you MUST read these documentation files:

1. **API Reference**: `/root/.claude/AscendC_API_Reference/AscendC_API_Reference.md`
    - Contains all AscendC API documentation
	- Read this when using similar apis

2. **Basic Knowledge**: `/root/.claude/AscendC_Basic_Knowledge/AscendC_Basic_Knowledge.md`
    - Contains AscendC programming fundamentals
    - Read this before starting any kernel development

3. **Performance Optimization**: `/root/.claude/AscendC_Imporve/AscendC_Imporve.md`
    - Contains optimization techniques and best practices
    - Read this when optimizing kernel performance

### Tips

1. Please calculate the shape based on the shape of input tensors or retrieve the shape from an input attribute.

2. In TilingContext, to get the shape size of an input tensor, you need to first call GetInputShape(index) to get the corresponding input tensor and then call GetOriginShape() to get the shape.
* 2.1 To get the total size of the shape, you should call GetShapeSize(). For example:
`uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();`
* 2.2 To get the number of dimensions of the shape, you should call GetDimNum(). For example:
`uint32_t dims = context->GetInputShape(0)->GetOriginShape().GetDimNum();`
* 2.3 To get the length of a certain dimension of the shape, you should call GetDim(idx). For example:
`uint32_t batch = context->GetInputShape(0)->GetOriginShape().GetDim(0);`

3. To set the shape of an output tensor in InferShapeContext, you need to first call GetOutputShape(index) to get the corresponding output tensor.
* 3.1 To set the dimension number of the output tensor, you should call SetDimNum(dim_num). For example:
`context->GetOutputShape(0)->SetDimNum(3);`
* 3.2 To set the value of a dimension for the output tensor, you should call SetDim(idx, value). For example:
`context->GetOutputShape(0)->SetDim(0, 128);`

4. To get the shape of an input tensor in InferShapeContext, you need to first call GetInputShape(index) to get the corresponding input tensor's shape. Note that you do not need to call GetOriginShape() like you do in TilingContext.
* 4.1 To get the dimension number of the input tensor, you should call GetDimNum(). For example:
`context->GetInputShape(0)->GetDimNum();`
* 4.2 To get the value of a dimension for the input tensor, you should call GetDim(idx). For example:
`context->GetInputShape(0)->GetDim(0);`

5. There is no API called HasAttr. Please directly get the input attributes by following the examples provided in the API knowledge section, as attributes have default values.
* 5.1 For the default value of a float attr, this is no need to add a 'f' after the float number, i.e., it should be `AttrType(OPTIONAL).Float(0.0)` instead of `AttrType(OPTIONAL).Float(0.0f)`.

6. There is no need to get the values of input tensors in the tiling part on the op_host side. You only need to read the attribute values on the op_host side. The input tensor's value should be read on the op_kernel side.

7. In OpDef, for an input or an output variable, the number of parameters in the fields of DataType, Format, and UnknownShapeFormat should be the same.

8. To compare a vector variable to a constant, you should use the CompareScalar API. The scalar value can only be passed as the third parameter.
* Correct Example: `AscendC::CompareScalar(dstLocal, src0Local, 1.0f, AscendC::CMPMODE::LT, tileLength)`
* Incorrect Example: `CompareScalar(dstLocal, 1.0f, src0Local, AscendC::CMPMODE::LT, tileLength)`
* 8.1 You cannot directly use the output of CompareScalar (or Compare) to do math calculations as the output is a bitmask. You should use the Select or Duplicate API to fill data in a tensor based on the bitmask outputted by CompareScalar, and then use the filled tensor for math calculations.
* 8.2 The dstLocal output of the Compare and CompareScalar APIs should be a `LocalTensor<uint_8t>`.
* 8.3 For CompareScalar API, the input tileLength should be aligned with 256 bytes. So you need to do alignment before passing in, i.e., you need to do `tileLength = (tileLength * sizeof(data_type) + 256 - 1) / 256 * 256 / sizeof(data_type)`.
* 8.4 Note that, for other APIs, you do not need to do 256-bytes alignment before passing the tileLength.

9. To calculate a vector variable with a constant, you should use specific APIs like Muls and Adds. For more API uses, please refer to the API info listed in the knowledge part.
* 9.1 Specifically, there are no Subs (subtract scalar) or Divs (divide scalar) APIs. You should use Adds or Muls instead.
* For example, use `AscendC::Adds(dstLocal, srcLocal, -1.0f, tileLength)` to subtract a scalar value.
* Use `AscendC::Muls(dstLocal, srcLocal, 1.0f/factor, tileLength)` to divide by a scalar value.

10. The Select API should have 7 arguments according to the API knowledge. For example:
`AscendC::Select(dstLocal, maskLocal, src0Local, src1Local, CMPMODE, tileLength);`
* 10.1 When you want to select values between a tensor and a scalar value, the CMPMODE should be VSEL_TENSOR_SCALAR_MODE. The scalar value can only be passed as the fourth parameter.
* Correct Example: `Select(dstLocal, maskLocal, src0Local, -1.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength)`
* Incorrect Example: `Select(dstLocal, maskLocal, -1.0f, src0Local, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength)`
* 10.2 When you want to select values between 2 tensors, The CMPMODE should be VSEL_TENSOR_TENSOR_MODE.
* Correct Example: `Select(dstLocal, maskLocal, src0Local, src1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength)`

11. To use temporary variables for calculations in the kernel implementation, you can use TBuf. For TBuf uses, please refer to the TBuf example:
	```
	TBuf变量声明
	private:
		AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
	--------------------------------------------------
	TBuf初始化
	pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half));
	pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
	--------------------------------------------------
	Tbuf使用示例
	AscendC::LocalTensor<float> p1 = tmp1.Get<float>();
	AscendC::Add(p1, xLocal, yLocal, this->processDataNum);
	--------------------------------------------------
	```

12. To compute the sign value of the a tensor, you can directly call the Sign API. The Sign API is used as `Sign(dstTensor, srcTensor, calCount)`.

13. To avoid using the prefix "AscendC::", you should declare using namespace AscendC; at the beginning of the file.

14. To initialize the pipe buffer for a TBuf variable, the only correct way is:
* `pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));`
* You should not pass in the BUFFER_NUM.

15. When you want to declare a const float variable for PI, name it differently, such as custom_pi.

16. Please remember to set the number of used cores by using `context->SetBlockDim(core_num);`

17. To calculate the tile length, you need to think about how many pipes are sharing the UB size. You need to not only count the inputs and outputs, but also count the tbufs. For example, if there is 1 input, 1 output, and 2 tbufs used in the calculation. Then there are a total number of 4 pipes sharing the ub size.

18. For input tensor with shape (1), to get the value in the op_kernel side, where is passed as a GM_ADDR, e.g.,  `__aicore__ inline void Init(GM_ADDR lr)`:
* You should not do: 
`lr = *(reinterpret_cast<__gm__ float*>(lr));`
* You should do:
`lrGm.SetGlobalBuffer((__gm__ float *)lr, 1);`
`this->lrLocal = lrGm.GetValue(0);`
Note that: There must be a SetGlobalBuffer before GetValue(0)

19. To calculate a formula, like `(1 - beta1) * grad`, where `beta1` is a scalar value and `grad` is a vector, you can do in either one of the following two ways:
* 19.1 `Muls(tmpLocal, gradLocal, 1.0f - beta1, this->tileLength);`
* 19.2 `Duplicate(one_minus_beta1, 1.0f - beta1, this->tileLength);` `Mul(tmpLocal, gradLocal, one_minus_beta1, this->tileLength);`

20. To copyout data that is not aligned with 32 bytes between local tensor and global memory, you can use DataCopyPad API. For example, to CopyIn/CopyOut 20 half-type elements:
* You should not use `DataCopy(dstGlobal, dstLocal, 20)`, as the 20 half-type elements have 40 bytes in total and DataCopy API need the data to be 32 bytes aligned. 
* You should use you should use `DataCopyPad(dstLocal, srcGlobal, DataCopyParams, DataCopyPadParams)` to copy in and `DataCopyPad(dstLocal, srcGlobal, DataCopyParams)` to copy out.
* The DataCopyParams struct has four parameters, i.e., blockCount, blockLen, srcStride, dstStride. Note that, the blockLen parameter should be set to the actual data size processed each time, which is datalength * sizeof(data_type). For example: `{{1, static_cast<uint16_t>(datalength * sizeof(float)), 0, 0}}`
* The DataCopyPadParams struct has four parameters, i.e., isPad, leftPadding, rightPadding, and paddingValue. You can set them to `{{false, 0, 0, 0}}`
* Code example is as follows:
	```
	__aicore__ inline void CopyIn()
	{{
		LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
		DataCopyParams copyParams{{1, static_cast<uint16_t>(20 * sizeof(half)), 0, 0}};
		DataCopyPadParams padParams{{false, 0, 0, 0}};
		DataCopyPad(dstGlobal, dstLocal, copyParams, padParams);
		outQueueDst.FreeTensor(dstLocal);
	}}

	__aicore__ inline void CopyOut()
	{{
		LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
		DataCopyParams copyParams{{1, static_cast<uint16_t>(20 * sizeof(half)), 0, 0}};
		DataCopyPad(dstGlobal, dstLocal, copyParams);
		outQueueDst.FreeTensor(dstLocal);
	}}
	```

21. If you want to do AllocTensor() a VECOUT TQue, you should make sure that you have done pipe.InitBuffer() for that TQue in Init().

22. When performing precision conversion using the function `void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal, const RoundMode& round_mode, const uint32_t calCount)`:
* if converting from `int32_t` type to `half` type, it is necessary to execute the instruction `SetDeqScale((half)1.000000e+00f)` in advance to set the value of the DEQSCALE register. 

23. To split the input xLocalF32 into two parts, glu and linear: `x_glu = x[..., ::2]` takes elements with even indices, and `x_linear = x[..., 1::2]` takes elements with odd indices, it can be done as follows:
* Set an arithmetic progression as the address offset: `ArithProgression(xOffsetLocal, (int32_t)0, (int32_t)8, tensorLen)`.
* Cast offset dtype to uint32_t: `LocalTensor<uint32_t> xOffsetLocalU32 = xOffsetLocal.ReinterpretCast<uint32_t>();`
* Collect even-indexed elements: `Gather(x_glu, xLocalF32, xOffsetLocalU32, (uint32_t)0, calEleNum)`, collect odd-indexed elements: `Gather(x_linear, xLocalF32, xOffsetLocalU32, (uint32_t)4, calEleNum)`. 

24. When the computation of the tensor to be output is completed, the next step is to copy the data from the local tensor to the global memory. At this point, it is necessary to use EnQue and DeQue to synchronize the local tensor. Refer to the following example:
	```
	outQueue_.EnQue<int8_t>(yOut);
	yOut = outQueue_.DeQue<int8_t>(); 
	```

25. When using Cast to convert a Tensor from float type to int8 type, you need to first convert the float type to half type, and then convert it to int8 type.

26. On the NPU machine 910b3, there are 40 Vector AI Cores (AIVs), and each AIV has a Unified Buffer with a memory size of 192 KB.

27. When using APIs with mask/mask[] parameters, pay attention to the value range which depends on operand data type. **Incorrect mask values will NOT cause API errors but will produce wrong computation results.**
* **Bit-wise mode** (mask as array): 16-bit operands use array length 2 with mask[0], mask[1] ∈ [0, 2^64-1]; 32-bit operands use array length 1 with mask[0] ∈ (0, 2^64-1]; 64-bit operands use array length 1 with mask[0] ∈ (0, 2^32-1]. Example: `mask=[8, 0]` means only the 4th element participates (8=0b1000).
* **Continuous mode** (mask as scalar): 16-bit operands allow mask ∈ [1, 128]; 32-bit operands allow mask ∈ [1, 64]; 64-bit operands allow mask ∈ [1, 32].

### Debugging Workflow

When debugging AscendC kernels, use the DumpTensor API to systematically track data flow and identify issues. Follow this structured debugging methodology:

#### 1. Add DumpTensor at Key Computation Points

Insert `DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)` after each critical computation step to trace intermediate results.

**Example:**
```cpp
// After input copy
DumpTensor(inputLocal, 100, dumpSize);

// After first computation
Adds(tmpLocal, inputLocal, 1.0f, tileLength);
DumpTensor(tmpLocal, 200, dumpSize);

// After second computation
Mul(outputLocal, tmpLocal, scaleLocal, tileLength);
DumpTensor(outputLocal, 300, dumpSize);
```

#### 2. Use Systematic Desc Numbering

Assign desc numbers with a consistent pattern for easy log analysis:
- **100-199**: Input tensors
- **200-299**: Intermediate computation results
- **300-399**: Output tensors
- Within each range, increment by 10 for each new dump point

This numbering scheme makes it easy to identify which stage of computation produced each log entry.

#### 3. Add Corresponding CPU Golden Prints

In your CPU reference implementation, add print statements at the same logical points to enable step-by-step comparison. Use the same numbering scheme:

```cpp
// CPU golden implementation
printf("[CPU-100] input: %.6f, %.6f, ...\n", input[0], input[1]);
float tmp = input + 1.0f;
printf("[CPU-200] tmp after add: %.6f, %.6f, ...\n", tmp[0], tmp[1]);
float output = tmp * scale;
printf("[CPU-300] output: %.6f, %.6f, ...\n", output[0], output[1]);
```

#### 4. Verify Input Data First

Always start debugging by confirming input data correctness. Add DumpTensor immediately after DataCopy operations to verify data is loaded correctly before investigating computation issues.

#### 5. Use Segmented Verification

Narrow down the problem scope by dividing the computation into segments. Verify each segment independently:
- **First verify**: Input copying stage
- **Then verify**: Computation pipeline
- **Finally verify**: Output copying stage

This approach helps isolate the problematic stage quickly.

#### 6. Analyze Error Patterns

When comparing NPU output with CPU golden results, look for patterns in the errors to identify root causes:
- **All values off by a constant** → Check for missing bias or scale factors
- **Every Nth value is wrong** → Check for stride or alignment issues
- **Values are NaN or Inf** → Check for division by zero or overflow
- **First/last few values are wrong** → Check for boundary conditions or padding issues
- **Errors accumulate over iterations** → Check for uninitialized variables or queue synchronization issues

#### 7. DumpTensor Usage Notes

- The `dumpSize` parameter specifies how many elements to dump
- DumpTensor will output to the system log, which can be viewed using appropriate log viewing tools
- Be mindful of log volume - dump only necessary data points to avoid overwhelming the logs
- Consider dumping a subset of data (e.g., first 32 elements) rather than entire large tensors