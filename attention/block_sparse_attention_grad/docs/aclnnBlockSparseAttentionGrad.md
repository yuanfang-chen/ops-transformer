# aclnnBlockSparseAttentionGrad

## 产品支持情况

| 产品                                                               | 是否支持 |
| :------------------------------------------------------------------- | :--------: |
| 昇腾910_95 AI处理器                                               |    ×    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                        |    √    |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |
| Atlas 200I/500 A2 推理产品                                         |    ×    |
| Atlas 推理系列产品                                                 |    ×    |
| Atlas 训练系列产品                                                 |    ×    |
| Atlas 200I/300/500 推理产品                                        |    ×    |

## 功能说明

* ​**算子功能**​：BlockSparseAttention稀疏注意力反向计算，支持灵活的块级稀疏模式，通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。
* ​**计算公式**​：稀疏块大小：$blockShapeX×blockShapeY$，BlockSparseMask指定稀疏模式
  
  已知正向计算公式为：
  
  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$
  
  为方便表达，以变量$S$和$P$表示计算公式：
  
  $$
  S = Mask(scale⋅query⋅key_{sparse}^{T}+atten\_mask)
  $$
  
  $$
  P = SoftMax(S)
  $$
  
  $$
  Out = PV
  $$
  
  则反向计算公式为：

$$
softmax\_grad = softmaxGrad(dOut, attentionOut)
$$

$$
dP=dOut * V^T
$$

$$
dS = P * (dP-softmax\_grad)
$$

$$
dV=P^T * dOut
$$

$$
dQ=(dS*K)*scale
$$

$$
dK=(dS^T*Q)*scale
$$

BlockSparseAttentionGrad输入dout, query、key、value, attentionOut的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。

* B：表示输入样本批量大小（Batch）
* T：B和S合轴紧密排列的长度（Total tokens）
* S：表示输入样本序列长度（Seq-Length）
* H：表示隐藏层的大小（Head-Size）
* N：表示多头数（Head-Num）
* D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

当前支持的布局：

* qInputLayout: "TND" "BNSD"
* kvInputLayout: "TND" "BNSD"

## 函数原型

每个算子分为[两段式接口](https://wiki.huawei.com/domains/docs/context/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用"aclnnBlockSparseAttentionGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBlockSparseAttentionGrad"接口执行计算。

<pre class="language-none"><div class="lineNumberForCode"></div><code class="language-c++ hljs language-none">aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
  const aclTensor   *dout,
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attentionOut,
  const aclTensor   *softmaxLse,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            preTokens,
  int64_t            nextTokens,
  aclTensor         *dq,
  aclTensor         *dk,
  aclTensor         *dv,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
</code><span></span></pre>

<pre class="language-none"><div class="lineNumberForCode"></div><code class="language-c++ hljs language-none">aclnnStatus aclnnBlockSparseAttentionGrad(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
</code><span></span></pre>

### aclnnBlockSparseAttentionGradGetWorkspaceSize

* **参数说明：**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| - | - | - | - | - | - | - | - |
| dout | 输入 | Device侧的aclTensor，公式中的dout。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, headDim]。<br>• BNSD: [batch, headNum, maxQSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3/4 | × |
| query | 输入 | Device侧的aclTensor，公式中的query。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, headDim]。<br>• BNSD: [batch, headNum, maxQSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3/4 | × |
| key                                                                                                                                                                                               | 输入 | Device侧的aclTensor，公式中的key。 | 支持的shape为：<br>• TND: [totalKTokens, numKeyValueHeads, headDim]。<br>• BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3/4 | × |
| value                                                                                                                                                                                             | 输入 | Device侧的aclTensor，公式中的value。 | 支持的shape为：<br>• TND: [totalVTokens, numKeyValueHeads, headDim]。<br>• BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3/4 | × |
| attentionOut | 输入 | Device侧的aclTensor，公式中的attentionOut。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, headDim]。<br>• BNSD: [batch, headNum, maxQSeqLength, headDim]。 | FLOAT16、BFLOAT16| ND | 3/4 | × |
| softmaxLse | 输入 | Device侧的aclTensor，Softmax计算的log-sum-exp中间结果。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, 1]。<br>• BNSD: [batch, headNum, maxQSeqLength, 1]。 | FLOAT | ND | 3/4 | × |
| blockSparseMaskOptional                                                                                                                                                                           | 输入 | Device侧的aclTensor，表示实际的稀疏pattern。 | 可选输入（当前版本为必选）<br>• shape为[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]。<br>• 表示按block划分后哪些block需要参与计算（为1），哪些block不参与计算（为0）<br>• 如传入nullptr，则视为不开启块稀疏计算，即所有token之间的注意力分数都会被计算 | BOOL | ND | 4 | × |
| attenMaskOptional                                                                                                                                                                                 | 输入 | Device侧的aclTensor，公式中的atten_mask。 | atten_mask会与稀疏pattern叠加产生作用。当前不支持，应传入nullptr。 | BOOL | ND | 2 | × |
| blockShapeOptional                                                                                                                                                                                | 输入 | Host侧的aclIntArray，稀疏块形状数组。 | 与blockSparseMaskOptional配合使用：<br>• 当配置了blockSparseMaskOptional时：如配置此输入，算子会从中获取稀疏块尺寸；如不配置此输入，算子将默认稀疏块尺寸为[128,128]。<br>• 当未配置blockSparseMaskOptional时：无论此项如何配置，算子均将忽略。<br> 当配置此输入时：必须包含至少两个元素[blockShapeX, blockShapeY]<br>• blockShapeX: Q方向块大小，值必须大于0。<br>• blockShapeY: KV方向块大小，值必须大于0。 | INT64 | - | 1 | - |
| actualSeqLengthsOptional                                                                                                                                                                          | 输入 | Host侧的aclIntArray，描述每个Batch对应的query序列长度。 | 可选输入，用于变长序列场景：<br>• 当qInputLayout为"TND"时：该项输入必须配置<br>• 当qInputLayout为"BNSD"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；如不配置该项输入(传入nullptr)，算子内会按照query的shape中的S进行处理。 | INT64 | - | 1 | - |
| actualSeqLengthsKvOptional                                                                                                                                                                        | 输入 | Host侧的aclIntArray，描述每个Batch对应的key/value序列长度。 | 可选输入，用于变长序列场景：<br>• 当kvInputLayout为"TND"时：该项输入必须配置<br>• 当kvInputLayout为"BNSD"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；如不配置该项输入(传入nullptr)，算子内会按照key/value的shape中的S进行处理。 | INT64 | - | 1 | - |
| qInputLayout                                                                                                                                                                                      | 输入 | Host侧的string，代表输入query的数据排布格式。 | 当前仅支持"TND""BNSD"，qInputLayout与kvInputLayout需要保持一致。 | String | - | - | - |
| kvInputLayout                                                                                                                                                                                     | 输入 | Host侧的string，代表输入key、value的数据排布格式。 | 当前仅支持"TND""BNSD"，qInputLayout与kvInputLayout需要保持一致。 | String | - | - | - |
| numKeyValueHeads                                                                                                                                                                                  | 输入 | Host侧的int64_t，代表key/value的head个数。 | - | INT64 | - | - | - |
| maskType                                                                                                                                                                                          | 输入 | Host侧的int64_t，表示attention计算中的掩码类型。 | 当前只支持传 0：代表不加mask场景 | INT64 | - | - | - |
| scaleValue                                                                                                                                                                                        | 输入 | Host侧的double，公式中的scale，代表缩放系数。 | 一般设置为D^-0.5。 | DOUBLE | - | - | - |
| preTokens                                                                                                                                                                                         | 输入 | Host侧的int64_t，滑窗attention场景下，滑窗需要向前包含多少个token。 | 用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。 | INT64 | - | - | - |
| nextTokens                                                                                                                                                                                        | 输入 | Host侧的int64_t，滑窗attention场景下，滑窗需要向后包含多少个token。 | 用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。 | INT64 | - | - | - |
| dq                                                                                                                                                                                      | 输出 | Device侧的aclTensor，公式中的dq。 | 数据类型和shape与query保持一致。 | FLOAT16、BFLOAT16 | ND | 3/4 | √ |
| dk                                                                                                                                                                                      | 输出 | Device侧的aclTensor，公式中的dk。 | 数据类型和shape与key保持一致。 | FLOAT16、BFLOAT16 | ND | 3/4 | √ |
| dv                                                                                                                                                                                      | 输出 | Device侧的aclTensor，公式中的dv。 | 数据类型和shape与value保持一致。 | FLOAT16、BFLOAT16 | ND | 3/4 | √ |
| workspaceSize                                                                                                                                                                                     | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
| executor                                                                                                                                                                                          | 输出 | 返回op执行器，包含算子计算流程。 | - | - | - | - | - |

* **返回值：**

返回aclnnStatus状态码，具体参见[aclnn返回码](https://wiki.huawei.com/domains/docs/context/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

| 返回码 | 错误码 | 描述 |
| - | - | - |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 输入query，key，value，attentionOut传入的是空指针。 |
| qInputLayout为"TND"时，actualSeqLengthsOptional传入的是空指针。                           |
| kvInputLayout为"TND"时，actualSeqLengthsKvOptional传入的是空指针。                        |
| ACLNN_ERR_PARAM_INVALID                                                                | 161002 | query，key，value 数据类型不在支持的范围之内。 |
| qInputLayout或kvInputLayout输入不合法。                                                   |

### aclnnBlockSparseAttentionGrad

* **参数说明：**
  
  | 参数名 | 输入/输出 | 描述 |
| - | - | - |  
| workspace | 输入 | 在Device侧申请的workspace内存地址。 |
| workspaceSize                                          | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnRainFusionAttentionGetWorkspaceSize获取。 |
| executor                                               | 输入 | op执行器，包含了算子计算流程。 |
| stream                                                 | 输入 | 指定执行任务的AscendCL stream流。 |
  
  
* **返回值：**
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://wiki.huawei.com/domains/docs/context/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

## 约束说明

* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
* blockShapeOptional如果传入，则必须包含至少两个元素[blockShapeX, blockShapeY]，且值必须大于0。
* attentionMaskOptional当前只支持传入nullptr。
* actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
* qSeqlen和kvSeqlen不需要被blockShape整除，支持非对齐场景，实际分块数通过向上取整计算。
* 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0。
* preTokens，nextTokens当前只支持输入2147483647（极大值），表示当前所有token都要参与计算。
* masktype当前只支持输入0，表示不加mask。
* headdim <= 128
* 默认确定性实现

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://wiki.huawei.com/domains/docs/context/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%A0%B7%E4%BE%8B.md)。

<pre class="language-none"><div class="lineNumberForCode"
</div><code class="language-c++ hljs language-none">#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>&amp; shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>&amp; hostData, const std::vector<int64_t>&amp; shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}


int main() {
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &amp;stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置参数
    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;

    // 计算TND格式维度
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Q块的X维度数量
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // KV块的Y维度数量
    // totalQBlocks = qBlockNum * numHeads (每个Q块对应一个head)
    int32_t totalQBlocks = qBlockNum * batch;
    int32_t maxKvBlockNum = kvBlockNum;


    // 3. 创建Query tensor (TND format: [totalQTokens, numHeads, headDim])
    void *queryDeviceAddr = nullptr;
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    aclTensor *queryTensor = nullptr;
    ret = CreateAclTensor(queryHostData, queryShape, &amp;queryDeviceAddr, aclDataType::ACL_FLOAT16, &amp;queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);

    // 4. 创建Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim])
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    std::vector<op::fp16_t> valueHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    ret = CreateAclTensor(keyHostData, kvShape, &amp;keyDeviceAddr, aclDataType::ACL_FLOAT16, &amp;keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n"); return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &amp;valueDeviceAddr, aclDataType::ACL_FLOAT16, &amp;valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n"); return ret);

    // 5. 生成稀疏pattern blockSparseMask
    // [batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)] - 四维tensor
    // 稀疏率为1，即不做稀疏，每个Q块选择所有KV块
    std::vector<bool> blockSparseMaskHostData(batch * numHeads * qBlockNum * kvBlockNum , 1);

    void *blockSparseMaskDeviceAddr = nullptr;
    std::vector<int64_t> blockSparseMaskShape = {batch, numHeads, qBlockNum, kvBlockNum};
    aclTensor *blockSparseMaskTensor = nullptr;
    ret = CreateAclTensor(blockSparseMaskHostData, blockSparseMaskShape , &amp;blockSparseMaskDeviceAddr , aclDataType::ACL_BOOL, &amp;blockSparseMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create blockSparseMask tensor\n"); return ret);

    // 6. 创建输出tensor
    void *outputDeviceAddr = nullptr;
    std::vector<int64_t> outputShape = {totalQTokens, numHeads, headDim};
    int64_t outputElementCount = totalQTokens * numHeads * headDim;
    std::vector<op::fp16_t> outputHostData(outputElementCount, 0.0f);
    aclTensor *outputTensor = nullptr;
    ret = CreateAclTensor(outputHostData, outputShape, &amp;outputDeviceAddr, aclDataType::ACL_FLOAT16, &amp;outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create output tensor\n"); return ret);

    // 7. 创建blockShape数组
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n"); return -1);

    // 8. 创建actualSeqLengths和actualSeqLengthsKv (必需参数)
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));

    void *actualSeqLengthsDevice = nullptr;
    void *actualSeqLengthsKvDevice = nullptr;
    size_t seqLengthsSize = batch * sizeof(int64_t);

    ret = aclrtMalloc(&amp;actualSeqLengthsDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n"); return ret);
    ret = aclrtMalloc(&amp;actualSeqLengthsKvDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n"); 
              aclrtFree(actualSeqLengthsDevice); return ret);

    ret = aclrtMemcpy(actualSeqLengthsDevice, seqLengthsSize, actualSeqLengthsHost.data(), 
                     seqLengthsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n"); 
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDevice, seqLengthsSize, actualSeqLengthsKvHost.data(), 
                     seqLengthsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n"); 
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);

    // aclCreateIntArray 期望的是 host 侧的数据指针，而不是 device 侧的数据
    aclIntArray *actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    aclIntArray *actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr &amp;&amp; actualSeqLengthsKv != nullptr, 
              LOG_PRINT("Failed to create actualSeqLengths arrays\n"); 
              if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
              if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice); return -1);

    // 9. 准备字符串参数（确保缓冲区大小足够，包含null terminator）
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);

    // 10. 计算scaleValue
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));

    // 11. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
        doutTensor,           // dout
        queryTensor,           // query
        keyTensor,             // key
        valueTensor,           // value
        outTensor,           // out
        softmaxLseTensor,           // softmaxLse
        blockSparseMaskTensor,      // blockSparseMaskOptional
        nullptr,               // attenMaskOptional
        blockShape,            // blockShapeOptional
        actualSeqLengths,      // actualSeqLengthsOptional
        actualSeqLengthsKv,    // actualSeqLengthsKvOptional
        qLayoutBuffer,         // qInputLayout
        kvLayoutBuffer,        // kvInputLayout
        numKvHeads,            // numKeyValueHeads
        0,                     // maskType
        scaleValue,            // scaleValue
        2147483647,     // preTokens
        2147483647,     // nextTokens
        dqTensor,          // dq
        dkTensor,          // dk
        dvTensor,          // dv
        &amp;workspaceSize,        // workspaceSize (out)
        &amp;executor);            // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);

    // 12. 分配workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&amp;workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 12. 调用第二段接口
    ret = aclnnBlockSparseAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 13. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 14. 获取输出的值，将device侧内存上的结果拷贝至host侧
    int64_t outputSize = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(outputSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), outputDeviceAddr,
                     outputSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 15. 打印部分结果
    uint64_t printNum = 10;
    LOG_PRINT("Output results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum &amp;&amp; i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }

    // 16. 释放资源
    if (workspaceAddr) aclrtFree(workspaceAddr);
    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (valueDeviceAddr) aclrtFree(valueDeviceAddr);
    if (outputDeviceAddr) aclrtFree(outputDeviceAddr);
    if (blockSparseMaskDeviceAddr ) aclrtFree(blockSparseMaskDeviceAddr );
    if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
    if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice);

    if (queryTensor) aclDestroyTensor(queryTensor);
    if (keyTensor) aclDestroyTensor(keyTensor);
    if (valueTensor) aclDestroyTensor(valueTensor);
    if (outputTensor) aclDestroyTensor(outputTensor);
    if (blockSparseMaskTensor) aclDestroyTensor(blockSparseMaskTensor);
    if (blockShape) aclDestroyIntArray(blockShape);
    if (actualSeqLengths) aclDestroyIntArray(actualSeqLengths);
    if (actualSeqLengthsKv) aclDestroyIntArray(actualSeqLengthsKv);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}

</code><span></span></pre>