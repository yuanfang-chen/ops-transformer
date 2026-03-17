# aclnnBlockSparseAttentionGrad

## 产品支持情况

| 产品                                                               | 是否支持 |
| :------------------------------------------------------------------- | :--------: |
| Ascend 950PR/Ascend 950DT                                                |    ×    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                        |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品              |    √    |
| Atlas 200I/500 A2 推理产品                                         |    ×    |
| Atlas 推理系列产品                                                 |    ×    |
| Atlas 训练系列产品                                                 |    ×    |

## 功能说明

* ​接口功能​：BlockSparseAttention稀疏注意力反向计算，支持灵活的块级稀疏模式，通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。
* ​计算公式​：稀疏块大小：$blockShapeX×blockShapeY$，BlockSparseMask指定稀疏模式
  
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

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度/长度(Shape/Size) | 非连续Tensor |
| - | - | - | - | - | - | - | - |
| dout（aclTensor*）| 输入 | 反向输出梯度，即公式中的dout。代表最终输出对当前算子的梯度信息。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, headDim]。<br>• BNSD: [batch, headNum, maxQSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3-4 | × |
| query（aclTensor*） | 输入 | 注意力计算中的查询向量，即公式中的query。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, headDim]。<br>• BNSD: [batch, headNum, maxQSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3-4 | × |
| key（aclTensor*） | 输入 | 注意力计算中的键向量，即公式中的key。 | 支持的shape为：<br>• TND: [totalKTokens, numKeyValueHeads, headDim]。<br>• BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3-4 | × |
| value（aclTensor*） | 输入 | 注意力计算中的值向量，即公式中的value。 | 支持的shape为：<br>• TND: [totalVTokens, numKeyValueHeads, headDim]。<br>• BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。 | FLOAT16、BFLOAT16 | ND | 3-4 | × |
| attentionOut（aclTensor*） | 输入 | 正向 BlockSparseAttention 计算的输出结果，即公式中的attentionOut。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, headDim]。<br>• BNSD: [batch, headNum, maxQSeqLength, headDim]。 | FLOAT16、BFLOAT16| ND | 3-4 | × |
| softmaxLse（aclTensor*） | 输入 | Softmax计算的log-sum-exp中间结果。用于反向计算梯度的对数和指数逆推。 | 支持的shape为：<br>• TND: [totalQTokens, headNum, 1]。<br>• BNSD: [batch, headNum, maxQSeqLength, 1]。 | FLOAT | ND | 3-4 | × |
| blockSparseMaskOptional（aclTensor*） | 输入 | 块状稀疏掩码，表示实际的稀疏pattern。决定哪些block实际参与注意力计算。 | 可选输入（当前版本为必选）：<br>• shape为[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]。<br>• 表示按block划分后哪些block需要参与计算（为1），哪些block不参与计算（为0）。<br>• 如传入nullptr，则视为不开启块稀疏计算，即所有token之间的注意力分数都会被计算。 | BOOL | ND | 4 | × |
| attenMaskOptional（aclTensor*） | 输入 | 注意力掩码，即公式中的atten_mask。用于屏蔽不应参与计算的特定token。 | atten_mask会与稀疏pattern叠加产生作用。当前不支持，应传入nullptr。 | BOOL | ND | 2 | × |
| blockShapeOptional（aclIntArray*）| 输入 | 稀疏块形状数组。指定每个稀疏块的二维尺寸（行数和列数）。 | 1. 与blockSparseMaskOptional配合使用：<br>  • 当配置了blockSparseMaskOptional时：如配置此输入，算子会从中获取稀疏块尺寸；如不配置此输入，算子将默认稀疏块尺寸为[128,128]。<br>  • 当未配置blockSparseMaskOptional时：无论此项如何配置，算子均将忽略。<br>2. 当配置此输入时的元素要求：<br>  • 必须包含至少两个元素 `[blockShapeX, blockShapeY]`。<br>  • blockShapeX: Q方向块大小，值必须大于0。<br>  • blockShapeY: KV方向块大小，值必须大于0。 | INT64 | - | 1 | - |
| actualSeqLengthsOptional（aclIntArray*） | 输入 | query的实际序列长度数组。描述变长场景下每个Batch中实际有效的query token数量。 | 可选输入，用于变长序列场景：<br>• 当qInputLayout为"TND"时：该项输入必须配置<br>• 当qInputLayout为"BNSD"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；如不配置该项输入(传入nullptr)，算子内会按照query的shape中的S进行处理。 | INT64 | - | 1 | - |
| actualSeqLengthsKvOptional（aclIntArray*） | 输入 | key/value的实际序列长度数组。描述变长场景下每个Batch中实际有效的key/value token数量。 | 可选输入，用于变长序列场景：<br>• 当kvInputLayout为"TND"时：该项输入必须配置<br>• 当kvInputLayout为"BNSD"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；如不配置该项输入(传入nullptr)，算子内会按照key/value的shape中的S进行处理。 | INT64 | - | 1 | - |
| qInputLayout（char*） | 输入 | query的数据排布格式。指示输入张量在内存中的具体排布（如连续或合轴排列）。 | 当前仅支持"TND"、"BNSD"，qInputLayout与kvInputLayout需要保持一致。 | String | - | - | - |
| kvInputLayout（char*） | 输入 | key和value的数据排布格式。指示输入张量在内存中的具体排布。 | 当前仅支持"TND"、"BNSD"，qInputLayout与kvInputLayout需要保持一致。 | String | - | - | - |
| numKeyValueHeads（int64_t） | 输入 | key/value的注意力头数。用于支持GQA（分组查询注意力）机制下的头数比例映射。 | - | INT64 | - | - | - |
| maskType（int64_t） | 输入 | 注意力计算中的掩码类型。指定采用何种预设规则的掩码逻辑。 | 当前只支持传 0：代表不加mask场景 | INT64 | - | - | - |
| scaleValue（double） | 输入 | 缩放系数，即公式中的scale。用于注意力分数的归一化处理。 | 一般设置为D^-0.5。 | DOUBLE | - | - | - |
| preTokens（int64_t） | 输入 | 滑窗向前包含的token数量。限制当前token只能与前方的多少个历史token计算注意力。 | 用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。 | INT64 | - | - | - |
| nextTokens（int64_t） | 输入 | 滑窗向后包含的token数量。限制当前token只能与后方的多少个未来token计算注意力。 | 用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。 | INT64 | - | - | - |
| dq（aclTensor*） | 输出 | query的梯度输出结果，即公式中的dq。 | 数据类型和shape与输入query保持一致。 | FLOAT16、BFLOAT16 | ND | 3-4 | √ |
| dk（aclTensor*） | 输出 | key的梯度输出结果，即公式中的dk。 | 数据类型和shape与输入key保持一致。 | FLOAT16、BFLOAT16 | ND | 3-4 | √ |
| dv（aclTensor*） | 输出 | value的梯度输出结果，即公式中的dv。 | 数据类型和shape与输入value保持一致。 | FLOAT16、BFLOAT16 | ND | 3-4 | √ |
| workspaceSize（uint64_t*） | 输出 | 工作空间大小。返回算子执行时需要底层额外预留的临时内存字节数。 | - | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 算子执行器。返回包含算子实际计算流程、资源分配和参数调度的执行器指针。 | - | - | - | - | - |

* **返回值：**                                             |

    aclnnStatus：返回状态码，具体参见[aclnn返回码](https://wiki.huawei.com/domains/docs/context/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

    | 返回码 | 错误码 | 描述 |
    | - | - | - |
    | ACLNN_ERR_PARAM_NULLPTR | 161001 | 有必选参数传入了空指针，包含以下场景：<br>• 输入dout，query，key，value，attentionOut传入的是空指针。<br>• qInputLayout为"TND"时，actualSeqLengthsOptional传入的是空指针。<br>• kvInputLayout为"TND"时，actualSeqLengthsKvOptional传入的是空指针。 |
    | ACLNN_ERR_PARAM_INVALID | 161002 | 参数有效性校验失败，包含以下场景：<br>• dout，query，key，value 数据类型不在支持的范围之内。<br>• qInputLayout或kvInputLayout输入不合法。 |

### aclnnBlockSparseAttentionGrad

* **参数说明：**

    | 参数名 | 输入/输出 | 描述 |
    | - | - | - |
    | workspace | 输入 | 在Device侧申请的workspace内存地址。 |
    | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnRainFusionAttentionGetWorkspaceSize获取。 |
    | executor | 输入 | op执行器，包含了算子计算流程。 |
    | stream | 输入 | 指定执行任务的AscendCL stream流。 |  
  
* **返回值：**
  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://wiki.huawei.com/domains/docs/context/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

## 约束说明

* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
* attentionMaskOptional当前只支持传入nullptr。
* actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
* qSeqlen和kvSeqlen不需要被blockShape整除，支持非对齐场景，实际分块数通过向上取整计算。
* 根据算子支持的输入 Layout，query 张量 Shape 中对应的 head 维度大小记为 N1，key 和 value 张量 Shape 中对应的 head 维度大小记为 N2。必须满足 N1 >= N2 且 N1 % N2 == 0。(例如：在 BNSD 布局下，N1 对应 query 的第 2 维，N2 对应 key/value 的第 2 维)
* headdim <= 128

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://wiki.huawei.com/domains/docs/context/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%A0%B7%E4%BE%8B.md)。
```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_block_sparse_attention_grad.cpp
 * \brief BlockSparseAttentionGrad 算子测试用例 (BNSD Layout)
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "../op_host/op_api/aclnn_block_sparse_attention_grad.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
    
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置核心参数 (以 BNSD Layout 为例)
    int32_t batch = 1;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t headDim = 128;
    int32_t blockShapeX = 64;
    int32_t blockShapeY = 64;

    // 块数量计算
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilKv = (kvSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. 构建张量 Shape
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kvShape = {batch, numKvHeads, kvSeqlen, headDim};
    std::vector<int64_t> lseShape = {batch, numHeads, qSeqlen}; // LSE 通常没有尾部 1 维，防止 GE squeeze
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilKv};

    // 4. 分配并初始化 Host 数据
    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);

    // 将 Q, K, V 初始化为 0.1f 等较小的数
    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kvSize, 0.1f);
    std::vector<op::fp16_t> vData(kvSize, 0.1f);
    
    // 梯度初始值可以给一个小正数
    std::vector<op::fp16_t> doutData(qSize, 0.01f);
    std::vector<op::fp16_t> outData(qSize, 0.1f);
    
    // LSE 给一个合理的正数，比如 5.0f，这样 exp(S - LSE) 就是一个非常安全的负指数，绝对不会溢出
    std::vector<float> lseData(GetShapeSize(lseShape), 5.0f);
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 1);

    // 创建所有的前向输入/输出 aclTensor
    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr;
    void *lseAddr = nullptr, *maskAddr = nullptr;
    
    aclTensor *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *doutTensor = nullptr, *outTensor = nullptr;
    aclTensor *lseTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(vData, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CreateAclTensor(doutData, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &doutTensor);
    CreateAclTensor(outData, qShape, &outAddr, aclDataType::ACL_FLOAT16, &outTensor);
    
    CreateAclTensor(lseData, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lseTensor);     // 严格使用 FP32
    CreateAclTensor(maskData, maskShape, &maskAddr, aclDataType::ACL_UINT8, &maskTensor); // 严格使用 UINT8

    // 5. 创建反向输出梯度 (dq, dk, dv)
    std::vector<op::fp16_t> dqData(qSize, 0.0f);
    std::vector<op::fp16_t> dkData(kvSize, 0.0f);
    std::vector<op::fp16_t> dvData(kvSize, 0.0f);
    
    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;

    CreateAclTensor(dqData, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dqTensor);
    CreateAclTensor(dkData, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dkTensor);
    CreateAclTensor(dvData, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dvTensor);

    // 6. 创建 aclIntArray 属性参数 (BlockShape & ActualSeqLengths)
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());
    
    std::vector<int64_t> qSeqLenVec(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> kvSeqLenVec(batch, static_cast<int64_t>(kvSeqlen));
    aclIntArray *qSeqLenArr = aclCreateIntArray(qSeqLenVec.data(), batch);
    aclIntArray *kvSeqLenArr = aclCreateIntArray(kvSeqLenVec.data(), batch);

    // 7. 标量与字符串参数配置
    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    // 强制规定滑动窗口极大值
    int64_t preTokens = 2147483647; 
    int64_t nextTokens = 2147483647;

    // 8. 调用第一段接口: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnBlockSparseAttentionGradGetWorkspaceSize...\n");
    ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
        doutTensor, 
        qTensor, 
        kTensor, 
        vTensor, 
        outTensor, 
        lseTensor, 
        maskTensor,                 // blockSparseMaskOptional
        nullptr,                    // attenMaskOptional 必须为空
        blockShapeArr, 
        qSeqLenArr, 
        kvSeqLenArr, 
        qLayoutBuffer, 
        kvLayoutBuffer, 
        static_cast<int64_t>(numKvHeads), 
        maskType, 
        scaleValue, 
        preTokens, 
        nextTokens, 
        dqTensor, 
        dkTensor, 
        dvTensor, 
        &workspaceSize, 
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 9. 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 10. 调用第二段接口: 执行计算
    LOG_PRINT("Calling aclnnBlockSparseAttentionGrad...\n");
    ret = aclnnBlockSparseAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 11. 同步 Stream，等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 12. 将结果拷贝回 Host 侧打印
    ret = aclrtMemcpy(dqData.data(), qSize * sizeof(op::fp16_t), dqAddr, qSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! Output results (first 10 elements of dQ):\n");
    for (uint64_t i = 0; i < 10 && i < dqData.size(); i++) {
        LOG_PRINT("  dQ index %lu: %f\n", i, static_cast<float>(dqData[i]));
    }

    // 13. 释放所有资源
    LOG_PRINT("Cleaning up resources...\n");
    if (workspaceAddr) aclrtFree(workspaceAddr);
    
    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(doutAddr); aclrtFree(outAddr);
    aclrtFree(lseAddr); aclrtFree(maskAddr);
    aclrtFree(dqAddr); aclrtFree(dkAddr); aclrtFree(dvAddr);

    aclDestroyTensor(qTensor); aclDestroyTensor(kTensor); aclDestroyTensor(vTensor);
    aclDestroyTensor(doutTensor); aclDestroyTensor(outTensor);
    aclDestroyTensor(lseTensor); aclDestroyTensor(maskTensor);
    aclDestroyTensor(dqTensor); aclDestroyTensor(dkTensor); aclDestroyTensor(dvTensor);
    
    aclDestroyIntArray(blockShapeArr);
    aclDestroyIntArray(qSeqLenArr);
    aclDestroyIntArray(kvSeqLenArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BlockSparseAttentionGrad Test completed successfully!\n");
    return 0;
}
```
