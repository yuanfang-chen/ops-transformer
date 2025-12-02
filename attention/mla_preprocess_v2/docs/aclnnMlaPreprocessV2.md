# aclnnMlaPreprocessV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200/300/500 推理产品</term>|      ×     |

## 功能说明
-  **算子功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程如下：
    -  首先对输入$x$ RmsNormQuant后乘以$W^{DQKV}$进行下采样后分为通路1和通路2。
    -  通路1做RmsNormQuant后乘以$W^{UQ}$后再分为通路3和通路4。
    -  通路3后乘以$W^{uk}$后输出$q^N$。
    -  通路4后经过旋转位置编码后输出$q^R$。
    -  通路2拆分为通路5和通路6。
    -  通路5经过RmsNorm后传入Cache中得到$k^N$。
    -  通路6经过旋转位置编码后传入另一个Cache中得到$k^R$。


-  **计算公式**：

    RmsNormQuant公式
    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    RmsNormQuant(x) = ({RmsNorm}(x) + bias) * deqScale
    $$
  
    Query计算公式，包括W^{DQKV}矩阵乘、W^{UK}矩阵乘、RmsNormQuant和ROPE旋转位置编码处理

    $$
    q^N =  RmsNormQuant(x) \cdot W^{DQKV} \cdot W^{UK}
    $$

    $$
    q^R = ROPE(x^Q)
    $$

    Key计算公式，包括RmsNorm和rope，将计算结果存入cache

    $$
    k^N = Cache({RmsNorm}(RmsNormQuant(x)))
    $$

    $$
    k^R = Cache(ROPE(RmsNormQuant(x)))
    $$



## 实现原理

图1 计算流程图

![MlaPreprocess图](../../../docs/zh/figures/MlaPreprocess计算过程.png)

## 算子执行接口
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMlaPreprocessGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnMlaPreprocess”接口执行计算。

* `aclnnStatus aclnnMlaPreprocessV2GetWorkspaceSize(const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0, const aclTensor *quantOffset0,const aclTensor *wdqkv, const aclTensor *deScale0, const aclTensor *bias0, const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1, const aclTensor *wuq, const aclTensor *deScale1, const aclTensor *bias1, const aclTensor *gamma2, const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache, const aclTensor *kvCacheRope, const aclTensor *slotMapping, const aclTensor *ctkvScale, const aclTensor *qNopeScale, int64_t wdqDim, int64_t qRopeDim, int64_t kRopeDim, float epsilon, int64_t qRotaryCoeff, int64_t kRotaryCoeff, bool transposeWdq, bool transposeWuq, bool transposeWuk, int64_t cacheMode, int64_t quantMode, bool doRmsNorm, int64_t wdkvSplitCount, bool qDownOutFlag, aclTensor *qOut, aclTensor *kvCacheOut, aclTensor *qRopeOut, aclTensor *krCacheOut, aclTensor *qDownOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMlaPreprocessV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMlaPreprocessGetWorkspaceSize
- **参数说明：**
  - input（aclTensor*，计算输入）：Device侧的aclTensor，用于计算Query和Key的x，shape为[tokenNum,hiddenSize]，dtype支持FLOAT16和BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - gamma0（aclTensor*，计算输入）：Device侧的aclTensor，首次RmsNorm计算中的γ参数，shape为[hiddenSize]，dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - beta0（aclTensor*，计算输入）：Device侧的aclTensor，首次RmsNorm计算中的β参数，shape为[hiddenSize]，dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - quantScale0（aclTensor*，计算输入）：Device侧的aclTensor，首次RmsNorm公式中量化缩放的参数，shape为[1]，dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - quantOffset0（aclTensor*，计算输入）：Device侧的aclTensor，首次RmsNorm公式中的量化偏移参数，shape为[1]，dtype支持INT8，[数据格式](../../../docs/zh/context/数据格式.md)支持NZ格式。

  - wdqkv（aclTensor*，计算输入）：Device侧的aclTensor，与输入首次做矩阵乘的降维矩阵，shape为[2112,hiddenSize]，dtype支持INT8和BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - deScale0（aclTensor*，计算输入）：Device侧的aclTensor，输入首次做矩阵乘的降维矩阵中的系数，shape为[2112]，dtype支持INT64和FLOAT，与input的dtype对应，input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT。[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。
  
  - bias0（aclTensor*，计算输入）：Device侧的aclTensor，输入首次做矩阵乘的降维矩阵中的系数，shape为[2112]，dtype支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。支持传入空tensor，quantMode为1、3时不传入。
  
  - gamma1（aclTensor*，计算输入）：Device侧的aclTensor，第二次RmsNorm计算中的γ参数，shape为[1536]。dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - beta1（aclTensor*，计算输入）：Device侧的aclTensor，第二次RmsNorm计算中的β参数，shape为[1536]。dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - quantScale1（aclTensor*，计算输入）：Device侧的aclTensor，第二次RmsNorm公式中量化缩放的参数，shape为[1536]。dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。仅在quantMode为0时传入。
  
  - quantOffset1（aclTensor*，计算输入）：Device侧的aclTensor，第二次RmsNorm公式中的量化偏移参数，shape为[1]。dtype支持INT8，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。仅在quantMode为0时传入。
  
  - wuq（aclTensor*，计算输入）：Device侧的aclTensor，权重矩阵，shape为[headNum * 192,1536]。dtype支持INT8和BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持NZ格式。

  - deScale1（aclTensor*，计算输入）：Device侧的aclTensor，参与wuq矩阵乘的系数，shape为[headNum*192,1536]。dtype支持INT64和FLOAT，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT。
  
  - bias1（aclTensor*，计算输入）：Device侧的aclTensor，参与wuq矩阵乘的系数，shape为[[headNum*192]]。dtype支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)支持NZ格式。quantMode为1、3时不传入。
  
  - gamma2（aclTensor*，计算输入）：Device侧的aclTensor，参与RmsNormAndreshapeAndCache计算的γ参数，shape为[512]。dtype支持BLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - cos（aclTensor*，计算输入）：Device侧的aclTensor，表示用于计算旋转位置编码的正弦参数矩阵，shape为[tokenNum,64]。dtype支持INT8，[数据格式](../../../docs/zh/context/数据格式.md)支持NZ格式。

  - sin（aclTensor*，计算输入）：Device侧的aclTensor，表示用于计算旋转位置编码的余弦参数矩阵，shape为[tokenNum,64]。dtype支持INT8，[数据格式](../../../docs/zh/context/数据格式.md)支持NZ格式。

  - wuk（aclTensor*，计算输入）：Device侧的aclTensor，表示计算Key的上采样权重，shape为[headNum * 192, 1536]。dtype支持FLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND/NZ格式。ND格式时的shape为[headNum,128,512]，NZ格式时的shape为[headNum,32,128,16]。
  
  - kvCache（aclTensor*，计算输入）：Device侧的aclTensor，与输出的kvCacheOut为同一tensor，输入格式随cacheMode变化。
    - cacheMode为0：shape为[blockNum,blockSize,1,576]，dtype与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为1：shape为[blockNum,blockSize,1,512]，tensor的shape为拆分情况，dtype与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为2：shape为[blockNum,headNum*512/32,block_size,32]，dtype为int8，[数据格式](../../../docs/zh/context/数据格式.md)为NZ。
    - cacheMode为3：shape为[blockNum,headNum*512/16,block_size,16]，dtype与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)为NZ。
  
  - kvCacheRope（aclTensor*，计算输入）Device侧的aclTensor，可选参数，支出传入空指针。与输出的krCacheOut为同一tensor，输入格式随cacheMode变化。
    - cacheMode为0：不传入。
    - cacheMode为1：shape为[blockNum,blockSize,1,64]，dtype与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为2或3：shape为[blockNum, headNum*64 / 16 ,block_size, 16]，dtype与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)为NZ。
  
  - slotMapping（aclTensor*，计算输入）：Device侧的aclTensor，表示用于存储kv_cache和kr_cache的索引，shape为[tokenNum]。dtype支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - ctkvScale（aclTensor*，计算输入）：Device侧的aclTensor，输出量化处理中参与计算的系数，仅在cacheMode为2时传入，shape为[1]。dtype支持BLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - qNopeScale（aclTensor*，计算输入）：Device侧的aclTensor，输出量化处理中参与计算的系数，仅在cacheMode为2时传入，shape为[headNum]。dtype支持BLOAT16和BFLOAT16，与input保持一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND格式。

  - wdqDim（int64_t，计算输入）：表示经过matmul后拆分的dim大小。预留参数，目前只支持1536。

  - qRopeDim（int64_t，计算输入）：表示q传入rope的dim大小。预留参数，目前只支持64。

  - kRopeDim（int64_t，计算输入）：表示k传入rope的dim大小。预留参数，目前只支持64。

  - epsilon（float，计算输入）：表示加在分母上防止除0。

  - qRotaryCoeff（int64_t，计算输入）：表示q旋转系数。预留参数，目前只支持2。

  - kRotaryCoeff（int64_t，计算输入）：表示k旋转系数。预留参数，目前只支持2。

  - transposeWdq（bool，计算输入）：表示wdq是否转置。预留参数，目前只支持true。

  - transposeWuq（bool，计算输入）：表示wuq是否转置。预留参数，目前只支持true。

  - transposeWuk（bool，计算输入）：表示wuk是否转置。预留参数，目前只支持true。

  - cacheMode（int64_t，计算输入）：表示指定cache的类型，取值范围[0, 3]。
    - 0：kcache和q均经过拼接后输出。
    - 1：输出的kvCacheOut拆分为kvCacheOut和krCacheOut，qOut拆分为qOut和qRopeOut。
    - 2：krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型。
    - 3：krope和ctkv转为NZ格式输出。
  
  - quantMode（int64_t，计算输入）：表示指定RmsNorm量化的类型，取值范围[0, 3]。
    - 0：per_tensor静态非对称量化，默认量化类型。
    - 1：per_token动态对称量化，未实现。
    - 2：per_token动态非对称量化，未实现。
    - 3：不量化，浮点输出，未实现。
  
  - doRmsNorm（bool，计算输入）：表示是否对input输入进行RmsNormQuant操作，false表示不操作，true表示进行操作。预留参数，目前只支持true。

  - wdkvSplitCount（int64_t，计算输入）：表示指定wdkv拆分的个数，支持[1-3]，分别表示不拆分、拆分为2个、拆分为3个降维矩阵。预留参数，目前只支持1。

  - qDownOutFlag（bool，计算输入）：表示是否输出qDownOut，false表示不输出，true表示输出。

  - qOut（aclTensor*，计算输出）：计算输出，表示Query的输出tensor，对应计算流图中右侧经过NOPE和矩阵乘后的输出，shape和dtype随cacheMode变化。
    - cacheMode为0：shape为[tokenNum, headNum, 576]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为1或3：shape为[tokenNum, headNum, 512]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为2：shape为[tokenNum, headNum, 512]，dtype为INT8，[数据格式](../../../docs/zh/context/数据格式.md)为ND格式。
  
  - kvCacheOut（aclTensor*，计算输出）：计算输出，表示Key经过ReshapeAndCache后的输出，shape和dtype随cacheMode变化。
    - cacheMode为0：shape为[blockNum, blockSize, 1, 576]， dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为1：shape为[blockNum, blockSize, 1, 512]， dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为2：shape为[blockNum, headNum*512/32, block_size, 32]，dtype为INT8，[数据格式](../../../docs/zh/context/数据格式.md)为NZ。
    - cacheMode为3：shape为[blockNum, headNum*512/16, block_size, 16]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为NZ。
  
  - qRopeOut（aclTensor*，计算输出）：计算输出，表示Query经过旋转编程后的输出，shape和dtype随cacheMode变化。
    - cacheMode为0：不输出。
    - cacheMode为1或3：shape为[tokenNum, headNum, 64]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为2：shape为[tokenNum, headNum, 64]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。

  - krCacheOut（aclTensor*，计算输出）：表示Key经过ROPE和ReshapeAndCache后的输出，shape和[数据格式](../../../docs/zh/context/数据格式.md)随cacheMode变化，
    - cacheMode为0：不输出。
    - cacheMode为1：shape为[blockNum, blockSize, 1, 64]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
    - cacheMode为2或3：shape为[blockNum, headNum*64 / 16 ,block_size, 16]，dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为NZ。
  
  - qDownOut（aclTensor*，计算输出）：表示Query经过降维后的输出，shape为[tokenNum, 1536], dtype与input一致，[数据格式](../../../docs/zh/context/数据格式.md)为ND。
  
  - workspaceSize（uint64_t*，出参）：返回用户需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：必须传入的参数中存在空指针。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：输入参数的shape、dtype和数据类型不在支持的范围之内。
  - 返回361001（ACLNN_ERR_RUNTIME_ERROR）：API内存调用npu runtime的接口异常。
  - 返回561002 (ACLNN_ERR_INNER_TILING_ERROR) : tiling发生异常，入参的dtype类型或者shape错误。
  ```

### aclnnMlaPreprocess

- **参数说明：**

  * workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMlaPreprocessGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   shape格式字段含义及约束
    -  tokenNum：tokenNum 表示输入样本批量大小，取值范围：0~256
    -  hiddenSize：hiddenSize 表示隐藏层的大小，取值固定为：2048-10240，为256的倍数
    -  headNum：表示多头数，取值范围：16、32、64、128
    -  blockNum：PagedAttention场景下的块数，取值范围：192
    -  blockSize：PagedAttention场景下的块大小，取值范围：128
    -  当wdqkv和wuq的数据类型为bfloat16时，输入input也需要为bflot16，且hiddenSize只支持6144

## 算子原型
```
REG_OP(MlaPreprocess)
    .INPUT(input, ge::TensorType::ALL())
    .INPUT(gamma_0, ge::TensorType::ALL())
    .INPUT(beta_0, ge::TensorType::ALL())
    .INPUT(quant_scale_0, ge::TensorType::ALL())
    .INPUT(quant_offset_0, ge::TensorType::ALL())
    .INPUT(wdqkv, ge::TensorType::ALL())
    .INPUT(descale_0, ge::TensorType::ALL())
    .INPUT(bias_0, ge::TensorType::ALL())
    .INPUT(gamma_1, ge::TensorType::ALL())
    .INPUT(beta_1, ge::TensorType::ALL())
    .INPUT(quant_scale_1, ge::TensorType::ALL())
    .INPUT(quant_offset_1, ge::TensorType::ALL())
    .INPUT(wuq, ge::TensorType::ALL())
    .INPUT(descale_1, ge::TensorType::ALL())
    .INPUT(bias_1, ge::TensorType::ALL())
    .INPUT(gamma_2, ge::TensorType::ALL())
    .INPUT(cos, ge::TensorType::ALL())
    .INPUT(sin, ge::TensorType::ALL())
    .INPUT(wuk, ge::TensorType::ALL())
    .INPUT(kv_cache, ge::TensorType::ALL())
    .INPUT(kv_cache_rope, ge::TensorType::ALL())
    .INPUT(slot_mapping, ge::TensorType::ALL())
    .INPUT(ctkv_scale, ge::TensorType::ALL())
    .INPUT(q_nope_scale, ge::TensorType::ALL())
    .OUTPUT(q_out, ge::TensorType::ALL())
    .OUTPUT(kv_cache_out, ge::TensorType::ALL())
    .OUTPUT(q_rope_out, ge::TensorType::ALL())
    .OUTPUT(kr_cache_out, ge::TensorType::ALL())
    .OUTPUT(q_down_out, ge::TensorType::ALL())
    .ATTR(wdq_dim, Int, 128)
    .ATTR(q_rope_dim, Int, 64)
    .ATTR(k_rope_dim, Int, 64)
    .ATTR(epsilon, Float, 1e-05)
    .ATTR(q_rotary_coeff, Int, 2)
    .ATTR(k_rotary_coeff, Int, 2)
    .ATTR(transepose_wdq, Bool, true)
    .ATTR(transepose_wuq, Bool, true)
    .ATTR(transepose_wuk, Bool, true)
    .ATTR(cache_mode, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(do_rms_norm, Bool, true)
    .ATTR(wdkv_split_count, Int, 1)
    .OP_END_FACTORY_REG(MlaPreprocess);

```


参数解释请参见**算子执行接口**。

## 调用示例
通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_mla_preprocess.h"

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i; 
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.reserve(fileSize);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "读取文件失败" << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr, int num)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}


template <typename T>
int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size,  ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc ND tensor device failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMalloc申请host侧内存
    ret = aclrtMalloc(hostAddr, size,   ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc ND tensor host failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr,   GetShapeSize(shape)*aclDataTypeSize(dataType),  ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy  failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorNZ(const std::vector<T>& shape,  void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor**   tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size,  ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc NZ tensor device failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMalloc申请host侧内存
    ret = aclrtMalloc(hostAddr, size,   ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc NZ tensor device failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size  (), dataType, nullptr, 0,   aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size  (), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr,   GetShapeSize(shape)*aclDataTypeSize(dataType),  ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy  failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int TransToNZShape(std::vector<int64_t> &shapeND, size_t  typeSize) {
    int64_t h = shapeND[0];
    int64_t w = shapeND[1];
    int64_t h0 = 16;
    int64_t w0 = 32U / typeSize;
    int64_t h1 = h / h0;
    int64_t w1 = w / w0;
    shapeND[0] = w1;
    shapeND[1] = h1;
    shapeND.emplace_back(h0);
    shapeND.emplace_back(w0);
    return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 5;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);
  //属性
  int64_t tokenNum = 8;
  int64_t hiddenNum = 7168;
  int64_t headNum = 32;
  int64_t blockNum = 192;
  int64_t blockSize = 128;

  int64_t wdqDim = 128;
  int64_t qRopeDim = 0; 
  int64_t kRopeDim = 0;
  float epsilon = 1e-05f;
  int64_t qRotaryCoeff = 2;
  int64_t kRotaryCoeff = 2;
  bool transposeWdq = true;
  bool transposeWuq = true;
  bool transposeWuk = true;
  int64_t cacheMode =  1;
  int64_t quantMode =  0;
  bool doRmsNorm = true;
  int64_t wdkvSplitCount = 1;

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {tokenNum, hiddenNum};
  std::vector<int64_t> gamma0Shape = {hiddenNum};
  std::vector<int64_t> beta0Shape = {hiddenNum};
  std::vector<int64_t> quantScale0Shape = {1};
  std::vector<int64_t> quantOffset0Shape = {1};
  std::vector<int64_t> wdqkvShape = {2112, hiddenNum};
  std::vector<int64_t> deScale0Shape = {2112};
  std::vector<int64_t> bias0Shape = {2112};
  std::vector<int64_t> gamma1Shape = {1536};
  std::vector<int64_t> beta1Shape = {1536};
  std::vector<int64_t> quantScale1Shape = {1};
  std::vector<int64_t> quantOffset1Shape = {1};
  std::vector<int64_t> wuqShape = {headNum * 192, 1536};
  std::vector<int64_t> deScale1Shape = {headNum * 192};
  std::vector<int64_t> bias1Shape = {headNum * 192};
  std::vector<int64_t> gamma2Shape = {512};
  std::vector<int64_t> cosShape = {tokenNum, 64};
  std::vector<int64_t> sinShape = {tokenNum, 64};
  std::vector<int64_t> wukShape = {headNum, 128, 512};
  std::vector<int64_t> kvCacheShape = {blockNum, blockSize, 1, 576};
  std::vector<int64_t> kvCacheRopeShape = {blockNum, blockSize, 1, 64};
  std::vector<int64_t> slotMappingShape = {tokenNum};
  std::vector<int64_t> ctkvScaleShape = {1};
  std::vector<int64_t> qNopeScaleShape = {headNum};

  std::vector<int64_t> qOutShape = {tokenNum, headNum, 576};
  std::vector<int64_t> kvCacheOutShape = {blockNum, blockSize, 1, 576};
  std::vector<int64_t> qRopeOutShape = {tokenNum, headNum, 64};
  std::vector<int64_t> krCacheOutShape = {blockNum, blockSize, 1, 64};

  void* inputDeviceAddr = nullptr;
  void* gamma0DeviceAddr = nullptr;
  void* beta0DeviceAddr = nullptr;
  void* quantScale0DeviceAddr = nullptr;
  void* quantOffset0DeviceAddr = nullptr;
  void* wdqkvDeviceAddr = nullptr;
  void* deScale0DeviceAddr = nullptr;
  void* bias0DeviceAddr = nullptr;
  void* gamma1DeviceAddr = nullptr;
  void* beta1DeviceAddr = nullptr;
  void* quantScale1DeviceAddr = nullptr;
  void* quantOffset1DeviceAddr = nullptr;
  void* wuqDeviceAddr = nullptr;
  void* deScale1DeviceAddr = nullptr;
  void* bias1DeviceAddr = nullptr;
  void* gamma2DeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* wukDeviceAddr = nullptr;
  void* kvCacheDeviceAddr = nullptr;
  void* kvCacheRopeDeviceAddr = nullptr;
  void* slotMappingDeviceAddr = nullptr;
  void* ctkvScaleDeviceAddr = nullptr;
  void* qNopeScaleDeviceAddr = nullptr;
  void* qOutDeviceAddr = nullptr;
  void* kvCacheOutDeviceAddr = nullptr;
  void* qRopeOutDeviceAddr = nullptr;
  void* krCacheOutDeviceAddr = nullptr;

  void* inputHostAddr = nullptr;
  void* gamma0HostAddr = nullptr;
  void* beta0HostAddr = nullptr;
  void* quantScale0HostAddr = nullptr;
  void* quantOffset0HostAddr = nullptr;
  void* wdqkvHostAddr = nullptr;
  void* deScale0HostAddr = nullptr;
  void* bias0HostAddr = nullptr;
  void* gamma1HostAddr = nullptr;
  void* beta1HostAddr = nullptr;
  void* quantScale1HostAddr = nullptr;
  void* quantOffset1HostAddr = nullptr;
  void* wuqHostAddr = nullptr;
  void* deScale1HostAddr = nullptr;
  void* bias1HostAddr = nullptr;
  void* gamma2HostAddr = nullptr;
  void* cosHostAddr = nullptr;
  void* sinHostAddr = nullptr;
  void* wukHostAddr = nullptr;
  void* kvCacheHostAddr = nullptr;
  void* kvCacheRopeHostAddr = nullptr;
  void* slotMappingHostAddr = nullptr;
  void* ctkvScaleHostAddr = nullptr;
  void* qNopeScaleHostAddr = nullptr;
  void* qOutHostAddr = nullptr;
  void* kvCacheOutHostAddr = nullptr;
  void* qRopeOutHostAddr = nullptr;
  void* krCacheOutHostAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* gamma0 = nullptr;
  aclTensor* beta0 = nullptr;
  aclTensor* quantScale0 = nullptr;
  aclTensor* quantOffset0 = nullptr;
  aclTensor* wdqkv = nullptr;
  aclTensor* deScale0 = nullptr;
  aclTensor* bias0 = nullptr;
  aclTensor* gamma1 = nullptr;
  aclTensor* beta1 = nullptr;
  aclTensor* quantScale1 = nullptr;
  aclTensor* quantOffset1 = nullptr;
  aclTensor* wuq = nullptr;
  aclTensor* deScale1 = nullptr;
  aclTensor* bias1 = nullptr;
  aclTensor* gamma2 = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* wuk = nullptr;
  aclTensor* kvCache = nullptr;
  aclTensor* kvCacheRope = nullptr;
  aclTensor* slotMapping = nullptr;
  aclTensor* ctkvScale = nullptr;
  aclTensor* qNopeScale = nullptr;
  aclTensor* qOut = nullptr;
  aclTensor* kvCacheOut = nullptr;
  aclTensor* qRopeOut = nullptr;
  aclTensor* krCacheOut = nullptr;

  // 转换三个NZ格式变量的shape
  ret = TransToNZShape(wdqkvShape, sizeof(int8_t));
  CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed. \n"); return ret);
  ret = TransToNZShape(wuqShape, sizeof  (int8_t));
  CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed. \n"); return ret);

  ret = CreateAclTensorND(inputShape, &inputDeviceAddr, &inputHostAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(gamma0Shape, &gamma0DeviceAddr, &gamma0HostAddr, aclDataType::ACL_FLOAT16, &gamma0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(beta0Shape, &beta0DeviceAddr, &beta0HostAddr, aclDataType::ACL_FLOAT16, &beta0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(quantScale0Shape, &quantScale0DeviceAddr, &quantScale0HostAddr, aclDataType::ACL_FLOAT16, &quantScale0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(quantOffset0Shape, &quantOffset0DeviceAddr, &quantOffset0HostAddr, aclDataType::ACL_INT8, &quantOffset0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //wdqkv转为NZ
  ret = CreateAclTensorNZ(wdqkvShape, &wdqkvDeviceAddr, &wdqkvHostAddr, aclDataType::ACL_INT8, &wdqkv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //fp16输入，则这里转int64
  ret = CreateAclTensorND(deScale0Shape, &deScale0DeviceAddr, &deScale0HostAddr, aclDataType::ACL_INT64, &deScale0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(bias0Shape, &bias0DeviceAddr, &bias0HostAddr, aclDataType::ACL_INT32, &bias0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(gamma1Shape, &gamma1DeviceAddr, &gamma1HostAddr, aclDataType::ACL_FLOAT16, &gamma1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(beta1Shape, &beta1DeviceAddr, &beta1HostAddr, aclDataType::ACL_FLOAT16, &beta1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(quantScale1Shape, &quantScale1DeviceAddr, &quantScale1HostAddr, aclDataType::ACL_FLOAT16, &quantScale1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(quantOffset1Shape, &quantOffset1DeviceAddr, &quantOffset1HostAddr, aclDataType::ACL_INT8, &quantOffset1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //wuq转为NZ
  ret = CreateAclTensorNZ(wuqShape, &wuqDeviceAddr, &wuqHostAddr, aclDataType::ACL_INT8, &wuq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //fp16输入，则这里转int64
  ret = CreateAclTensorND(deScale1Shape, &deScale1DeviceAddr, &deScale1HostAddr, aclDataType::ACL_INT64, &deScale1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(bias1Shape, &bias1DeviceAddr, &bias1HostAddr, aclDataType::ACL_INT32, &bias1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(gamma2Shape, &gamma2DeviceAddr, &gamma2HostAddr, aclDataType::ACL_FLOAT16, &gamma2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(cosShape, &cosDeviceAddr, &cosHostAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(sinShape, &sinDeviceAddr, &sinHostAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(wukShape, &wukDeviceAddr, &wukHostAddr, aclDataType::ACL_FLOAT16, &wuk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_FLOAT16, &kvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(kvCacheRopeShape, &kvCacheRopeDeviceAddr, &kvCacheRopeHostAddr, aclDataType::ACL_FLOAT16, &kvCacheRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(slotMappingShape, &slotMappingDeviceAddr, &slotMappingHostAddr, aclDataType::ACL_INT32, &slotMapping);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(ctkvScaleShape, &ctkvScaleDeviceAddr, &ctkvScaleHostAddr, aclDataType::ACL_FLOAT16, &ctkvScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(qNopeScaleShape, &qNopeScaleDeviceAddr, &qNopeScaleHostAddr, aclDataType::ACL_FLOAT16, &qNopeScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(qOutShape, &qOutDeviceAddr, &qOutHostAddr, aclDataType::ACL_FLOAT16, &qOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(kvCacheOutShape, &kvCacheOutDeviceAddr, &kvCacheOutHostAddr, aclDataType::ACL_FLOAT16, &kvCacheOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(qRopeOutShape, &qRopeOutDeviceAddr, &qRopeOutHostAddr, aclDataType::ACL_FLOAT16, &qRopeOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorND(krCacheOutShape, &krCacheOutDeviceAddr, &krCacheOutHostAddr, aclDataType::ACL_FLOAT16, &krCacheOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用acaclnnMlaPreprocess第一段接口
  ret = aclnnMlaPreprocessGetWorkspaceSize(
    input, gamma0, beta0, quantScale0, quantOffset0,
    wdqkv, deScale0, bias0, gamma1, beta1, quantScale1, quantOffset1, wuq, deScale1, bias1, gamma2, cos, sin, wuk, kvCache, kvCacheRope, slotMapping, ctkvScale, qNopeScale,
    wdqDim, qRopeDim, kRopeDim, epsilon, qRotaryCoeff, kRotaryCoeff, transposeWdq, transposeWuq, transposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount, qOut, kvCacheOut, qRopeOut, krCacheOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("acaclnnMlaPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用acaclnnMlaPreprocess第二段接口
  ret = aclnnMlaPreprocess(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("acaclnnMlaPreprocess failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto qOutSize = GetShapeSize(qOutShape);
  std::vector<float> qOutData(qOutSize, 0);
  ret = aclrtMemcpy(qOutData.data(), qOutData.size() * sizeof(qOutData[0]), qOutDeviceAddr, qOutSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  // 释放aclTensor资源
  aclDestroyTensor(input);
  aclDestroyTensor(gamma0);
  aclDestroyTensor(beta0);
  aclDestroyTensor(quantScale0);
  aclDestroyTensor(quantOffset0);
  aclDestroyTensor(wdqkv);
  aclDestroyTensor(deScale0);
  aclDestroyTensor(bias0);
  aclDestroyTensor(gamma1);
  aclDestroyTensor(beta1);
  aclDestroyTensor(quantScale1);
  aclDestroyTensor(quantOffset1);
  aclDestroyTensor(wuq);
  aclDestroyTensor(deScale1);
  aclDestroyTensor(bias1);
  aclDestroyTensor(gamma2);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(wuk);
  aclDestroyTensor(kvCache);
  aclDestroyTensor(kvCacheRope);
  aclDestroyTensor(slotMapping);
  aclDestroyTensor(ctkvScale);
  aclDestroyTensor(qNopeScale);

  // 7. 释放device 资源
  aclrtFree(inputDeviceAddr);
  aclrtFree(gamma0DeviceAddr);
  aclrtFree(beta0DeviceAddr);
  aclrtFree(quantScale0DeviceAddr);
  aclrtFree(quantOffset0DeviceAddr);
  aclrtFree(wdqkvDeviceAddr);
  aclrtFree(deScale0DeviceAddr);
  aclrtFree(bias0DeviceAddr);
  aclrtFree(gamma1DeviceAddr);
  aclrtFree(beta1DeviceAddr);
  aclrtFree(quantScale1DeviceAddr);
  aclrtFree(quantOffset1DeviceAddr);
  aclrtFree(wuqDeviceAddr);
  aclrtFree(deScale1DeviceAddr);
  aclrtFree(bias1DeviceAddr);
  aclrtFree(gamma2DeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(wukDeviceAddr);
  aclrtFree(kvCacheDeviceAddr);
  aclrtFree(kvCacheRopeDeviceAddr);
  aclrtFree(slotMappingDeviceAddr);
  aclrtFree(ctkvScaleDeviceAddr);
  aclrtFree(qNopeScaleDeviceAddr);

  // 8. 释放host 资源
  aclrtFree(inputHostAddr);
  aclrtFree(gamma0HostAddr);
  aclrtFree(beta0HostAddr);
  aclrtFree(quantScale0HostAddr);
  aclrtFree(quantOffset0HostAddr);
  aclrtFree(wdqkvHostAddr);
  aclrtFree(deScale0HostAddr);
  aclrtFree(bias0HostAddr);
  aclrtFree(gamma1HostAddr);
  aclrtFree(beta1HostAddr);
  aclrtFree(quantScale1HostAddr);
  aclrtFree(quantOffset1HostAddr);
  aclrtFree(wuqHostAddr);
  aclrtFree(deScale1HostAddr);
  aclrtFree(bias1HostAddr);
  aclrtFree(gamma2HostAddr);
  aclrtFree(cosHostAddr);
  aclrtFree(sinHostAddr);
  aclrtFree(wukHostAddr);
  aclrtFree(kvCacheHostAddr);
  aclrtFree(kvCacheRopeHostAddr);
  aclrtFree(slotMappingHostAddr);
  aclrtFree(ctkvScaleHostAddr);
  aclrtFree(qNopeScaleHostAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
