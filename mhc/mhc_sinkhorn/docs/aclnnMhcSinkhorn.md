# aclnnMhcSinkhorn

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_sinkhorn)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：aclnnMhcSinkhorn是mHC架构的核心算子接口，通过Sinkhorn-Knopp迭代算法将mHC层初始混合矩阵投影到双随机矩阵流形（Birkhoff多胞形），生成满足行和、列和均为1的h_res变换矩阵，为aclnnMhcPost算子提供关键输入，稳定深度网络信号传播、解决梯度消失/爆炸问题。

- 核心迭代公式：

  $$
  \begin{align}
  M^{(k+1)} &= M^{(k)} \oslash (1_n \cdot (M^{(k)})^T 1_n) \\
  M^{(k+2)} &= M^{(k+1)} \oslash ((M^{(k+1)} 1_n) \cdot 1_n^T) \\
  h_{res} &= M^{(k+2)}
  \end{align}
  $$
  其中：$M$ 为初始混合矩阵，$\oslash$ 为元素级除法，$1_n$ 为n维全1向量，$k$ 为迭代次数，迭代至矩阵行/列和与1的误差小于收敛阈值时停止。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnMhcSinkhornGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMhcSinkhorn"接口执行计算。

```c++
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(
    const aclTensor  *initMatrix,
    int32_t          maxIter,
    float            epsilon,
    aclTensor        *hRes,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMhcSinkhorn(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

aclnnMhcSinkhornGetWorkspaceSize
 	 
 	 参数说明：
 	 
 	 <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
    <col style="width: 145px">
    <col style="width: 90px">
    <col style="width: 441px">
    <col style="width: 158px">
    <col style="width: 186px">
    <col style="width: 80px">
    <col style="width: 155px">
    <col style="width: 145px">
    </colgroup>
      <thead>
          <tr>
              <th>参数名</th>
              <th>输入/输出</th>
              <th>描述</th>
              <th>使用说明</th>
              <th>数据类型</th>
              <th>数据格式</th>
              <th>维度(shape)</th>
              <th>非连续Tensor</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td>initMatrix</td>
              <td>输入</td>
              <td>待变换的mHC层初始混合矩阵，为超连接原始矩阵。</td>
              <td>需为非负矩阵</td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>[B,S,N,N]、[T,N,N]</td>
              <td>√</td>
          </tr>
          <tr>
              <td>maxIter</td>
              <td>输入</td>
              <td>Sinkhorn-Knopp迭代最大次数，控制迭代收敛过程。</td>
              <td>建议取值50~200</td>
              <td>INT32</td>
              <td>标量</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>epsilon</td>
              <td>输入</td>
              <td>收敛阈值，矩阵行/列和与1的误差小于该值时停止迭代。</td>
              <td>建议取值1e-6~1e-4</td>
              <td>FLOAT32</td>
              <td>标量</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>hRes</td>
              <td>输出</td>
              <td>经Sinkhorn变换后的双随机矩阵，作为aclnnMhcPost算子的hRes输入。</td>
              <td>维度与initMatrix保持一致</td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>[B,S,N,N]、[T,N,N]</td>
              <td>-</td>
          </tr>
          <tr>
              <td>workspaceSize</td>
              <td>输出</td>
              <td>返回需要在Device侧申请的workspace大小。</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>executor</td>
              <td>输出</td>
              <td>返回op执行器，包含了算子计算流程。</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
      </tbody>
  </table>
 	 
 	 返回值：
 	 
 	 返回 aclnnStatus 状态码，具体参见aclnn 返回码。
 	 
 	 第一段接口完成入参校验，出现以下场景时报错：
 	 
 	 <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
      <thead>
          <th>返回值</th>
          <th>错误码</th>
          <th>描述</th>
      </thead>
      <tbody>
          <tr>
              <td>ACLNN_ERR_PARAM_NULLPTR</td>
              <td>161001</td>
              <td>initMatrix、hRes存在空指针。</td>
          </tr>
          <tr>
              <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
              <td rowspan="4">161002</td>
              <td>initMatrix、hRes的数据类型不在支持的范围内。</td>
          </tr>
            <tr>
              <td>initMatrix的shape维度不在支持的范围内（需为二维N×N矩阵）。</td>
          </tr>
          <tr>
              <td>initMatrix为负矩阵，不满足非负约束。</td>
          </tr>
          <tr>
              <td>maxIter≤0或epsilon≤0，迭代参数不合法。</td>
          </tr>
      </tbody>
  </table>
 	 
 	 aclnnMhcSinkhorn
 	 
 	 参数说明：
 	 
 	 <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
    </colgroup>
      <thead>
          <tr>
              <th>参数名</th>
              <th>输入/输出</th>
              <th>描述</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td>workspace</td>
              <td>输入</td>
              <td>在Device侧申请的workspace内存地址。</td>
          </tr>
          <tr>
              <td>workspaceSize</td>
              <td>输入</td>
              <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcSinkhornGetWorkspaceSize获取。</td>
          </tr>
          <tr>
              <td>executor</td>
              <td>输入</td>
              <td>op执行器，包含了算子计算流程。</td>
          </tr>
          <tr>
              <td>stream</td>
              <td>输入</td>
              <td>指定执行任务的AscendCL stream流。</td>
          </tr>
      </tbody>
  </table>
 	 
 	 返回值：
 	 
 	 返回 aclnnStatus 状态码，具体参见aclnn 返回码。
 	 
 	 约束说明
 	 
 	 aclnnMhcSinkhorn 默认确定性实现。
 	 
 	 输入 initMatrix 需为二维非负矩阵（N×N），确保迭代后可形成双随机矩阵。
 	 
 	 maxIter 建议取值范围 1~100，epsilon 建议取值范围 1e-6~1e-4，平衡收敛效果与计算效率。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: BSNxN format (B, S, N, N)
    std::vector<int64_t> initMatrixShape = {1, 1024, 4, 4};   // BSNxN
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};         // BSNxN
    int32_t maxIter = 100;                                     // 迭代最大次数
    float epsilon = 1e-6f;                                     // 收敛阈值

    void *initMatrixDeviceAddr = nullptr;
    void *hResDeviceAddr = nullptr;

    aclTensor *initMatrixTensor = nullptr;
    aclTensor *hResTensor = nullptr;

    int64_t initMatrixShapeSize = GetShapeSize(initMatrixShape);
    int64_t hResShapeSize = GetShapeSize(hResShape);

    // 构造非负初始混合矩阵（取值0.1~1.0）
    std::vector<float> initMatrixHostData(initMatrixShapeSize, 0.5f);
    std::vector<float> hResHostData(hResShapeSize, 0.0f);

    // Create initMatrix aclTensor.
    ret = CreateAclTensor(initMatrixHostData, initMatrixShape, &initMatrixDeviceAddr, aclDataType::ACL_FLOAT, &initMatrixTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hRes aclTensor.
    ret = CreateAclTensor(hResHostData, hResShape, &hResDeviceAddr, aclDataType::ACL_FLOAT, &hResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcSinkhornGetWorkspaceSize(
        initMatrixTensor, maxIter, epsilon, hResTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcSinkhornGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnMhcSinkhorn(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcSinkhorn failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side.
    auto size = GetShapeSize(hResShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), hResDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    // 验证输出矩阵是否为双随机矩阵（行和/列和接近1）
    LOG_PRINT("hRes matrix first row sum: %f\n", resultData[0] + resultData[1] + resultData[2] + resultData[3]);

    // 6. Release resources.
    aclDestroyTensor(initMatrixTensor);
    aclDestroyTensor(hResTensor);
    aclrtFree(initMatrixDeviceAddr);
    aclrtFree(hResDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
