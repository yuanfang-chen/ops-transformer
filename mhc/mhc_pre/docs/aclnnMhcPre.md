# aclnnMhcPre

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term> |      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Atten或MLP层的输入矩阵$h^{in}$。

- 计算公式
$$
\begin{aligned}
\vec{x^{'}_{l}} &=RMSNorm(\vec{x_{l}})\\
H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
H^{pre}_l &= \sigma (H^{pre}_{l})\\
H^{post}_l &= 2\sigma (H^{post}_{l})\\
h_{in} &=\vec{x^{'}_{l}}H^{pre}_l
\end{aligned}
$$

---

## 函数原型

算子采用两段式接口调用：需先调用`aclnnMhcPreGetWorkspaceSize`获取计算所需的Device侧内存大小，再调用`aclnnMhcPre`执行实际计算。

```c++
aclnnStatus aclnnMhcPreGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gammaOptional, float normEps, float hcEps,
    aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
    aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```
```c++
aclnnStatus aclnnMhcPre(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## aclnnMhcPreGetWorkspaceSize

### 参数说明
| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| x | 输入 | 待计算数据，表示网络中mHC层的输入数据 | 必选参数，不能为空Tensor | BFLOAT16 或 FLOAT16 | ND | ($B,S,n,D$) 或 ($T,n,D$) | √ |
| phi | 输入 | mHC的参数矩阵 | 必选参数，不能为空Tensor | FLOAT32 | ND | ($n^2+2n, nD$) | √ |
| alpha | 输入 | mHC的缩放参数 | 必选参数，不能为空Tensor | FLOAT32 | - | (3) | - |
| bias | 输入 | mHC的bias参数 | 必选参数，不能为空Tensor | FLOAT32 | - | ($n^2+2n$) | - |
| gammaOptional | 可选输入 | 表示进行RmsNorm计算的缩放因子 | 可选参数 | FLOAT32 | ND | ($n, D$) | √ |
| normEps | 可选输入 | RmsNorm的防除零参数，建议值：1e-6 | 可选参数 | FLOAT32 | - | - | - |
| hcEps | 可选输入 | $H_{pre}$的sigmoid后的eps参数，建议值：1e-6 | 可选参数 | FLOAT32 | - | - | - |
| hIn | 输出 | 输出的h_in作为Atten/MLP层的输入 | 必选参数 | BFLOAT16 或 FLOAT16  | ND | ($B,S,D$) 或 ($T,D$)  | - |
| hPost | 输出 | 输出的mHC的h_post变换矩阵 | 必选参数 | FLOAT32 | ND | ($B,S,D$) 或 ($T,D$)  | - |
| hRes | 输出 | 输出的mHC的h_res变换矩阵（未做sinkhorn变换） | 必选参数 | FLOAT32 | ND | ($B,S,n,n$) 或 ($T,n,n$) | - |
| invRmsOptional | 可选输出 | RmsRorm计算得到的1/r | 可选参数 | FLOAT32 | ND | ($B,S$) 或 ($T$) | - |
| hMixOptional | 可选输出 | x与phi矩阵乘的结果 | 可选参数 | FLOAT32 | ND | ($B,S,n^2+2n$) 或 ($T,n^2+2n$) | - |
| hPreOptional | 可选输出 | 做完sigmoid计算之后的h_pre矩阵 | 可选参数 | FLOAT32 | ND | ($B,S,n$) 或 ($T,n$) | - |
| workspaceSize | 输出 | 计算所需的Device侧workspace内存大小（字节） | 由算子内部计算得出，用于后续申请内存 | UINT64 | - | - | - |
| executor | 输出 | 算子执行器，包含计算流程和参数信息 | 需传递给第二段接口使用 | aclOpExecutor | - | - | - |

### 返回值

返回`aclnnStatus`状态码，第一段接口主要完成入参校验，异常场景如下：

| 返回值 | 错误码 | 描述 |
|:--- |:--- |:--- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选参数或者输出是空指针。|
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入变量的数据类型和数据格式不在支持的范围内。 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API内存调用npu runtime的接口异常。 |

## aclnnMhcPre

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|:--- |:--- |:--- |
| workspace | 输入 | Device侧申请的workspace内存地址，需与第一段接口返回的workspaceSize匹配。 |
| workspaceSize | 输入 | Device侧workspace内存大小，由`aclnnMhcPreGetWorkspaceSize`接口返回。 |
| executor | 输入 | 算子执行器，由第一段接口创建，包含计算流程和参数信息。 |
| stream | 输入 | 指定执行计算任务的Stream，需提前创建并绑定Device。 |

### 返回值

返回`aclnnStatus`状态码。

## 约束说明

### 确定性计算

- aclnnMhcPre 默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束
1. 输入约束：
   - 输入Tensor `x`、`phi`、`alpha`、`bias` 不能为空，且必须为Device侧Tensor；
   - 所有输入/输出Tensor的数据格式仅支持`ACL_FORMAT_ND`；
2. 内存约束：
   - Workspace内存需在Device侧申请，且大小需严格匹配第一段接口返回值；
   - 非连续Tensor无需提前转为连续，算子内部自动处理。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~65536 | B*S 或T支持512~65536范围（训练及推理Prefill），支持1~512（推理Decode）。|
| n | 4、6、8 | n目前支持4, 6, 8。|
| D | 512~16384 | D支持512~16384范围以内，需满足D为32对齐。|

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre.h"

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

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float类型）
void PrintTensorDataFloat(const std::vector<int64_t>& shape, void* device_addr) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);
  
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(float),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); 
            return);

  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float16类型）
void PrintTensorDataFloat16(const std::vector<int64_t>& shape, void* device_addr) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);
  
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(aclFloat16),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); 
            return);

  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
  aclError ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclInit failed, error: %d\n", ret); 
            return -1);

  ret = aclrtSetDevice(device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); 
            return -1);

  ret = aclrtCreateContext(&context, device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); 
            return -1);

  ret = aclrtSetCurrentContext(context);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); 
            return -1);

  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); 
            return -1);

  return 0;
}

// 创建FLOAT32类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat32(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape) * sizeof(float);

  aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  ret = aclrtMemcpy(
      device_addr, size,
      host_data.data(), size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); 
            return -1);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建FLOAT16类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat16(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape);
  std::vector<aclFloat16> host_data_fp16(size);
  for (int64_t i = 0; i < size; ++i) {
    host_data_fp16[i] = aclFloat16(host_data[i]);
  }

  int64_t byte_size = size * sizeof(aclFloat16);

  aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  ret = aclrtMemcpy(
      device_addr, byte_size,
      host_data_fp16.data(), byte_size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); 
            return -1);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT16, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建FLOAT16类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat16Output(
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape);
  int64_t byte_size = size * sizeof(aclFloat16);

  aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT16, nullptr, 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建FLOAT32类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat32Output(
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape);
  int64_t byte_size = size * sizeof(float);

  aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, nullptr, 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

int main() {
  int32_t device_id = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  
  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0, 
            LOG_PRINT("InitAcl failed, error: %d\n", ret); 
            return -1);

  int B = 1;
  int S = 2048;
  int n = 4;
  int D = 2560;

  std::vector<int64_t> x_shape = {B * S, n, D};
  int64_t x_size = GetShapeSize(x_shape);
  std::vector<float> x_host_data(x_size, 1.0f);

  std::vector<int64_t> phi_shape = {n * n + 2 * n, n * D};
  std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);

  std::vector<int64_t> alpha_shape = {3};
  std::vector<float> alpha_host_data(3, 1.0f);

  std::vector<int64_t> bias_shape = {n * n + 2 * n};
  std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);

  std::vector<int64_t> gamma_shape = {n, D};
  std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);

  std::vector<int64_t> output_hin_shape = {B * S, D};
  std::vector<int64_t> output_h_post_shape = {B * S, n};
  std::vector<int64_t> output_h_res_shape = {B * S, n, n};
  std::vector<int64_t> output_inv_rms_shape = {B * S};
  std::vector<int64_t> output_h_mix_shape = {B * S, n * n + 2 * n};
  std::vector<int64_t> output_h_pre_shape = {B * S, n};

  void* x_device_addr = nullptr;
  aclTensor* x_tensor = nullptr;
  ret = CreateAclTensorFloat16(x_host_data, x_shape, x_device_addr, x_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);

  void* phi_device_addr = nullptr;
  aclTensor* phi_tensor = nullptr;
  ret = CreateAclTensorFloat32(phi_host_data, phi_shape, phi_device_addr, phi_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);

  void* alpha_device_addr = nullptr;
  aclTensor* alpha_tensor = nullptr;
  ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, alpha_device_addr, alpha_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);

  void* bias_device_addr = nullptr;
  aclTensor* bias_tensor = nullptr;
  ret = CreateAclTensorFloat32(bias_host_data, bias_shape, bias_device_addr, bias_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);

  void* gamma_device_addr = nullptr;
  aclTensor* gamma_tensor = nullptr;
  ret = CreateAclTensorFloat32(gamma_host_data, gamma_shape, gamma_device_addr, gamma_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create gamma_tensor failed\n"); return -1);

  void* output_hin_device_addr = nullptr;
  aclTensor* output_hin_tensor = nullptr;
  ret = CreateAclTensorFloat16Output(output_hin_shape, output_hin_device_addr, output_hin_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_hin_tensor failed\n"); return -1);

  void* output_h_post_device_addr = nullptr;
  aclTensor* output_h_post_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_post_shape, output_h_post_device_addr, output_h_post_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_post_tensor failed\n"); return -1);

  void* output_h_res_device_addr = nullptr;
  aclTensor* output_h_res_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_res_shape, output_h_res_device_addr, output_h_res_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_res_tensor failed\n"); return -1);

  void* output_inv_rms_device_addr = nullptr;
  aclTensor* output_inv_rms_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_inv_rms_shape, output_inv_rms_device_addr, output_inv_rms_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_inv_rms_tensor failed\n"); return -1);

  void* output_h_mix_device_addr = nullptr;
  aclTensor* output_h_mix_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_mix_shape, output_h_mix_device_addr, output_h_mix_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_mix_tensor failed\n"); return -1);

  void* output_h_pre_device_addr = nullptr;
  aclTensor* output_h_pre_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_pre_shape, output_h_pre_device_addr, output_h_pre_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_pre_tensor failed\n"); return -1);

  double normEps = 1e-6;
  double hcEps = 1e-6;

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus aclnn_ret = aclnnMhcPreGetWorkspaceSize(
    x_tensor, phi_tensor, alpha_tensor, bias_tensor, gamma_tensor,
    normEps, hcEps,
    output_hin_tensor, output_h_post_tensor, output_h_res_tensor,
    output_inv_rms_tensor, output_h_mix_tensor, output_h_pre_tensor,
    &workspace_size, &executor);

  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcPreGetWorkspaceSize failed, error: %d\n", aclnn_ret); 
            return -1);
  
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); 
              return -1);
  }

  aclnn_ret = aclnnMhcPre(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcPre failed, error: %d\n", aclnn_ret); 
            return -1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("MhcPre compute success!\n");
  LOG_PRINT("Output tensor data: \n");

  PrintTensorDataFloat16(output_hin_shape, output_hin_device_addr);
  PrintTensorDataFloat(output_h_post_shape, output_h_post_device_addr);
  PrintTensorDataFloat(output_h_res_shape, output_h_res_device_addr);
  PrintTensorDataFloat(output_inv_rms_shape, output_inv_rms_device_addr);
  PrintTensorDataFloat(output_h_mix_shape, output_h_mix_device_addr);
  PrintTensorDataFloat(output_h_pre_shape, output_h_pre_device_addr);

  aclDestroyTensor(x_tensor);
  aclDestroyTensor(phi_tensor);
  aclDestroyTensor(alpha_tensor);
  aclDestroyTensor(bias_tensor);
  aclDestroyTensor(gamma_tensor);

  aclDestroyTensor(output_hin_tensor);
  aclDestroyTensor(output_h_post_tensor);
  aclDestroyTensor(output_h_res_tensor);
  aclDestroyTensor(output_inv_rms_tensor);
  aclDestroyTensor(output_h_mix_tensor);
  aclDestroyTensor(output_h_pre_tensor);

  aclrtFree(x_device_addr);
  aclrtFree(phi_device_addr);
  aclrtFree(alpha_device_addr);
  aclrtFree(bias_device_addr);
  aclrtFree(gamma_device_addr);

  aclrtFree(output_hin_device_addr);
  aclrtFree(output_h_post_device_addr);
  aclrtFree(output_h_res_device_addr);
  aclrtFree(output_inv_rms_device_addr);
  aclrtFree(output_h_mix_device_addr);
  aclrtFree(output_h_pre_device_addr);

  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }

  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(device_id);
  aclFinalize();

  LOG_PRINT("All resources released successfully!\n");
  return 0;
}
```