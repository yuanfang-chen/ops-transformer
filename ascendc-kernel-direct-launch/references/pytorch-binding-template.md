# PyTorch 绑定代码模板

## 目录结构

```
{operator}/ascend910_93/
├── {operator}_entry.h      # <<<>>> 调用函数声明
├── {operator}_entry.cpp    # Kernel 入口 + <<<>>> 调用
├── {operator}_torch.h      # PyTorch 工具函数声明
├── {operator}_torch.cpp    # Host 实现 + PyTorch 绑定
└── op_kernel/              # Kernel 头文件
```

## {operator}_entry.h

```cpp
#ifndef {OPERATOR}_ENTRY_H
#define {OPERATOR}_ENTRY_H
#include "op_kernel/{operator}_tiling.h"

// <<<>>> 调用函数声明
void {operator}_entry(
    int32_t tilingKey, 
    uint32_t blockDim, 
    void* stream, 
    GM_ADDR input1, 
    GM_ADDR input2,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR mc2Context,      // 如需要通信
    {Operator}Info tilingData);

#endif
```

## {operator}_entry.cpp

```cpp
#include "{operator}_entry.h"
#include "op_kernel/{operator}.h"

extern "C" __global__ __aicore__ void {operator}_generic(
    int32_t tilingKey,
    GM_ADDR input1, GM_ADDR input2, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR mc2Context,
    {Operator}Info tilingData);

void {operator}_entry(
    int32_t tilingKey, uint32_t blockDim, void* stream,
    GM_ADDR input1, GM_ADDR input2, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR mc2Context,
    {Operator}Info tilingData)
{
    {operator}_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, input1, input2, output, workspace, mc2Context, tilingData
    );
}
```

## {operator}_torch.h

```cpp
#ifndef {OPERATOR}_TORCH_H
#define {OPERATOR}_TORCH_H
#include <torch/extension.h>
#include "{operator}_entry.h"

namespace ascend_ops {
namespace {Operator} {

// 获取 Tensor 数据指针
template<typename T>
GM_ADDR get_first_tensor_address(at::Tensor& tensor) {
    return (GM_ADDR)tensor.data_ptr();
}

// 计算 TilingData
void calculate_tilingdata(
    {Operator}Info& tilingData,
    int64_t param1, int64_t param2, ...);

// 计算 TilingKey
int32_t calculate_tilingkey(
    int quant_mode, bool is_bfloat16, ...);

}}
#endif
```

## {operator}_torch.cpp

```cpp
#include "{operator}_torch.h"
#include <acl/acl.h>

namespace ascend_ops {
namespace {Operator} {

// ========== 1. Schema 注册 ==========
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("{Operator}(Tensor input1, Tensor input2, Tensor mc2_context, "
          "int param1, int param2, *, Tensor? optional_input=None, "
          "int quant_mode=0) -> (Tensor, Tensor)");
}

// ========== 2. Meta 函数（图模式 InferShape）==========
static std::tuple<at::Tensor, at::Tensor> {operator}_meta(
    const at::Tensor &input1, const at::Tensor &input2,
    const at::Tensor &mc2_context, int64_t param1, int64_t param2,
    const c10::optional<at::Tensor> &optional_input, int64_t quant_mode)
{
    // 推导输出形状
    auto output1_sizes = std::vector<int64_t>{param1 * param2, input1.size(-1)};
    auto output2_sizes = std::vector<int64_t>{param1};
    
    // 创建空 Tensor（Meta 设备）
    auto output1 = at::empty(output1_sizes, input1.options());
    auto output2 = at::empty(output2_sizes, input1.options().dtype(at::kInt));
    
    return std::make_tuple(output1, output2);
}

// ========== 3. NPU 调用函数 ==========
void {operator}_api(
    aclrtStream stream,
    int32_t tilingKey,
    const at::Tensor &input1,
    const at::Tensor &input2,
    const at::Tensor &mc2_context,
    at::Tensor &output1,
    at::Tensor &output2,
    at::Tensor &workspace,
    const {Operator}Info &tilingData)
{
    // 获取 Tensor 数据指针
    auto input1_ptr = get_first_tensor_address(input1);
    auto input2_ptr = get_first_tensor_address(input2);
    auto output1_ptr = get_first_tensor_address(output1);
    auto output2_ptr = get_first_tensor_address(output2);
    auto workspace_ptr = get_first_tensor_address(workspace);
    auto mc2_context_ptr = get_first_tensor_address(mc2_context);
    
    // <<<>>> 调用
    {operator}_entry(
        tilingKey, 
        tilingData.aivNum,  // blockDim
        (void*)stream,
        input1_ptr, input2_ptr,
        output1_ptr, output2_ptr,
        workspace_ptr, mc2_context_ptr,
        tilingData
    );
}

// ========== 4. NPU 实现函数 ==========
static std::tuple<at::Tensor, at::Tensor> npu_{operator}(
    const at::Tensor &input1, const at::Tensor &input2,
    const at::Tensor &mc2_context, int64_t param1, int64_t param2,
    const c10::optional<at::Tensor> &optional_input, int64_t quant_mode)
{
    // 获取 NPU 流
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 输入校验
    TORCH_CHECK(input1.device().type() == c10::DeviceType::PrivateUse1,
        "input1 must be on NPU device");
    
    // 计算 TilingData
    {Operator}Info tilingData;
    calculate_tilingdata(tilingData, param1, param2, ...);
    
    // 计算 TilingKey
    int32_t tilingKey = calculate_tilingkey(quant_mode, input1.dtype() == at::kBFloat16, ...);
    
    // 创建输出 Tensor
    auto output1_sizes = std::vector<int64_t>{param1 * param2, input1.size(-1)};
    auto output2_sizes = std::vector<int64_t>{param1};
    auto output1 = at::empty(output1_sizes, input1.options());
    auto output2 = at::empty(output2_sizes, input1.options().dtype(at::kInt));
    
    // 创建 workspace
    auto workspace = at::empty({tilingData.workspaceSize}, 
        at::TensorOptions().dtype(at::kByte).device(c10::DeviceType::PrivateUse1));
    
    // 调用 API
    {operator}_api(stream, tilingKey, input1, input2, mc2_context,
        output1, output2, workspace, tilingData);
    
    return std::make_tuple(output1, output2);
}

// ========== 5. 注册实现 ==========
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("{Operator}", TORCH_FN(npu_{operator}));
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("{Operator}", &{operator}_meta);
}

}}
```

## CMakeLists.txt

```cmake
add_sources("--npu-arch=dav-2201")
```

## Python 调用示例

```python
import torch
import torch_npu

# 调用算子
output1, output2 = torch.ops.ascend_ops.{Operator}(
    input1=x.npu(),
    input2=expert_ids.npu(),
    mc2_context=context,
    param1=128,
    param2=16,
    quant_mode=0
)
```