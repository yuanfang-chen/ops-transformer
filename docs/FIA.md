# FusedInferAttentionScore 算子 pybind11 → torch.library 改造指南

## 1. 概述

本文档介绍如何将 `fused_infer_attention_score` 算子从 **pybind11** 方式改造为 **torch.library** 方式注册，实现通过 `torch.ops.xxx` 调用算子。

### 1.1 改造背景

当前样例使用 pybind11 注册算子，Python 侧通过 `import ascendc_ops` 调用。改造后使用 torch.library 注册，Python 侧通过 `torch.ops.ascendc_ops.ascendc_fia()` 调用，与 PyTorch 生态更契合。

### 1.2 改造前后对比

| 维度 | pybind11（改造前） | torch.library（改造后） |
|------|-------------------|------------------------|
| 头文件 | `<pybind11/pybind11.h>` | `<torch/library.h>` |
| CMake 命令 | `pybind11_add_module` | `add_library` |
| 注册宏 | `PYBIND11_MODULE` | `TORCH_LIBRARY_FRAGMENT` + `TORCH_LIBRARY_IMPL` |
| Python 调用 | `ascendc_ops.ascendc_fia()` | `torch.ops.ascendc_ops.ascendc_fia()` |
| 图模式支持 | 不支持 | 支持（需 Meta 函数） |

---

## 2. 需要修改的文件

```
examples/pybind/fused_infer_attention_score/
├── CMakeLists.txt                      # 修改
├── fused_infer_attention_score.cpp     # 修改
├── fia_test.py                         # 修改
└── include/                            # 不变
```

---

## 3. CMakeLists.txt 改造

### 3.1 删除 pybind11 相关配置

```diff
- execute_process(
-     COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
-     OUTPUT_STRIP_TRAILING_WHITESPACE
-     OUTPUT_VARIABLE PYBIND11_CMAKE_PREFIX_PATH
- )
- find_package(pybind11 REQUIRED HINTS ${PYBIND11_CMAKE_PREFIX_PATH})
```

### 3.2 修改目标定义

```diff
- pybind11_add_module(ascendc_ops SHARED
-     fused_infer_attention_score.cpp
-     # abc.cpp
- )
+ add_library(ascendc_ops SHARED
+     fused_infer_attention_score.cpp
+ )
```

### 3.3 添加 torch 链接库

```diff
  target_link_libraries(ascendc_ops PRIVATE
+     torch
+     torch_cpu
      torch_npu
      ascendcl
      platform
      register
      tiling_api
      runtime
  )
```

### 3.4 改造后完整 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS
    ${TORCH_NPU_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/block
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include/memcopy
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vf
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api
    ${ASCEND_HOME}/include/ascendc
    ${ASCEND_HOME}/include/ascendc/basic_api/interface
    ${ASCEND_HOME}/include/ascendc/basic_api
    ${ASCEND_HOME}/pkg_inc/op_common
    ${ASCEND_HOME}/pkg_inc/base
    ${ASCEND_HOME}/pkg_inc
)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

set_source_files_properties(
    fused_infer_attention_score.cpp
    PROPERTIES LANGUAGE ASC
)

set_source_files_properties(
    include/op_host/abc.cpp
    PROPERTIES LANGUAGE CXX
)

add_library(xxxlib SHARED
    include/op_host/abc.cpp
)

# torch.library 模式：使用 add_library 替代 pybind11_add_module
add_library(ascendc_ops SHARED
    fused_infer_attention_score.cpp
)

target_link_libraries(ascendc_ops PRIVATE
    xxxlib
)

target_include_directories(ascendc_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${COMMON_INCLUDE_DIRS}
)

target_link_libraries(ascendc_ops PRIVATE
    torch
    torch_cpu
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

target_link_directories(ascendc_ops PRIVATE
    ${TORCH_NPU_LIBRARIES}
    ${ASCEND_HOME}/lib64
)

target_compile_options(ascendc_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    # $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
    "-xasc"
    "-w"
    "-O3"
)

target_compile_options(xxxlib PRIVATE
    ${TORCH_CXX_FLAGS}
    "--save-temps"
    "-w"
    "-O3"
)
```

---

## 4. fused_infer_attention_score.cpp 改造

### 4.1 头文件修改

```diff
- #include <pybind11/pybind11.h>
+ #include <torch/library.h>
  #include <torch/extension.h>
```

### 4.2 新增 Meta 函数

Meta 函数用于图模式下的 shape 推导，不需要实际执行计算：

```cpp
// Meta 函数：仅推导输出 shape 和 dtype，不执行实际计算
at::Tensor ascendc_fia_meta(
    const at::Tensor& queryTensor,
    const at::Tensor& keyTensor,
    const at::Tensor& valueTensor,
    const at::Tensor& keyAntiquantScaleTensor,
    const at::Tensor& valueAntiquantScaleTensor,
    const at::Tensor& queryQuantScaleTensor)
{
    auto query_sizes = queryTensor.sizes().vec();
    std::vector<int64_t> output_sizes = {
        query_sizes[0],   // B
        query_sizes[1],   // N
        query_sizes[2],   // S
        query_sizes[3] * 2  // D * 2
    };
    return at::empty(output_sizes,
        at::dtype(queryTensor.dtype()).device(queryTensor.device()));
}
```

### 4.3 注册算子 Schema

```cpp
// 1. 注册算子 Schema（定义输入输出签名）
TORCH_LIBRARY_FRAGMENT(ascendc_ops, m)
{
    m.def("ascendc_fia("
          "Tensor query, "
          "Tensor key, "
          "Tensor value, "
          "Tensor key_antiquant_scale, "
          "Tensor value_antiquant_scale, "
          "Tensor query_quant_scale"
          ") -> Tensor");
}
```

### 4.4 注册 NPU 实现

```cpp
// 2. 注册 NPU 实现（实际计算逻辑）
TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
{
    m.impl("ascendc_fia", &ascendc_ops::ascendc_fia);
}
```

### 4.5 注册 Meta 实现

```cpp
// 3. 注册 Meta 实现（图模式 shape 推导）
TORCH_LIBRARY_IMPL(ascendc_ops, Meta, m)
{
    m.impl("ascendc_fia", &ascendc_fia_meta);
}
```

### 4.6 删除 pybind11 注册

```diff
- PYBIND11_MODULE(ascendc_ops, m)
- {
-     m.doc() = "ascendc_fia pybind11 interfaces";
-     m.def("ascendc_fia", &ascendc_ops::ascendc_fia, "");
- }
```

### 4.7 改造后完整代码结构

```cpp
/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* ...
*/

#include <torch/library.h>        // 改动：替换 pybind11
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include <iostream>
#include <cstdlib>
#include <memory>
#include <cstdint>

#define FIA_ENABLE_MLA
#include "common_utils.h"
#include "io_utils.h"
#include "flash_attention_score_tiling_regbase.h"
#include "fia_entry.h"
#include "op_host/abc.h"
#include "op_host/fused_infer_attention_score_tiling.h"
#include "op_host/fused_infer_attention_score_tiling_constants.h"

// ========== 核函数（不变） ==========
__global__ __aicore__ void FiaKernelFullQuant(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR keyAntiquantScale,
        GM_ADDR valueAntiquantScale, GM_ADDR dequantScaleQuery, GM_ADDR attentionOut,
        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FlashAttentionEntry(
        query, key, value,
        keyAntiquantScale, valueAntiquantScale, dequantScaleQuery, attentionOut,
        workspace, tiling);
    return;
}

// ========== 命名空间内函数（不变） ==========
namespace ascendc_ops {

optiling::TilingContext InitContext(...) { ... }

static std::vector<int64_t> getTensorShape(const at::Tensor &tensor) { ... }

at::Tensor ascendc_fia(
    const at::Tensor& queryTensor,
    const at::Tensor& keyTensor,
    const at::Tensor& valueTensor,
    const at::Tensor& keyAntiquantScaleTensor,
    const at::Tensor& valueAntiquantScaleTensor,
    const at::Tensor& queryQuantScaleTensor)
{
    // ... 原有实现不变 ...
}

} // namespace ascendc_ops

// ========== 新增：Meta 函数 ==========
at::Tensor ascendc_fia_meta(
    const at::Tensor& queryTensor,
    const at::Tensor& keyTensor,
    const at::Tensor& valueTensor,
    const at::Tensor& keyAntiquantScaleTensor,
    const at::Tensor& valueAntiquantScaleTensor,
    const at::Tensor& queryQuantScaleTensor)
{
    auto query_sizes = queryTensor.sizes().vec();
    std::vector<int64_t> output_sizes = {
        query_sizes[0], query_sizes[1], query_sizes[2], query_sizes[3] * 2
    };
    return at::empty(output_sizes,
        at::dtype(queryTensor.dtype()).device(queryTensor.device()));
}

// ========== 新增：torch.library 注册 ==========
TORCH_LIBRARY_FRAGMENT(ascendc_ops, m)
{
    m.def("ascendc_fia("
          "Tensor query, "
          "Tensor key, "
          "Tensor value, "
          "Tensor key_antiquant_scale, "
          "Tensor value_antiquant_scale, "
          "Tensor query_quant_scale"
          ") -> Tensor");
}

TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
{
    m.impl("ascendc_fia", &ascendc_ops::ascendc_fia);
}

TORCH_LIBRARY_IMPL(ascendc_ops, Meta, m)
{
    m.impl("ascendc_fia", &ascendc_fia_meta);
}
```

---

## 5. fia_test.py 改造

### 5.1 修改 import 方式

```diff
- sys.path.append(os.getcwd())
- import ascendc_ops
+ # 加载自定义算子库
+ torch.ops.load_library("./build/libascendc_ops.so")
```

### 5.2 修改调用方式

```diff
- output = ascendc_ops.ascendc_fia(
-     fia_input.q_tensor, fia_input.k_tensor, fia_input.v_tensor,
-     fia_input.dequant_scale_key, fia_input.dequant_scale_value,
-     fia_input.dequant_scale_query)
+ output = torch.ops.ascendc_ops.ascendc_fia(
+     fia_input.q_tensor, fia_input.k_tensor, fia_input.v_tensor,
+     fia_input.dequant_scale_key, fia_input.dequant_scale_value,
+     fia_input.dequant_scale_query)
```

### 5.3 改造后完整 Python 代码

```python
#!/usr/bin/python3
# coding=utf-8

import sys
import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import numpy as np
from typing import NamedTuple

ERROR_TOL = 5e-3
DATA_TYPE = np.float32

# 加载自定义算子库（关键改动）
torch.ops.load_library("./build/libascendc_ops.so")

class FiaInput(NamedTuple):
    q_tensor: torch.Tensor
    k_tensor: torch.Tensor
    v_tensor: torch.Tensor
    dequant_scale_key: torch.Tensor
    dequant_scale_value: torch.Tensor
    dequant_scale_query: torch.Tensor
    num_query_heads: int
    softmax_scale: float
    input_layout: str
    num_key_value_heads: int
    query_quant_mode: int
    key_quant_mode: int
    value_quant_mode: int
    inner_precise: int
    return_softmax_lse: int
    query_dtype: int
    key_dtype: int
    value_dtype: int


def gen_golden_data_simple(b, n1, n2, s1, s2, d) -> FiaInput:
    input_layout = 'BNSD'
    scale_value = 0.088388

    q_tensor = torch.randint(-127, 127, (b, n1, s1, d), dtype=torch.int8)
    k_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8)
    v_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8)
    key_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32)
    value_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32)
    dequant_scale_query = torch.rand((1, 1, 64, 1), dtype=torch.float32)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    q_tensor.npu()
    k_tensor.npu()
    v_tensor.npu()
    key_antiquant_scale = key_antiquant_scale.npu()
    value_antiquant_scale = value_antiquant_scale.npu()
    dequant_scale_query = dequant_scale_query.npu()

    npu_out = torch.ops.npu.npu_fused_infer_attention_score_v2(
        q_tensor.npu(), k_tensor.npu(), v_tensor.npu(),
        dequant_scale_key=key_antiquant_scale,
        dequant_scale_value=value_antiquant_scale,
        dequant_scale_query=dequant_scale_query,
        num_query_heads=n1,
        softmax_scale=scale_value,
        input_layout=input_layout,
        num_key_value_heads=n2, query_quant_mode=7,
        key_quant_mode=7, value_quant_mode=7,
        inner_precise=0, return_softmax_lse=0,
        query_dtype=torch_npu.float8_e4m3fn,
        key_dtype=torch_npu.float8_e4m3fn,
        value_dtype=torch_npu.float8_e4m3fn)

    fia_input = FiaInput(
        q_tensor, k_tensor, v_tensor,
        key_antiquant_scale, value_antiquant_scale, dequant_scale_query,
        n1, scale_value, input_layout, n2, 7, 7, 7, 0, 0,
        torch_npu.float8_e4m3fn, torch_npu.float8_e4m3fn, torch_npu.float8_e4m3fn)

    return fia_input, npu_out[0].cpu()


def verify_result(golden, output):
    golden = golden.view(torch.uint16).to(torch.bfloat16).flatten().cpu()
    output = output.view(torch.uint16).to(torch.bfloat16).flatten().cpu()
    print("output:")
    print(output)
    print("golden:")
    print(golden)

    output = output.float()
    golden = golden.float()

    output_nan = np.isnan(output)
    golden_nan = np.isnan(golden)
    both_nan = output_nan & golden_nan
    nan_mismatch = output_nan ^ golden_nan

    diff = np.abs(output - golden)
    diff_mask = diff > 1
    error_mask = (diff_mask | nan_mismatch) & (~both_nan)
    diff_indices = np.where(error_mask)[0]

    error_ratio = diff_indices.size / golden.numpy().size
    print("error count:", diff_indices.size)
    print("total count:", golden.numpy().size)

    return error_ratio <= ERROR_TOL


class TestFia(TestCase):
    def test_fia(self):
        fia_input, golden = gen_golden_data_simple(1, 1, 1, 8192, 8192, 128)

        # 关键改动：使用 torch.ops 调用
        output = torch.ops.ascendc_ops.ascendc_fia(
            fia_input.q_tensor, fia_input.k_tensor, fia_input.v_tensor,
            fia_input.dequant_scale_key,
            fia_input.dequant_scale_value,
            fia_input.dequant_scale_query)

        try:
            res = verify_result(golden, output)
            if not res:
                raise ValueError("[ERROR] result error")
            print("test pass")
        except Exception as e:
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    run_tests()
```

---

## 6. 改动总结

| 文件 | 改动项 | 说明 |
|------|--------|------|
| `CMakeLists.txt` | 删除 pybind11 依赖 | 移除 `find_package(pybind11)` |
| `CMakeLists.txt` | `pybind11_add_module` → `add_library` | 生成 .so 而非 .pyd |
| `CMakeLists.txt` | 添加 `torch` `torch_cpu` 链接库 | torch.library 依赖 |
| `fused_infer_attention_score.cpp` | 头文件替换 | `pybind11.h` → `torch/library.h` |
| `fused_infer_attention_score.cpp` | 新增 Meta 函数 | 图模式 shape 推导 |
| `fused_infer_attention_score.cpp` | 新增 Schema 注册 | `TORCH_LIBRARY_FRAGMENT` |
| `fused_infer_attention_score.cpp` | 新增实现注册 | `TORCH_LIBRARY_IMPL` (PrivateUse1 + Meta) |
| `fused_infer_attention_score.cpp` | 删除 pybind11 注册 | 移除 `PYBIND11_MODULE` |
| `fia_test.py` | 修改 import | `import ascendc_ops` → `torch.ops.load_library()` |
| `fia_test.py` | 修改调用方式 | `ascendc_ops.xxx` → `torch.ops.ascendc_ops.xxx` |

---

## 7. 编译运行

```bash
cd examples/pybind/fused_infer_attention_score
mkdir -p build && cd build
cmake ..
make -j
python3 ../fia_test.py
```

---

## 8. 常见问题

### 8.1 库名问题

- pybind11 生成：`ascendc_ops.cpython-311-x86_64-linux-gnu.so`
- torch.library 生成：`libascendc_ops.so`

Python 侧加载时注意路径和库名：
```python
torch.ops.load_library("./build/libascendc_ops.so")
```

### 8.2 Namespace 冲突

`TORCH_LIBRARY_FRAGMENT` 的 namespace 必须与 Python 调用时的 namespace 一致：
```python
torch.ops.ascendc_ops.ascendc_fia(...)
#              ^^^^^^^^^^^^ 对应 TORCH_LIBRARY_FRAGMENT(ascendc_ops, m)
```

### 8.3 Schema 签名

Schema 中的参数名必须与 C++ 函数签名顺序一致，类型使用 Tensor 类型别名：
```
Tensor query      → const at::Tensor&
Tensor key        → const at::Tensor&
...
```

---

## 参考链接

- [torch.library 官方文档](https://pytorch.org/docs/stable/library.html)
- [ops-transformer torch.library 参考](https://gitcode.com/cann/ops-transformer/pull/2141)
