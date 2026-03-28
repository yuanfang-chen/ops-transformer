## 1. 概述

本文档介绍将 `fused_infer_attention_score` 算子从传统的 **Host/Kernel 传递模式** 改造为 **torch.library 直调模式** 的方法，实现通过 `<<<>>>` 语法直接启动核函数，使算子能够通过 PyTorch 的 `torch.library` 机制被直接调用。

### 1.1 改造背景

原始算子位于 `attention/fused_infer_attention_score` 目录下，采用传统的 Host/Kernel 传递模式：

- **op_host**: 包含 Tiling 校验、Shape 推导等 Host 侧代码（如 `fused_infer_attention_score_tiling.h`）
- **op_kernel**: 包含核函数实现、算子计算逻辑等 Kernel 侧代码
- **op_api**: 包含 aclnn 二段式接口封装

改造后的算子采用 `torch.library` 直调模式：
- 使用 `TORCH_LIBRARY_FRAGMENT` 注册算子 Schema
- 使用 `TORCH_LIBRARY_IMPL` 注册实现
- 通过 `<<<>>>` 语法直接启动核函数

### 1.2 改造目标

- 在目标目录中新增 `<<<>>>` 直调方法
- 提供从 PyTorch 接口到 Kernel 的一整套流程
- 提供 Python 测试脚本以供验证

### 1.3 改造前后对比

| 维度 | 传递模式（原始） | 直调模式（改造） |
|------|-----------------|-------------------|
| 代码组织 | `op_host` + `op_kernel` 分离 | 单一目录统一编译 |
| Host 入口 | `gert::TilingContext*` GE 上下文 | 自定义 `optiling::TilingContext` |
| Kernel 启动 | 二段式调度 | `<<<blockDim, nullptr, stream>>>` |
| Python 绑定 | aclnn API | `torch.library` 绑定 |
| TilingKey | 全量支持 | 按需裁剪 |

---

## 2. 目录结构
### 2.1 原始分离模式目录

```
attention/fused_infer_attention_score/
├── CMakeLists.txt
├── README.md
├── op_host/                                        # Host 侧代码
│   ├── CMakeLists.txt
│   ├── fused_infer_attention_score_def.cpp         # 算子原型定义
│   ├── fused_infer_attention_score_infershape.cpp  # Shape推导
│   ├── fused_infer_attention_score_tiling.cpp      # Tiling实现
│   ├── fused_infer_attention_score_tiling.h        # TilingData定义
│   ├── ... ...
│   ├── checkers/                                   # 校验器
│   └── op_api/                                     # aclnn接口
├── op_kernel/                                      # Kernel 侧代码
│   ├── CMakeLists.txt
│   ├── fused_infer_attention_score.cpp             # 核函数实现
│   └── include/
│       ├── block/                                  # Block相关
│       ├── kernel/                                 # Kernel相关
│       ├── memcopy/                                # 内存拷贝相关
│       ├── utils/                                  # 工具函数
│       └── vf/                                     # VF相关
└── examples/                                       # 调用示例
    └── test_aclnn_fused_infer_attention_score.cpp
```

### 2.2 改造后目录结构

```
<root_dir>/fused_infer_attention_score/
├── CMakeLists.txt                                    # 编译配置
├── fused_infer_attention_score.cpp                   # 主C++文件（Kernel入口 + PyTorch绑定）
├── fused_infer_attention_score.py                    # Python测试脚本
├── test.sh                                           # 测试脚本
├── include/
│   ├── block/                                        # Block相关
│   ├── kernel/                                       # Kernel相关
│   │   ├── fia_entry.h                               # FIA Kernel入口
│   │   ├── flash_attention_score_kernel_base.h       # Kernel Base实现
│   │   └── flash_attention_score_kernel_infer.h      # Kernel Infer实现
│   ├── memcopy/                                      # 内存拷贝相关
│   ├── utils/                                        # 工具函数
│   └── vf/                                           # VF相关
└── op_host/                                          # Host侧代码
    ├── fallback_fused_infer_attention_score.cpp      # Fallback实现
    ├── flash_attention_infer_tiling.h                # Flash Attention Tiling
    ├── fused_infer_attention_score_def.cpp           # 算子定义
    ├── fused_infer_attention_score_infershape.cpp    # Shape推导
    ├── fused_infer_attention_score_tiling.cpp        # Tiling实现
    ├── fused_infer_attention_score_tiling.h          # TilingData定义
    ├── ... ...
    ├── arch35/                                         # arch35架构相关
    └── checkers/                                       # 校验器
```

---

## 3. CMakeLists.txt 配置详解

参考样例：[torch_library](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/02_features/02_framework/00_pytorch/torch_library)

### 3.1 编译配置要点

```cmake
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

# 查找torch_npu
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)

# 配置include路径
set(TORCH_NPU_INCLUDE_DIRS
    ${TORCH_NPU_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
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

# 指定源文件语言为Ascend C
set_source_files_properties(
    fused_infer_attention_score.cpp
    PROPERTIES LANGUAGE ASC
)

# 添加Ascend C算子库
add_library(ascendc_ops SHARED
    fused_infer_attention_score.cpp
)

# 配置include目录
target_include_directories(ascendc_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# 链接库依赖
target_link_libraries(ascendc_ops PRIVATE
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    Python3::Python
)

# 配置链接目录
target_link_directories(ascendc_ops PRIVATE
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_HOME}/lib64
)

# ======================================================================================
# NPU 编译选项配置
#
# 说明：
#   - 需根据实际部署的 NPU 硬件架构选择对应的 `npu-arch` 参数。
# ======================================================================================
target_compile_options(ascendc_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>  # ascend910b/ascend910_93
)
```

**关键配置说明**：

| 配置项 | 说明 |
|--------|------|
| `find_package(ASC REQUIRED)` | 查找Ascend C编译器 |
| `find_package(Torch REQUIRED)` | 查找PyTorch |
| `TORCH_NPU_PATH` | torch_npu安装路径 |
| `--npu-arch=dav-2201` | Ascend 910B/910_93架构 |
| `torch_npu` | NPU扩展库 |
| `Python3::Python` | Python库依赖 |

**支持的产品**：
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

---

## 4. PyTorch 绑定实现

### 4.1 Schema 注册

```cpp
TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("ascendc_fia(Tensor query, Tensor key, Tensor value, "
          "Tensor keyAntiquantScale, Tensor valueAntiquantScale, "
          "Tensor queryAntiquantScale) -> Tensor");
}
```

### 4.2 实现注册

```cpp
TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
{
    m.impl("ascendc_fia", TORCH_FN(ascendc_ops::ascendc_fia));
}
```

### 4.3 NPU 调用函数

```cpp
at::Tensor ascendc_fia(const at::Tensor& queryTensor, const at::Tensor& keyTensor,
    const at::Tensor& valueTensor, const at::Tensor& keyAntiquantScaleTensor,
    const at::Tensor& valueAntiquantScaleTensor, const at::Tensor& queryQuantScaleTensor)
{
    // 1. 获取输入Tensor的shape信息
    std::vector<int64_t> shapeQueryTensor = getTensorShape(queryTensor);

    // 2. 创建输出Tensor
    at::Tensor outputTensor = at::empty({shapeQueryTensor[0], shapeQueryTensor[1],
        shapeQueryTensor[2], shapeQueryTensor[3] * 2},
        at::dtype(queryTensor.dtype()).device(queryTensor.device()).layout(queryTensor.layout()));

    // 3. 初始化TilingContext并执行Tiling
    optiling::TilingContext context = InitContext(shapeQueryTensor, shapeKeyTensor, shapeValueTensor);
    optiling::DoOpTilingFusedInferAttentionScore(&context);

    // 4. 创建ACL运行时和流
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 5. 分配设备内存
    GM_ADDR workspace = nullptr;
    GM_ADDR tilingDataDevice = nullptr;
    aclrtMalloc((void**)&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDataDevice, tilingDataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 启动Kernel
    FiaKernelFullQuant<<<blockDimToBeSet, nullptr, stream>>>(
        (uint8_t*)(queryTensor.mutable_data_ptr()),
        (uint8_t*)(keyTensor.mutable_data_ptr()),
        (uint8_t*)(valueTensor.mutable_data_ptr()),
        (uint8_t*)(keyAntiquantScaleTensor.mutable_data_ptr()),
        (uint8_t*)(valueAntiquantScaleTensor.mutable_data_ptr()),
        (uint8_t*)(queryQuantScaleTensor.mutable_data_ptr()),
        (uint8_t*)(outputTensor.mutable_data_ptr()),
        workspace,
        tilingDataDevice
    );

    // 8. 同步流并清理
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);

    return outputTensor;
}
```

---

## 5. Tiling 侧改造详解
### 5.1 去掉 GE Context 依赖

**传递模式**：使用 `gert::TilingContext*` 获取输入信息

```cpp
// 原始代码 (op_host/fused_infer_attention_score_tiling.cpp)
ge::graphStatus FusedInferAttentionScoreTilingFunc(gert::TilingContext *context)
{
    // 通过 GE Context 获取输入 Tensor
    const gert::Tensor *queryTensor = context->GetInputTensor(0);
    const gert::Tensor *keyTensor = context->GetInputTensor(1);

    // 获取输入 Shape
    const gert::Shape *queryShape = context->GetInputShape(0);
    int64_t batchSize = queryShape->GetDim(0);
    int64_t seqLen = queryShape->GetDim(1);

    // 获取 TilingData
    FusedInferAttentionScoreTilingData *tilingData = context->GetTilingData<FusedInferAttentionScoreTilingData>();

    // 设置输出 Shape
    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDim(0, batchSize);

    // 设置 Workspace
    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = workspaceSizeCalc;
}
```

**直调模式**：使用自定义 `optiling::TilingContext` 替代 GE Context

```cpp
// 改造后代码 (op_host/abc.cpp)
optiling::TilingContext InitContext(std::vector<int64_t> shapeQueryTensor,
    std::vector<int64_t> shapeKeyTensor, std::vector<int64_t> shapeValueTensor)
{
    optiling::TilingContext context;

    // 直接从 at::Tensor 获取 Shape 信息
    context.SetInputDesc(optiling::QUERY_INDEX, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::QUERY_INDEX, shapeQueryTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::KEY_INDEX, shapeKeyTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::VALUE_INDEX, shapeValueTensor, optiling::DT_FLOAT16);

    // 设置属性
    auto& attrs = context.GetAttrs();
    attrs.SetAttr(ATTR_N_INDEX, (uint32_t)1);
    attrs.SetAttr(ATTR_SCALE_INDEX, 3.14f);
    attrs.SetAttr(ATTR_INPUT_LAYOUT_INDEX, "string");

    return context;
}

// 调用 Tiling 函数
optiling::DoOpTilingFusedInferAttentionScore(&context);
```

### 5.2 直接从 at::Tensor 解析shape信息

**传递模式**：从 GE Context 获取 TilingData

```cpp
// 原始代码 (op_host/fused_infer_attention_score_tiling.cpp)
ge::graphStatus FusedInferAttentionScoreTilingFunc(gert::TilingContext *context)
{
    FusedInferAttentionScoreTilingData *tilingData = context->GetTilingData<FusedInferAttentionScoreTilingData>();
    tilingData->batchSize = static_cast<uint32_t>(context->GetInputShape(0)->GetDim(0));
    tilingData->seqSize = static_cast<uint32_t>(context->GetInputShape(0)->GetDim(1));
    tilingData->headSize = static_cast<uint32_t>(context->GetInputShape(0)->GetDim(2));
    tilingData->headDim = static_cast<uint32_t>(context->GetInputShape(0)->GetDim(3));
    // ...
}
```

**直调模式**：直接从 `at::Tensor` 获取参数
```cpp
static std::vector<int64_t> getTensorShape(const at::Tensor &tensor)
{
    std::vector<int64_t> shape;
    try {
        shape = std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
    } catch (const std::exception &e) {
        std::cerr << "Error getting tensor shpae: " << e.what() << std::endl;
        shape = {0};
    }
    return shape;
}
```

### 5.3 初始化自定义context
```cpp
optiling::TilingContext InitContext(std::vector<int64_t> shapeQueryTensor, std::vector<int64_t> shapeKeyTensor, std::vector<int64_t> shapeValueTensor) {
    optiling::TilingContext context;
    context.SetInputDesc(optiling::QUERY_INDEX, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::QUERY_INDEX, shapeQueryTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::KEY_INDEX, shapeKeyTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::VALUE_INDEX, shapeValueTensor, optiling::DT_FLOAT16);
    context.SetInputShapeFromVector(optiling::ATTENTION_OUT_INDEX, shapeQueryTensor, optiling::DT_FLOAT16);
    auto& attrs = context.GetAttrs();
    attrs.SetAttr(ATTR_SCALE_INDEX, 3.14f);
    return context;
}
```

### 5.4 将shape信息传递给 TilingData

```cpp
void SetTilingData(FusedInferAttentionScoreTilingData &tilingData, optiling::TilingContext &context)
{
    // 从 at::Tensor 获取的 Shape 信息
    tilingData.batchSize = static_cast<uint32_t>(context.GetInputShape(0).GetDim(0));
    tilingData.seqSize = static_cast<uint32_t>(context.GetInputShape(0).GetDim(1));
    tilingData.headSize = static_cast<uint32_t>(context.GetInputShape(0).GetDim(2));
    tilingData.headDim = static_cast<uint32_t>(context.GetInputShape(0).GetDim(3));
}
```

---

## 6. Python 调用示例

### 6.1 加载库

```python
import torch
import torch_npu
torch.ops.load_library("libascendc_ops.so")
```

### 6.2 调用算子

```python
# 准备输入Tensor
q_tensor = torch.randint(-127, 127, (b, n1, s1, d), dtype=torch.int8).npu()
k_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8).npu()
v_tensor = torch.randint(-127, 127, (b, n2, s2, d), dtype=torch.int8).npu()
key_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32).npu()
value_antiquant_scale = torch.rand((1, 1, 32, 1), dtype=torch.float32).npu()
dequant_scale_query = torch.rand((1, 1, 64, 1), dtype=torch.float32).npu()

# 调用算子
output = torch.ops.ascendc_ops.ascendc_fia(
    q_tensor, k_tensor, v_tensor,
    key_antiquant_scale, value_antiquant_scale, dequant_scale_query
)
```

---

## 7. 构建与安装

### 7.1 编译命令

```bash
# 进入算子目录
cd examples/torch_lib/fused_infer_attention_score

# 创建构建目录
mkdir -p build && cd build

# 配置CMake
cmake ..

# 编译
make -j
```

### 7.2 运行测试

```bash
# 返回算子目录
cd ..

# 运行Python测试
python fia_test.py
```

---

## 8. 改造清单

### 8.1 必须改造项

| 序号 | 改造项 | 说明 |
|-----|--------|------|
| 1 | 创建 `fused_infer_attention_score.cpp` | 主C++文件，包含Kernel入口和PyTorch绑定 |
| 2 | 创建 `CMakeLists.txt` | 编译配置 |
| 3 | 实现 `TORCH_LIBRARY` 注册 | Schema注册 |
| 4 | 实现 `TORCH_LIBRARY_IMPL` 注册 | 实现注册 |
| 5 | 实现 TilingContext 初始化 | 替换 GE Context 依赖 |
| 6 | 实现 `<<<>>>` 直调 | Kernel启动 |

### 8.2 可选改造项

| 序号 | 改造项 | 说明 |
|-----|--------|------|
| 1 | 移除不必要的TilingKey分支 | 按需裁剪 |
| 2 | 简化Kernel模板 | 仅保留必要数据类型 |
| 3 | 优化内存分配 | 复用workspace |

---

## 9. 注意事项

### 9.1 TilingKey 选择

改造后的 TilingKey 与原始模式可能不同，需要根据实际场景重新设计：
- 原始模式：使用模板化的 TilingKey 编码
- 直调模式：使用简化的 TilingKey 编码

### 9.2 数据类型映射

| PyTorch 类型 | Ascend C 类型 |
|-------------|--------------|
| `torch.int8` | `int8_t` |
| `torch.float16` | `float16_t` |
| `torch.bfloat16` | `bfloat16_t` |
| `torch.float32` | `float32_t` |

### 9.3 NPU 架构配置

不同 NPU 架构需要设置不同的编译参数：
- `ascend910b`: `--npu-arch=dav-2201`
- `ascend910_93`: `--npu-arch=dav-2201`
- `ascend950`: `--npu-arch=dav-3101`

### 9.4 内存管理

- workspace 由 PTA 接口内构造，使用 Tensor 作为输入传入 kernel
- 使用 `aclrtMalloc` 分配设备内存
- 使用 `aclrtMemcpyAsync` 进行异步内存拷贝
- 使用 `unique_ptr` 自动管理内存释放

---

## 11. 总结

通过 `torch.library` 直调改造，`fused_infer_attention_score` 算子实现了以下目标：

### 11.1 PTA 接口侧修改

- **对外接口**

```python
import torch
torch.ops.ascendc_ops.ascendc_fia
```

- **PTA 侧**

```cpp
at::Tensor ascendc_fia(const at::Tensor& queryTensor, const at::Tensor& keyTensor,
    const at::Tensor& valueTensor, const at::Tensor& keyAntiquantScaleTensor,
    const at::Tensor& valueAntiquantScaleTensor, const at::Tensor& queryQuantScaleTensor)
```

- **PTA 调用 kernel**

```cpp
FiaKernelFullQuant<<<blockDimToBeSet, nullptr, stream>>>(
    (uint8_t*)(queryTensor.mutable_data_ptr()),
    (uint8_t*)(keyTensor.mutable_data_ptr()),
    (uint8_t*)(valueTensor.mutable_data_ptr()),
    (uint8_t*)(keyAntiquantScaleTensor.mutable_data_ptr()),
    (uint8_t*)(valueAntiquantScaleTensor.mutable_data_ptr()),
    (uint8_t*)(queryQuantScaleTensor.mutable_data_ptr()),
    (uint8_t*)(outputTensor.mutable_data_ptr()),
    workspace,
    tilingDataDevice
);
```

### 11.2 PTA 侧详细流程

1. **PTA 对外接口定义**

```cpp
at::Tensor ascendc_fia(const at::Tensor& queryTensor, const at::Tensor& keyTensor, ...)
```

2. **PTA 对外接口注册**

```cpp
TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("ascendc_fia(Tensor query, Tensor key, Tensor value, "
          "Tensor keyAntiquantScale, Tensor valueAntiquantScale, "
          "Tensor queryAntiquantScale) -> Tensor");
}
TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m) {
    m.impl("ascendc_fia", TORCH_FN(ascendc_ops::ascendc_fia));
}
```

3. **PTA 接口内检测输入是否合规**

4. **PTA 接口内构造输出**

```cpp
at::Tensor outputTensor = at::empty({shapeQueryTensor[0], shapeQueryTensor[1],
    shapeQueryTensor[2], shapeQueryTensor[3] * 2},
    at::dtype(queryTensor.dtype()).device(queryTensor.device()).layout(queryTensor.layout()));
```

5. **PTA 接口构造 tilingdata**

```cpp
optiling::TilingContext context = InitContext(shapeQueryTensor, shapeKeyTensor, shapeValueTensor);
optiling::DoOpTilingFusedInferAttentionScore(&context);
```

6. **PTA 结构构造 kernel 输入**

```cpp
// 获取 blockDim 和 stream
uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(...);
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
```

7. **<<<>>> 直调 kernel**

```cpp
FiaKernelFullQuant<<<blockDimToBeSet, nullptr, stream>>>(...);
```

---

## 参考链接

- [torch.library 官方文档](https://pytorch.org/docs/stable/library.html)
- [Ascend C 算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/)
