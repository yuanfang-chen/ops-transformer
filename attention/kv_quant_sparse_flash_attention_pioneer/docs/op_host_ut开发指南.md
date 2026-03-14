# KvQuantSparseFlashAttentionPioneer op_host UT 开发指南

本文档记录 KvQuantSparseFlashAttentionPioneer（QSFAP）算子 op_host 单元测试的完整开发流程，包含目录结构搭建、CMakeLists 编写、测试用例编写、构建运行，以及排障过程中踩过的坑。

## 1. 目录结构

最终交付的 UT 文件结构如下：

```
attention/kv_quant_sparse_flash_attention_pioneer/
├── op_host/
│   ├── kv_quant_sparse_flash_attention_pioneer_def.cpp          # OpDef 定义
│   ├── kv_quant_sparse_flash_attention_pioneer_infershape.cpp   # InferShape 实现
│   ├── kv_quant_sparse_flash_attention_pioneer_tiling.cpp       # Tiling 实现
│   └── ...
└── tests/
    ├── CMakeLists.txt                          # 递归扫描子目录
    └── ut/
        ├── CMakeLists.txt                      # 递归扫描子目录
        └── op_host/
            ├── CMakeLists.txt                  # InferShape UT 注册 + 递归扫描 arch 子目录
            ├── test_*_infershape.cpp           # InferShape 测试用例
            └── arch32/
                ├── CMakeLists.txt              # Tiling UT 注册
                └── test_*_tiling.cpp           # Tiling 测试用例
```

关键约定：
- **InferShape 测试** 放在 `tests/ut/op_host/` 下，与架构无关。
- **Tiling 测试** 放在 `tests/ut/op_host/arch32/` 下，因为 Tiling 策略因芯片架构而异（ascend910b 对应 arch32）。
- 测试文件命名必须匹配 `test_*_infershape.cpp` 或 `test_*_tiling.cpp`，构建系统通过 `file(GLOB ...)` 自动发现。

## 2. CMakeLists 编写

### 2.1 tests/CMakeLists.txt 和 tests/ut/CMakeLists.txt

这两级 CMakeLists 均为标准的递归扫描模板，无需自定义逻辑：

```cmake
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

### 2.2 tests/ut/op_host/CMakeLists.txt

注册 InferShape UT 源文件，并递归扫描 arch 子目录：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

`add_modules_ut_sources` 宏（定义在 `cmake/ut.cmake`）会自动用 `file(GLOB)` 匹配当前目录下的 `test_*_infershape.cpp`，将其编译到 `${OP_INFERSHAPE_MODULE_NAME}_cases_obj` 目标中。

### 2.3 tests/ut/op_host/arch32/CMakeLists.txt

注册 Tiling UT 源文件：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
```

同理，该宏自动匹配 `test_*_tiling.cpp`，编译到 `${OP_TILING_MODULE_NAME}_cases_obj` 目标中。

> **重要：不要在此处添加 `_def.cpp`。** 详见第 5 节"踩坑记录"。

## 3. 测试用例编写

### 3.1 UT 框架概述

ops-transformer 的 op_host UT 基于 GoogleTest，通过框架层封装了 `TilingContextFaker` 和 `InferShapeContextFaker`，模拟 CANN 运行时构造 `TilingContext` / `InferShapeContext`，驱动算子的 Tiling 或 InferShape 函数执行。

构建产物：
- **SO**（`libophost_transformer_ut.so`）：包含算子实现代码（`_tiling.cpp`、`_infershape.cpp` 等），通过 `IMPL_OP_OPTILING` / `IMPL_OP_INFERSHAPE` 宏完成算子注册。
- **EXE**（`transformer_op_host_ut`）：包含测试用例代码和 CANN 框架库（`libmetadef.so`、`libregister.so` 等）。EXE 在 `main()` 中通过 `AddSoToRegistry` dlopen 加载 SO，扫描并注册算子实现。

### 3.2 InferShape 测试用例

```cpp
#include <gtest/gtest.h>
#include <climits>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class QSFAPProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QSFAPProto SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "QSFAPProto TearDown" << std::endl; }
};

TEST_F(QSFAPProto, BSND_InferShape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvQuantSparseFlashAttentionPioneer",    // opName: 必须与 IMPL_OP_INFERSHAPE 注册名一致
        { /* 输入 TensorDescription 列表 */ },
        { /* 输出 TensorDescription 列表（shape 留空，由 InferShape 推导） */ },
        { /* 属性列表 */ });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 8, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

关键点：
- `opName` 必须与 `_infershape.cpp` 中 `IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttentionPioneer)` 的注册名完全一致。
- 输出 TensorDescription 的 shape 留空（`{{}，{}}`），由 InferShape 函数推导，测试用例验证推导结果是否等于 `expectOutputShape`。
- 可选输入通过空 shape（DimNum=0）表示"未提供"，框架会自动将其 IrInstanceNum 设为 0。

### 3.3 Tiling 测试用例

```cpp
#include <gtest/gtest.h>
#include <climits>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
struct QSFAPCompileInfo {
    int64_t coreNum;
};
} // namespace

class QSFAPTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QSFAPTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "QSFAPTiling TearDown" << std::endl; }
};

TEST_F(QSFAPTiling, PA_BSND_WithSink_TilingSuccess)
{
    QSFAPCompileInfo compileInfo = {24};   // compileInfo 不能为 nullptr!
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        { /* 输入 TensorDescription 列表 */ },
        { /* 输出 TensorDescription 列表 */ },
        { /* 属性列表 */ },
        &compileInfo,       // compileInfo 指针，必须非空
        "Ascend910B",       // SoC 版本
        24,                 // coreNum
        262144,             // ubSize
        8192);              // tilingDataSize

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_GT(tilingInfo.tilingKey, static_cast<int64_t>(0));
}
```

关键点：
- `compileInfo` **必须传非空指针**。`OpTilingContextBuilder::Build()` 内部检查 compileInfo，若为 nullptr 则返回空 ContextHolder，导致后续 `GetContext()` 返回 null、SEGV 崩溃。
- `QSFAPCompileInfo` 结构体应与算子 Tiling 函数中通过 `context->GetCompileInfo<T>()` 获取的类型一致。
- 正向用例使用 `ExecuteTiling()` + `EXPECT_TRUE`；反向用例使用 `ExecuteTestCase(para, ge::GRAPH_FAILED)` 验证 Tiling 函数返回失败。

### 3.4 TensorDescription 构造方式

```cpp
// 有效输入：提供完整 shape
{{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND}
//  ↑ storageShape  ↑ originShape

// 可选输入（未提供）：shape 留空，DimNum=0 → IrInstanceNum 自动设为 0
{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
```

### 3.5 属性设置

属性通过 `Ops::Transformer::AnyValue::CreateFrom<T>()` 构造：

```cpp
{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
{"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
{"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
```

属性的**顺序**必须与 `_def.cpp` 中 OpDef 定义的属性声明顺序一致，因为框架通过有序索引（`GetAttrs()->GetXxx(index)`）访问属性。

## 4. 构建与运行

### 4.1 环境准备

```bash
# 设置 CANN 环境变量
source /home/r00947674/CANN/cann0311/ascend-toolkit/set_env.sh
```

### 4.2 编译 UT

```bash
# 编译 UT（不执行）
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention_pioneer --noexec -j16

# 如需清理重编
bash build.sh --make_clean
bash build.sh -u --ophost --ops=kv_quant_sparse_flash_attention_pioneer --noexec -j16
```

构建产物路径：
- SO: `build/tests/ut/framework_normal/op_host/libophost_transformer_ut.so`
- EXE: `build/tests/ut/framework_normal/op_host/transformer_op_host_ut`

### 4.3 运行测试

```bash
export BUILD_PATH=$(pwd)/build
export LD_LIBRARY_PATH=${BUILD_PATH}/tests/ut/framework_normal/op_host:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

# 运行所有 QSFAP 测试
${BUILD_PATH}/tests/ut/framework_normal/op_host/transformer_op_host_ut --gtest_filter="*QSFAP*"

# 单独运行 Tiling 测试
${BUILD_PATH}/tests/ut/framework_normal/op_host/transformer_op_host_ut --gtest_filter="QSFAPTiling.*"

# 单独运行 InferShape 测试
${BUILD_PATH}/tests/ut/framework_normal/op_host/transformer_op_host_ut --gtest_filter="QSFAPProto.*"
```

### 4.4 调试技巧

遇到崩溃或异常时，开启 CANN 调试日志定位根因：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
${BUILD_PATH}/tests/ut/framework_normal/op_host/transformer_op_host_ut --gtest_filter="QSFAPTiling.PA_BSND_WithSink_TilingSuccess"
```

重点关注以下日志：
- `AddSoToRegistry` — SO 加载是否成功
- `Failed to dlopen` — SO 加载失败原因（通常是未定义符号）
- `OpImplRegisterV2` — 算子注册是否成功
- `BuildTilingContext` — Build() 是否报错

## 5. 踩坑记录

### 5.1 不要将 `_def.cpp` 编译到 UT 的 SO 中

**现象**：SO 加载失败，日志报：
```
[WARNING] Failed to dlopen .../libophost_transformer_ut.so!
    errmsg: undefined symbol: _ZN7metadef8RealPathEPKc
[WARNING] AddSoToRegistry: Nothing in so
```
随后 `Build()` 返回空 ContextHolder，`GetContext()` 返回 null，SEGV 崩溃。

**原因**：`_def.cpp` 中的 `OP_ADD` 宏引入了对 `metadef::RealPath`（位于 `libmetadef.so`）的依赖。UT 的 SO 不链接 `libmetadef.so`（只有 EXE 链接），因此 `dlopen(RTLD_NOW)` 无法解析该符号，SO 加载失败。

**结论**：算子的注册机制已由 `_tiling.cpp` 中的 `IMPL_OP_OPTILING` 和 `_infershape.cpp` 中的 `IMPL_OP_INFERSHAPE` 完成，无需在 UT CMakeLists 中额外引入 `_def.cpp`。其他现有算子也都不在 UT CMakeLists 中添加 `_def.cpp`。

### 5.2 compileInfo 不能传 nullptr

**现象**：SO 加载成功、算子注册成功，但 `Build()` 仍然返回空 ContextHolder。日志报：
```
[ERROR] BuildTilingContext: Compile info is nullptr
[ERROR] ctx_holder_impl is null while creating ContextHolder
```

**原因**：`OpTilingContextBuilder::Build()` 内部检查 compileInfo 是否为空，若为 nullptr 直接返回失败。

**修复**：所有 Tiling 测试用例的 `TilingContextPara` 构造函数中，compileInfo 参数传 `&compileInfo`（一个有效的结构体指针），而非 `nullptr`。

### 5.3 CANN 包版本一致性

构建时确保 `set_env.sh` 对应的 CANN 版本与目标一致。如果之前构建过其他版本，CMakeCache 会缓存旧路径，需要 `bash build.sh --make_clean` 清理后重新编译。

## 6. 测试用例覆盖

最终通过的 7 个测试用例：

| 测试套件 | 用例名 | 类型 | 验证点 |
|---------|--------|------|--------|
| QSFAPTiling | PA_BSND_WithSink_TilingSuccess | 正向 | PA_BSND 布局 + param_sink 特性，Tiling 成功且 tilingKey > 0 |
| QSFAPTiling | PA_BSND_NoSink_TilingSuccess | 正向 | PA_BSND 布局、无 sink 输入，Tiling 成功 |
| QSFAPTiling | BSND_BatchContinuous_TilingSuccess | 正向 | BSND 连续布局（非 PageAttention），Tiling 成功 |
| QSFAPTiling | Sink_NonPA_Layout_Fail | 反向 | BSND 布局下传入 sink 输入（非法），期望 GRAPH_FAILED |
| QSFAPTiling | SinkKeyOnly_Fail | 反向 | 只传 key_sink 不传 value_sink（非法），期望 GRAPH_FAILED |
| QSFAPProto | BSND_InferShape | 正向 | BSND 4D 输入，输出 shape = (B,S,N, qHeadDim-ropeHeadDim) |
| QSFAPProto | TND_InferShape | 正向 | TND 3D 输入，输出 shape = (T,N, qHeadDim-ropeHeadDim) |

运行结果：
```
[==========] Running 7 tests from 2 test suites.
[  PASSED  ] 7 tests.
```

## 7. 新增算子 UT 速查清单

为其他新算子新增 op_host UT 时，可参考以下步骤：

1. **创建目录结构**：`tests/CMakeLists.txt` → `tests/ut/CMakeLists.txt` → `tests/ut/op_host/CMakeLists.txt` → `tests/ut/op_host/arch32/CMakeLists.txt`，前两级用递归扫描模板。
2. **注册 InferShape UT**：在 `op_host/CMakeLists.txt` 调用 `add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} ...)`。
3. **注册 Tiling UT**：在 `op_host/arch32/CMakeLists.txt` 调用 `add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} ...)`。
4. **编写测试文件**：InferShape → `test_*_infershape.cpp`，Tiling → `test_*_tiling.cpp`。
5. **确保算子已注册**：`_tiling.cpp` 末尾有 `IMPL_OP_OPTILING(OpName)`，`_infershape.cpp` 末尾有 `IMPL_OP_INFERSHAPE(OpName)`。
6. **不添加 `_def.cpp`** 到任何 UT CMakeLists 中。
7. **Tiling 用例 compileInfo 不传 nullptr**。
8. **属性顺序**与 `_def.cpp` 中 OpDef 定义一致。
9. **编译**：`bash build.sh -u --ophost --ops=<op_name> --noexec -j16`。
10. **运行**：设置 `BUILD_PATH` 和 `LD_LIBRARY_PATH`，用 `--gtest_filter` 过滤执行。
