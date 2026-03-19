# 备注

```
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.610 [api_impl.cc:7751]695120 GetResInCurrentThread:ctx is NULL!
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.631 [api_c_stream.cc:749]695120 rtsGetResInCurrentThread:ErrCode=107002, desc=[the context is a null pointer], InnerCode=0x7070001
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.634 [error_message_manage.cc:55]695120 FuncErrorReason:rtsGetResInCurrentThread execution failed, the context is a null pointer.
[2026-03-19 15:04:57] [ERROR] ASCENDCL(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.642 [device.cpp:528]695120 aclrtGetResInCurrentThreadImpl:call rtsGetResInCurrentThread failed, runtime result = 107002.
[2026-03-19 15:04:57] [       OK ] prompt_flash_attention_v3_opapi_ut.prompt_flash_attention_v3_aclnn_1 (0 ms)
```

lcov op_api里面报上面的日志内容属于正常现象，不必检查此错误，重点排查其他错误问题

项目里面我负责的主要模块是

-   attention/prompt_flash_attention
-   attention/lightning_indexer
-   attention/sparse_flash_attention

这是三个算子，我的任务是写算子目录下的tests/ut，例如attention/prompt_flash_attention/tests/ut，你只能动这些目录下的文件，不能改算子原来的代码

lcov看护覆盖率中：

ut下面的op_host/arch32看护的是算子目录下的op_host/*tiling.cpp文件，例如attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp

ut下面的op_api看护的是算子目录下op_host/op_api/里面的算子接口，例如attention/prompt_flash_attention/op_host/op_api/aclnn_prompt_flash_attention.cpp

./log/下面是运行覆盖率命令后的日志文件，例如log/li_opapi_ut.log就是lightning_indexer关注op_api下面算子接口的覆盖日志

你的所有回答全部写在这个文件，包括代码。

# 问题

1、读取这个tests/ut/framework_normal/common/mc2_hcom_topology_mocker.cpp，这个文件是对mc2/common/utils/mc2_hcom_topo_info.h的打桩，op_host ut运行时，不会编mc2/common/utils/mc2_hcom_topo_info.cpp，而是编打桩文件（tests/ut/framework_normal/common/mc2_hcom_topology_mocker.cpp）中的打桩函数，请问这是怎么实现的（读取tests/），以及我想自己实现一个打桩，针对attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp中6244和6545的GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize实现一个自己的api接口，这两个api接口时/home/j60100428/Ascend_0320/cann-9.0.0/include/tiling/tiling_api.h头文件中的\#include "activation/softmax_tiling.h"引入的。

## 回答

### 一、mc2_hcom_topology_mocker.cpp 打桩机制分析

**核心文件：`mc2/common/CMakeLists.txt`，关键片段（第37-46行）：**

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
   # 步骤1：把真实实现文件从源文件列表中排除
   list(FILTER OPTILING_SRCS EXCLUDE REGEX "utils/mc2_hcom_topo_info\\.cpp")

   # 步骤2：把打桩文件编译成独立的动态库 mc2_tiling_ut_stub
   file(GLOB MC2_TILING_UT_STUB
       ${UT_COMMON_INC}/mc2_hcom_topology_mocker.cpp
       ${UT_COMMON_INC}/op_cache_tiling.cpp)
   add_library(mc2_tiling_ut_stub SHARED ${MC2_TILING_UT_STUB})
   ...
endif()
```

**步骤3：UT 可执行程序链接 stub 库（`tests/ut/framework_normal/op_host/CMakeLists.txt` 第63行）：**

```cmake
target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
    ...
    $<$<TARGET_EXISTS:mc2_tiling_ut_stub>:mc2_tiling_ut_stub>
    ...
)
```

**机制总结（三步缺一不可）：**

| 步骤 | 操作 | 目的 |
|------|------|------|
| 排除 | `list(FILTER ... EXCLUDE ...)` 把真实 `.cpp` 从编译列表去掉 | 避免符号冲突/ODR违规 |
| 替换 | 打桩 `.cpp` 编译成 `mc2_tiling_ut_stub.so` | 提供相同接口的可控实现 |
| 链接 | UT exe 链接 stub 库 | 运行时使用打桩函数 |

**打桩文件本身的设计模式：**

打桩文件 `mc2_hcom_topology_mocker.cpp`：
- `#include "mc2_hcom_topo_info.h"` ← 包含**同一个头文件**，保证符号签名一致
- 重新实现头文件中声明的所有类方法（`MC2HcomTopology::CommGetInstSizeByGroup` 等）
- 通过 `MC2HcomTopologyMocker` 单例（key-value map）让测试用例控制返回值
- 测试用例调用 `MC2HcomTopologyMocker::GetInstance().SetValue("rankNum", 16)` 即可控制行为

---

### 二、为 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize 实现打桩

**与 mc2 情况的关键区别：**

| | mc2_hcom_topo_info | GetSoftMaxMinTmpSize |
|---|---|---|
| 实现来源 | 项目内 `.cpp` 文件 | 外部预编译动态库 `tiling_api.so` |
| 打桩方式 | CMake 排除 `.cpp`，替换 stub | 编译期符号覆盖（ELF 符号解析规则） |

**ELF 动态链接符号解析规则（关键原理）：**

在链接生成 `libophost_transformer_ut.so` 时，如果目标文件（`.o`）中定义了 `GetSoftMaxMinTmpSize`，同时 `tiling_api.so` 也提供该符号，**链接器优先使用目标文件中的定义**，不再从 `tiling_api.so` 引入该符号。这与 mc2 机制的本质相同，只是触发方式不同（mc2 是排除 `.cpp`，这里是直接在目标中重定义）。

---

**实现步骤：**

#### 第1步：创建 mocker 头文件

新建 `tests/ut/framework_normal/common/softmax_tiling_mocker.h`：

```cpp
#pragma once
#include <cstdint>

// Mocker 单例，测试用例通过它控制 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize 的返回值
class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance() {
        static SoftmaxTilingMocker instance;
        return instance;
    }

    void SetSoftMaxMinTmpSize(uint32_t val) { softMaxMinTmpSize_ = val; }
    void SetSoftMaxFlashV2MinTmpSize(uint32_t val) { softMaxFlashV2MinTmpSize_ = val; }
    void Reset() { softMaxMinTmpSize_ = 0; softMaxFlashV2MinTmpSize_ = 0; }

    uint32_t GetSoftMaxMinTmpSize() const { return softMaxMinTmpSize_; }
    uint32_t GetSoftMaxFlashV2MinTmpSize() const { return softMaxFlashV2MinTmpSize_; }

private:
    SoftmaxTilingMocker() = default;
    uint32_t softMaxMinTmpSize_{0};
    uint32_t softMaxFlashV2MinTmpSize_{0};
};
```

#### 第2步：创建 stub 实现文件

新建 `tests/ut/framework_normal/common/softmax_tiling_mocker.cpp`：

```cpp
// 包含声明这两个函数的头文件，确保签名一致
#include "activation/softmax_tiling.h"
#include "softmax_tiling_mocker.h"

// 覆盖 tiling_api.so 中的 GetSoftMaxMinTmpSize
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
{
    return SoftmaxTilingMocker::GetInstance().GetSoftMaxMinTmpSize();
}

// 覆盖 tiling_api.so 中的 GetSoftMaxFlashV2MinTmpSize
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock,
    const bool isFlashOutputBrc)
{
    return SoftmaxTilingMocker::GetInstance().GetSoftMaxFlashV2MinTmpSize();
}
```

#### 第3步：在 CMakeLists.txt 中注册 stub

参考 mc2 的做法，在 PFA tiling 对应的 CMakeLists.txt（或 `mc2/common/CMakeLists.txt` 中统一管理的地方）的 `if(UT_TEST_ALL OR OP_HOST_UT)` 块中加入：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    # 将打桩文件编译进 mc2_tiling_ut_stub（或单独建 softmax_tiling_ut_stub）
    # 方式一：追加到已有 stub 库（最简单）
    target_sources(mc2_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}/softmax_tiling_mocker.cpp
    )
    target_include_directories(mc2_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}
        # softmax_tiling.h 所在的 include 路径
        ${ASCEND_DIR}/include/tiling
    )
endif()
```

或者单独建立一个库（适合需要独立管控的场景）：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_library(softmax_tiling_ut_stub SHARED
        ${UT_COMMON_INC}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}
        ${ASCEND_DIR}/include/tiling
    )
endif()
```

并在 `tests/ut/framework_normal/op_host/CMakeLists.txt` 的 exe 链接中加入：

```cmake
target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
    ...
    $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>
    ...
)
```

#### 第4步：在测试用例中使用

```cpp
#include "softmax_tiling_mocker.h"

TEST(PromptFlashAttentionTilingTest, TestCVDiff) {
    // 设置打桩返回值（非零值，防止 tiling 逻辑进入缩减循环）
    SoftmaxTilingMocker::GetInstance().SetSoftMaxMinTmpSize(1024);
    SoftmaxTilingMocker::GetInstance().SetSoftMaxFlashV2MinTmpSize(2048);

    // ... 构造 tiling context 并调用 ...

    // 测试结束后重置
    SoftmaxTilingMocker::GetInstance().Reset();
}
```

---

**注意事项：**

1. **签名必须完全一致**：stub 中函数签名（参数类型、顺序、const修饰）必须与 `softmax_tiling.h` 中声明的完全一致，否则链接时会产生两个不同的符号，打桩无效。
2. **链接顺序**：stub 库（`.so` 或链接进主 `.so` 的 `.o`）必须在链接时**先于** `tiling_api` 被处理。由于 `libophost_transformer_ut.so` 显式包含 stub 的 `.o`，ELF 链接器会优先使用本地定义，无需额外调整顺序。
3. **只影响调用方，不影响 tiling_api 内部**：如果 `tiling_api.so` 内部有其他函数调用 `GetSoftMaxMinTmpSize`，那些调用不会被 stub 拦截（内部调用走 PLT 的本地绑定），但从 `prompt_flash_attention_tiling.cpp` 发起的调用会被拦截。这对 UT 来说已经足够。

2、读取log/pfa_ophost_ut.log，为什么会报

```
[2026-03-24 17:37:12]CMake Error at attention/prompt_flash_attention/op_host/CMakeLists.txt:37 (add_library):
  Cannot find source file:

    //softmax_tiling_mocker.cpp
...
CMake Generate step failed.
```

感觉像是没识别到UT_COMMON_INC，是我写错cmakelists文件了？如果不是检索一下全仓，看看UT_COMMON_INC怎么设置的怎么读取。同时"/home/j60100428/asc-devkit/impl/adv_api/tiling/activation/softmax_tiling.cpp"就是调用api仓的源码，主要问题是两个函数里面platform_ascendc::PlatformAscendC* platform = platform_ascendc::PlatformAscendCManager::GetInstance();调用的是无参函数，而我想要传递的是GetInstance有参函数（函数原型在/home/j60100428/asc-devkit/include/utils/tiling/platform/下面），所以需要你再当前仓打桩重写这两函数，其他地方不变可以吗？

## 回答

### 一、`UT_COMMON_INC` 为空的根因

**不是你写错了，是 CMake 处理顺序问题。**

关键文件：`cmake/custom_build.cmake`

```cmake
# 第 289 行：先处理 attention（包含 PFA 的 op_host/CMakeLists.txt）
add_subdirectory(attention)

# 第 294 行：后处理 framework_normal → 才设置 UT_COMMON_INC
add_subdirectory(tests/ut/framework_normal)

# 第 339 行：mc2/common 在 framework_normal 之后，所以 mc2 能正常读到 UT_COMMON_INC
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mc2/common)
```

`UT_COMMON_INC` 在 `tests/ut/framework_normal/CMakeLists.txt` 第14行设置：
```cmake
set(UT_COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/common CACHE STRING "ut common include path" FORCE)
```

PFA 的 `op_host/CMakeLists.txt` 在第289行就被处理了，这时 `UT_COMMON_INC` 还没有被赋值（CACHE 里也还没有），所以 `${UT_COMMON_INC}` 展开为空字符串，路径变成 `//softmax_tiling_mocker.cpp`。mc2 的 `common/CMakeLists.txt` 在第339行处理，晚于第294行，所以它能正常读到 `UT_COMMON_INC`。

**修复方法**：在 `attention/prompt_flash_attention/op_host/CMakeLists.txt` 中不使用 `UT_COMMON_INC`，改用 `PROJECT_SOURCE_DIR` 硬编码路径（已修复）：

```cmake
# 修复前（错误）：
if(UT_TEST_ALL OR OP_HOST_UT)
    add_library(softmax_tiling_ut_stub SHARED
        ${UT_COMMON_INC}/softmax_tiling_mocker.cpp   # UT_COMMON_INC 此时为空！
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}
        ${ASCEND_DIR}/include/tiling
    )
endif()

# 修复后（正确）：
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}   # 包含所有 tiling 相关头文件路径，比 ${ASCEND_DIR}/include/tiling 更完整
    )
endif()
```

---

### 二、同时发现并修复了原 stub 的命名空间 bug

原始 `softmax_tiling_mocker.cpp` 中函数定义在**全局命名空间**，但 `softmax_tiling.h` 中函数声明在 `namespace AscendC` 内。两个符号名不同（`::GetSoftMaxMinTmpSize` vs `AscendC::GetSoftMaxMinTmpSize`），导致 stub 实际上没有覆盖 `tiling_api.so` 中的函数，打桩无效。

---

### 三、重写的打桩文件

#### `tests/ut/framework_normal/common/softmax_tiling_mocker.h`（已修改）

```cpp
#pragma once
#include <cstdint>
#include <string>

// UT 通过 SetSocVersion("ascend910b") 等设置 SoC，
// 打桩函数将用 PlatformAscendCManager::GetInstance(socVersion) 初始化平台。
class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance() {
        static SoftmaxTilingMocker instance;
        return instance;
    }
    void SetSocVersion(const std::string& socVersion) { socVersion_ = socVersion; }
    const char* GetSocVersion() const { return socVersion_.empty() ? nullptr : socVersion_.c_str(); }
    void Reset() { socVersion_.clear(); }
private:
    SoftmaxTilingMocker() = default;
    std::string socVersion_;
};
```

#### `tests/ut/framework_normal/common/softmax_tiling_mocker.cpp`（已修改）

关键变化：
- 函数定义放在 `namespace AscendC` 内（修复命名空间 bug）
- `PlatformAscendCManager::GetInstance()` → `PlatformAscendCManager::GetInstance(SoftmaxTilingMocker::GetInstance().GetSocVersion())`
- 计算逻辑与 `softmax_tiling.cpp` 原版完全一致，去掉了无法包含的内部 `HighLevelApiCheck` 宏（这些只是输入校验，不影响计算结果）

#### 在 UT 测试用例中使用

```cpp
#include "softmax_tiling_mocker.h"

// 测试前设置 SoC 版本（字符串与 PlatformAscendCManager 支持的 socVersion 一致）
SoftmaxTilingMocker::GetInstance().SetSocVersion("ascend910b");

// ... 执行 tiling，此时 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize
//     会用 ascend910b 对应的 NpuArch 计算出真实的 size 值 ...

// 测试结束后复位
SoftmaxTilingMocker::GetInstance().Reset();
```
