# SparseFlashAttention op_api UT 越界内存访问问题分析报告

**生成时间**: 2026-03-19
**分析范围**: `attention/sparse_flash_attention/op_host/op_api/` 及 `tests/ut/op_api/`
**日志文件**: `./log/sfa_opapi_ut.log`

---

## 一、问题现象

运行 `transformer_op_api_ut` 时，日志显示以下关键错误：

```
[ RUN      ] sparse_flash_attention_opapi_ut.sparse_flash_attention_aclnn_0
[ERROR] RUNTIME: GetResInCurrentThread:ctx is NULL!
[ERROR] RUNTIME: rtsGetResInCurrentThread:ErrCode=107002, desc=[the context is a null pointer]
[ERROR] ASCENDCL: aclrtGetResInCurrentThreadImpl:call rtsGetResInCurrentThread failed, runtime result = 107002
[ERROR] Build failed for target: transformer_op_api_ut.
```

测试进程非正常退出，无法输出 `[ PASSED ]` 或 `[ FAILED ]`，说明进程发生了崩溃（ASAN 检测到内存违规后直接终止进程）。ASAN 相关的 heap-use-after-free / stack-use-after-scope 输出通常写入 `stderr`，未被日志文件捕获。

---

## 二、核心 Bug 分析

### Bug 1（最严重）：TensorHolder 析构导致 Use-After-Free —— 悬空指针越界访问

**文件**: `op_host/op_api/aclnn_sparse_flash_attention.cpp`
**位置**: 第 114–133 行（`aclnnSparseFlashAttentionGetWorkspaceSize` 函数的 else 分支）

**问题代码**:

```cpp
} else {
    if (softmaxMax == nullptr && softmaxSum == nullptr) {
        auto softmaxMaxHolder = TensorHolder(softmaxMax, aclDataType::ACL_FLOAT, std::string("softmaxMax"));
        auto softmaxSumHolder = TensorHolder(softmaxSum, aclDataType::ACL_FLOAT, std::string("softmaxSum"));
        // TensorHolder 构造函数将 softmaxMax/softmaxSum 赋值为新创建的 aclTensor*
        if (softmaxMax == nullptr) { ... }
        if (softmaxSum == nullptr) { ... }
    }
    // ← 此处 if 块结束，softmaxMaxHolder 和 softmaxSumHolder 析构！
    // ← 析构函数调用 aclDestroyTensor(inner_)，释放了 tensor 内存
    // ← 但 softmaxMax 和 softmaxSum 依然持有已释放内存的指针 —— 悬空指针！
}
// softmaxMax / softmaxSum 此时为悬空指针
return aclnnInnerSparseFlashAttentionGetWorkspaceSize(
    ..., softmaxMax, softmaxSum, workspaceSize, executor);  // ← UAF！越界访问！
```

**根因分析**:

`TensorHolder` 的设计意图是：当输出张量为 `nullptr` 时，自动创建一个占位张量并在析构时清理。但 `softmaxMaxHolder` / `softmaxSumHolder` 的作用域仅限于 `if` 块内部，而持有者将修改后的 `softmaxMax` / `softmaxSum` 指针（此时已为非空）传递给 `if` 块外部的 `aclnnInnerSparseFlashAttentionGetWorkspaceSize`。当 `if` 块结束，两个 `TensorHolder` 对象被析构，底层 `aclTensor` 内存被释放，但 `softmaxMax` / `softmaxSum` 仍持有指向已释放内存的指针。

**触发条件**: `returnSoftmaxLse == false` **且** `softmaxMax == nullptr && softmaxSum == nullptr`

**后果**: 在 ASAN 开启的情况下，`aclnnInnerSparseFlashAttentionGetWorkspaceSize` 内部访问已释放的 `softmaxMax` / `softmaxSum` 指针，ASAN 检测到 heap-use-after-free，立即终止进程。

**修复方向**（仅供参考，不修改源文件）:
`TensorHolder` 对象应声明在函数作用域（而非 `if` 块内部），确保其生命周期覆盖到 `aclnnInnerSparseFlashAttentionGetWorkspaceSize` 调用结束之后，例如：

```cpp
// 示意性修复思路（不动原代码）：
TensorHolder softmaxMaxHolder(softmaxMax, ACL_FLOAT, "softmaxMax");  // 声明在函数级别
TensorHolder softmaxSumHolder(softmaxSum, ACL_FLOAT, "softmaxSum");
...
return aclnnInnerSparseFlashAttentionGetWorkspaceSize(..., softmaxMax, softmaxSum, ...);
// 函数返回后 holder 才析构，生命周期正确
```

---

### Bug 2：TensorHolder 构造函数中传入栈变量地址作为 device data

**文件**: `op_host/op_api/aclnn_sparse_flash_attention.cpp`
**位置**: 第 52–55 行（`TensorHolder` 构造函数）

**问题代码**:

```cpp
TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
    ...
    if (output == nullptr) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;          // ← 栈上的局部变量
        inner_ = aclCreateTensor(shape.data(), shape.size(),
            dataType, shape.data(), 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), static_cast<void *>(&addr));  // ← 传入栈地址！
        output = inner_;
    }
}
```

**根因分析**:

`int64_t addr = 0xff;` 是构造函数内部的局部变量，存在于栈上。`&addr` 的有效性仅在构造函数执行期间有保证。构造函数返回后，此栈地址即为无效地址（stack-use-after-scope）。若框架在后续处理 `aclTensor` 时尝试读取其 device data 指针（即 `&addr`），会触发 ASAN 的 stack-use-after-scope 报错或 segfault。

---

### Bug 3：`IsTensorNotNull()` 返回值语义相反

**文件**: `op_host/op_api/aclnn_sparse_flash_attention.cpp`
**位置**: 第 75–77 行

**问题代码**:

```cpp
bool IsTensorNotNull() const {
    return inner_ == nullptr;   // ← 逻辑反转！inner_ 为空时返回 true
}
```

**根因分析**:
函数名语义为"张量不为空"，但实际实现返回 `inner_ == nullptr`，即在 `inner_` 为 `nullptr` 时返回 `true`，与函数名相反。当前代码中此函数未被调用，但若未来使用会产生逻辑错误。

---

## 三、UT 测试用例问题分析

### 问题 4：运行时 Context 未初始化（日志中直接可见的错误）

**文件**: `tests/ut/op_api/test_aclnn_sparse_flash_attention.cpp`
**位置**: 第 25–35 行（`SetUpTestCase`）

**日志对应行**:
```
[ERROR] RUNTIME: GetResInCurrentThread:ctx is NULL! (ErrCode=107002)
[ERROR] ASCENDCL: call rtsGetResInCurrentThread failed, runtime result = 107002
```

**根因分析**:
`SetUpTestCase` 中仅调用了 `op::SetPlatformSocVersion`，未初始化 ACL runtime context（即缺少 `aclrtSetDevice` / `aclrtCreateContext` 等调用或等效 mock）。`aclnnInnerSparseFlashAttentionGetWorkspaceSize` 内部通过 `rtsGetResInCurrentThread` 获取当前线程的运行时资源，由于没有有效 context，返回错误码 107002。

这是 UT 环境搭建问题，导致 `aclRet != ACL_SUCCESS`，`EXPECT_EQ(aclRet, ACL_SUCCESS)` 断言失败。

---

### 问题 5：OUTPUT 中 TensorDesc 使用空 shape `{}`

**文件**: `tests/ut/op_api/test_aclnn_sparse_flash_attention.cpp`
**位置**: 第 116–120 行

**问题代码**:

```cpp
OUTPUT(
    TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND),   // attentionOut: 空 shape
    TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND),     // softmaxMax:   空 shape
    TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND)      // softmaxSum:   空 shape
)
```

**根因分析**:
`TensorDesc({}, ...)` 创建 `view_dims_ = {}` 的零维 tensor。在 `ToAclTypeRawPtr()` 中：

```cpp
aclTensor * acl_tensor = aclCreateTensor(
    view_dims_.data(),  // 空向量的 .data()，行为依赖实现
    view_dims_.size(),  // 0
    ...
    storage_dims.data(), storage_dims.size(),  // 同样是空
    storage_data);  // nullptr
```

空 shape 的 tensor 在某些框架版本中可能引发内部校验失败。更重要的是，`softmaxMax` 和 `softmaxSum` 此时**不为 nullptr**（是有效的零维 aclTensor），因此 Bug 1 的代码路径（`softmaxMax == nullptr && softmaxSum == nullptr`）在此 UT 中**不会被触发**。

但 `attentionOut` 的正确 shape 应为 `{2, 1, 32, 64}`（与 query 相同），空 shape 会使框架无法推导输出内存大小，进而可能引发其他错误。

---

## 四、执行路径还原（本次 UT 的实际调用链）

```
TestGetWorkspaceSize()
  └─ GetWorkspaceSize()
       └─ GetConvertedAclArgs()           # TensorDesc -> aclTensor* 转换
       └─ InvokeGetWorkspaceSizeFunc()
            └─ aclnnSparseFlashAttentionGetWorkspaceSize(
                   query={2,1,32,64}, key={2,128,1,64}, ...,
                   returnSoftmaxLse=false,
                   attentionOut={空shape}, softmaxMax={空shape}, softmaxSum={空shape},
                   ...)
                    # returnSoftmaxLse=false, softmaxMax!=null, softmaxSum!=null
                    # → else 分支，但内部 if(softmaxMax==null && softmaxSum==null) 不执行
                    # → 直接调用 inner 函数
                    └─ aclnnInnerSparseFlashAttentionGetWorkspaceSize(...)
                         └─ 内部访问 rtsGetResInCurrentThread
                              └─ ErrCode=107002 (ctx is NULL) ← 此处报错
```

**结论**：本次 UT 运行中，Bug 1 因 softmaxMax/softmaxSum 非空而未触发，主要报错是 Bug 4（runtime context 缺失）。但 Bug 1 在其他调用路径下（softmaxMax/softmaxSum 传 nullptr 时）会触发 heap-use-after-free，这是代码中最危险的内存安全问题。

---

## 五、问题汇总表

| # | 类型 | 严重度 | 文件 | 行号 | 描述 |
|---|------|--------|------|------|------|
| 1 | Use-After-Free（悬空指针） | **严重** | `aclnn_sparse_flash_attention.cpp` | 115–132 | TensorHolder 析构后 softmaxMax/softmaxSum 成为悬空指针，被传入 inner 函数 |
| 2 | Stack-Use-After-Scope | **严重** | `aclnn_sparse_flash_attention.cpp` | 52–55 | TensorHolder 构造函数传入栈变量地址作为 device data |
| 3 | 逻辑错误 | **中** | `aclnn_sparse_flash_attention.cpp` | 75–77 | `IsTensorNotNull()` 返回值语义与函数名相反 |
| 4 | UT 环境问题 | **高** | `test_aclnn_sparse_flash_attention.cpp` | 25–35 | SetUpTestCase 未初始化 runtime context，导致 ctx is NULL 错误 |
| 5 | UT 参数错误 | **中** | `test_aclnn_sparse_flash_attention.cpp` | 116–120 | OUTPUT 中 TensorDesc 使用空 shape `{}`，attentionOut 应指定正确形状 |

---

## 六、建议修复优先级

1. **优先修复 Bug 1**：将 `TensorHolder` 对象移出 `if` 块，扩展其生命周期至函数返回之后（这是内存安全问题，会导致进程崩溃）。
2. **修复 Bug 2**：避免将栈变量地址作为 device data 传入 `aclCreateTensor`，改用 `nullptr` 或通过 `MallocDeviceMemory` 分配。
3. **修复 Bug 4（UT）**：在 `SetUpTestCase` 中补充 runtime context 的初始化或 mock，使 `aclnnInner*` 函数能够正常执行。
4. **修复 Bug 5（UT）**：为 OUTPUT 的 `attentionOut` 指定正确 shape（如 `{2, 1, 32, 64}`）。
5. **修复 Bug 3**：修正 `IsTensorNotNull()` 的返回值为 `inner_ != nullptr`。
