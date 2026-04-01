# AOT (Ahead-of-Time) 编译优化示例

## 概述

本示例展示了昇腾 NPU 上 AOT（Ahead-of-Time）编译优化的实现原理。通过在编译阶段对 kernel 进行额外常量化编译，在运行时根据 tiling 参数匹配预编译的特化版本，从而实现性能优化。

## AOT 原理

### 核心思想

AOT 编译的核心思想是：**将运行时才确定的参数提前到编译阶段常量化**，让编译器能够进行更激进的优化，如：

- **循环展开（Loop Unrolling）**：基于常量边界展开循环
- **死代码消除（Dead Code Elimination）**：根据常量操作类型裁剪分支
- **常量传播（Constant Propagation）**：在编译期计算常量表达式
- **更好的指令调度**：利用常量信息进行优化排布

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      编译阶段 (Compile Time)                   │
├─────────────────────────────────────────────────────────────┤
│  模板特化版本1: tile_compute<..., AOT_VecOpTiling_32x32_add>      │
│    ├── m=32, n=32, op=1 (常量)                                │
│    └── 编译器优化: 循环展开、死代码消除、常量折叠                  │
│                                                              │
│  模板特化版本2: tile_compute<..., AOT_VecOpTiling_16x16_rds>      │
│    ├── m=16, n=16, op=2 (常量)                                │
│    └── 编译器优化: 同上                                        │
│                                                              │
│  通用版本: tile_compute<..., TilingRuntimeHolder>                │
│    └── m, n, op 为运行时变量，无特殊优化                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      运行阶段 (Runtime)                        │
├─────────────────────────────────────────────────────────────┤
│  1. 用户传入 VecOpTiling 参数 (m, n, op)                       │
│  2. 匹配阶段: 通过 AOTDispatcher 自动分发                      │
│     → AOTDispatcher 遍历注册的 AOT Holder                      │
│     → 运行时比较参数，匹配对应的 AOT 版本                      │
│     → 调用对应的 tile_compute 特化版本                        │
│     → 未匹配则调用通用版本                                    │
└─────────────────────────────────────────────────────────────┘
```

## 代码详解

### 1. Tiling 数据结构

```cpp
struct VecOpTiling {
    int32_t m;   // 矩阵行数
    int32_t n;   // 矩阵列数
    int32_t op;  // 操作类型: OP_ADD=1, OP_RDS=2
};
```

### 2. Holder 类型定义

```cpp
// 运行时 Holder - 使用运行时的 tiling 值
struct TilingRuntimeHolder {
    static constexpr uint8_t value[] = {0};  // 占位符
};

// AOT Holder - 编译时常量化 tiling 值
static constexpr VecOpTiling AOT_VecOpTiling_32x32_add_Value = {32, 32, 1};
using AOT_VecOpTiling_32x32_add = AOTHolder<VecOpTiling, &AOT_VecOpTiling_32x32_add_Value>;

static constexpr VecOpTiling AOT_VecOpTiling_16x16_rds_Value = {16, 16, 2};
using AOT_VecOpTiling_16x16_rds = AOTHolder<VecOpTiling, &AOT_VecOpTiling_16x16_rds_Value>;
```

### 3. 通用常量折叠框架

```cpp
// AOTHolder 基类 - 使用指针指向全局 constexpr 变量
template <typename T, const T* ValuePtr>
struct AOTHolder {
    using value_type = T;
    static constexpr const T* value_ptr = ValuePtr;
    static constexpr const T& value = *ValuePtr;

    // 转换为字节序列（兼容原设计）
    static constexpr const uint8_t* bytes() {
        return reinterpret_cast<const uint8_t*>(ValuePtr);
    }
};

// AOT 注册表
using VecOpTilingAOTRegistry = AOTRegistry<
    AOT_VecOpTiling_32x32_add,
    AOT_VecOpTiling_16x16_rds
>;

// AOT 分发器 - 单入口，自动检测并分发
template <typename T, typename Registry>
struct AOTDispatcher {
    template <int BM, int BN, typename Func>
    static void dispatch(T value, Func&& kernel_func) {
        dispatch_impl<0, BM, BN>(value, std::forward<Func>(kernel_func));
    }

private:
    template <size_t N, int BM, int BN, typename Func>
    static void dispatch_impl(T value, Func&& kernel_func) {
        if constexpr (N < Registry::size) {
            using Holder = typename Registry::template get<N>;

            // 运行时比较：检查是否匹配此 AOT Holder
            if (compare_equal(value, Holder::value)) {
                // 匹配：调用 AOT 版本（Holder 作为模板参数）
                kernel_func(BM, BN, Holder{}, value);
            } else {
                // 不匹配：检查下一个 Holder
                dispatch_impl<N + 1, BM, BN>(value, std::forward<Func>(kernel_func));
            }
        } else {
            // 所有 Holder 都不匹配：调用运行时版本
            kernel_func(BM, BN, TilingRuntimeHolder{}, value);
        }
    }

    static bool compare_equal(const T& a, const T& b) {
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }
};
```

### 4. 运行时/编译时分派

```cpp
// 调用 Kernel: <<<blockDim, smem, stream>>>(x, y)
VecOpTiling til = {M, N, 1};

AOTDispatcher<VecOpTiling, VecOpTilingAOTRegistry>::template dispatch<blkM, blkN>(
    til,
    [&](int BM, int BN, auto tiling_holder, VecOpTiling value) {
        using Holder = decltype(tiling_holder);

        if constexpr (std::is_same_v<Holder, TilingRuntimeHolder>) {
            printf("##### 使用运行时 Tiling\n");
            tile_compute<float, blkM, blkN, Holder><<<1, nullptr, stream>>>(
                (uint8_t*)x1Device, (uint8_t*)x2Device, (uint8_t*)yDevice, value);
        } else {
            printf("##### AOT matched! m=%d, n=%d, op=%d\n", Holder::value.m, Holder::value.n, Holder::value.op);
            tile_compute<float, blkM, blkN, Holder><<<1, nullptr, stream>>>(
                (uint8_t*)x1Device, (uint8_t*)x2Device, (uint8_t*)yDevice, value);
        }
    }
);
```

### 5. Kernel 模板

```cpp
template <typename T, int BM, int BN, typename TH=TilingRuntimeHolder>
__global__ __vector__ void tile_compute(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y, VecOpTiling t)
{
    static_assert(sizeof(VecOpTiling) == 12, "VecOpTiling size is not equal 12");
    const VecOpTiling *tiling = &t;  // 直接使用传入的参数，无需 GetTiling
    int m = tiling->m;
    int n = tiling->n;
    int OP = tiling->op;

    // 后续代码中，编译器可以基于常量进行优化
    // 例如: if (OP == OP_RDS) 可能被编译器优化为无条件执行或消除
}
```

## 优化效果

### 1. 常量传播优化

当使用 `AOT_VecOpTiling_32x32_add` 时，`tiling->m`、`tiling->n`、`tiling->op` 都是编译期常量：

```cpp
// 原始代码
int MC = (m + BM - 1) / BM;  // 若 m=32, BM=32 → MC=1
int NC = (n + BN - 1) / BN;  // 若 n=32, BN=32 → NC=1

for (int bx = 0; bx < MC; bx++) {      // 编译器可展开为单次迭代
    for (int by = 0; by < NC; by++) {  // 编译器可展开为单次迭代
        ...
    }
}
```

### 2. 死代码消除

```cpp
if (OP == OP_RDS) { ... }  // 若 OP=1 (OP_ADD)，整个分支被消除
if (OP == OP_ADD) { ... }  // 若 OP=1，保留并可能内联展开
```

### 3. 内存访问优化

常量信息允许编译器：
- 预计算内存偏移量
- 优化数据预取（prefetch）策略
- 更好地分配寄存器/UB 资源

## 扩展更多 AOT 版本

如需添加更多预编译配置，只需：

1. **定义新的全局常量**：
```cpp
static constexpr VecOpTiling AOT_VecOpTiling_64x64_add_Value = {64, 64, 1};
using AOT_VecOpTiling_64x64_add = AOTHolder<VecOpTiling, &AOT_VecOpTiling_64x64_add_Value>;
```

2. **更新注册表**：
```cpp
using VecOpTilingAOTRegistry = AOTRegistry<
    AOT_VecOpTiling_32x32_add,
    AOT_VecOpTiling_16x16_rds,
    AOT_VecOpTiling_64x64_add
>;
```

3. **无需修改调用逻辑**：AOTDispatcher 会自动处理新添加的 AOT 版本

## 文件说明

| 文件 | 说明 |
|------|------|
| `aot_example.asc` | 完整的 AOT 示例代码（含 kernel 和 host） |
| `framework.h` | 通用常量折叠框架实现 |
| `tile_copy.h` | 数据搬移辅助类定义 |
| `CMakeLists.txt` | 构建配置 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
make
./aot_example
```

运行时将根据输入参数自动匹配 AOT 版本或回退到通用版本。

## 注意事项

1. **二进制兼容性**：`AOTHolder::value` 的字节布局必须与 `VecOpTiling` 完全一致（考虑对齐和字节序）
2. **编译时间**：每个 AOT 特化版本都会增加编译时间和二进制体积
3. **选择策略**：只针对高频使用的配置预编译 AOT 版本，避免过度特化
4. **运行时开销**：匹配过程使用 `memcmp`，应控制 AOT 版本数量（建议 < 10 个）