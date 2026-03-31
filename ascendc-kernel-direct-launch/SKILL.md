---
name: ascendc-kernel-direct-launch
description: Ascend C 算子 Kernel 直调改造指南。将算子从传统 Host/Kernel 传递模式改造为 Kernel 直调模式（使用 CUDA 风格的三重尖括号语法直接启动核函数），使算子能通过 torch.library 被 PyTorch 直接调用。使用场景：(1) 新增算子直调实现，(2) 将现有 mc2/moe 算子改造为直调模式，(3) 在 examples/fast_kernel_launch_example/csrc 下添加新算子。
---

# Ascend C Kernel 直调改造

将算子从 `op_host/op_kernel/op_api` 分离模式改造为单一目录的 <<<>>> 直调模式。

## 核心差异

| 维度 | 传递模式（原始） | 直调模式（改造） |
|------|-----------------|-------------------|
| 代码组织 | `op_host` + `op_kernel` 分离 | 单一目录统一编译 |
| Host 入口 | `gert::TilingContext*` GE 上下文 | 自定义 struct 直接传参 |
| Kernel 启动 | aclnn 二段式调度 | `<<<blockDim, nullptr, stream>>>` |
| Python 绑定 | aclnn API | pybind11 / torch.library |
| TilingKey | 全量模板支持 | 仅保留必要分支 |

## 改造工作流程

### Step 1: 创建目录结构

```
examples/fast_kernel_launch_example/csrc/
├── {operator}/ascend910_93/
│   ├── CMakeLists.txt              # add_sources("--npu-arch=dav-2201")
│   ├── {operator}_entry.h          # <<<>>>调用函数声明
│   ├── {operator}_entry.cpp        # Kernel入口 + <<<>>>调用
│   ├── {operator}_torch.h          # PyTorch工具函数
│   ├── {operator}_torch.cpp        # Host实现 + PyTorch绑定
│   └── op_kernel/                  # 精简的Kernel头文件
└── update_context/ascend910_93/    # MC2通信结构体（如需要）
```

### Step 2: Tiling 侧改造

1. **去掉 GE Context 依赖**：将 `gert::TilingContext*` 改为直接参数传递
2. **精简 TilingData**：移除 `Mc2InitTiling`、`Mc2CcTiling` 等通信结构体
3. **实现 Tiling 计算函数**：直接从 `at::Tensor` 获取参数

**API 改造对照**：见 [api-migration.md](references/api-migration.md)

### Step 3: Kernel 侧改造

1. **移除宏注册**：去掉 `GET_TILING_DATA_WITH_STRUCT`，改为直接传参
2. **精简 TilingKey**：自定义裁剪，仅保留必要分支
3. **创建通用入口**：`extern "C" __global__ __aicore__ void {operator}_generic(...)`

**TilingKey 设计**：见 [tilingkey-design.md](references/tilingkey-design.md)

### Step 4: 实现 <<<>>> 调用

```cpp
// {operator}_entry.cpp
void {operator}_entry(int32_t tilingKey, uint32_t blockDim, void* stream, 
    GM_ADDR x, ..., TilingData tilingData)
{
    {operator}_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, x, ..., tilingData
    );
}
```

### Step 5: PyTorch 绑定

1. **Schema 注册**：`TORCH_LIBRARY_FRAGMENT`
2. **Meta 函数**：实现 InferShape + InferDtype
3. **NPU 实现**：获取 Tensor 指针、计算 TilingKey、调用 entry
4. **注册实现**：`TORCH_LIBRARY_IMPL(PrivateUse1)` + `TORCH_LIBRARY_IMPL(Meta)`

**代码模板**：见 [pytorch-binding-template.md](references/pytorch-binding-template.md)

### Step 6: 通信结构体（MC2算子专用）

对于需要 HCCL 通信的算子（如 MoeDistribute），需创建 `update_context`：
- 从 `group_name` 获取 HCCL 通信句柄
- 填充 `Mc2ContextStru`（rankId、buffer地址）
- 返回 Tensor 传给算子 kernel

**通信结构体实现**：见 [mc2-context.md](references/mc2-context.md)

### Step 7: 编译与安装

```bash
cd examples/fast_kernel_launch_example
export NPU_ARCH=ascend910_93
python3 -m pip install -r requirements.txt
python3 -m build --wheel -n
python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

## 改造清单

必须改造项：
1. 创建 `{operator}_entry.h/cpp`（<<<>>>调用）
2. 创建 `{operator}_torch.h/cpp`（PyTorch绑定）
3. 精简 TilingData 结构
4. 实现 Tiling 计算函数（替换 GE Context）
5. 精简 TilingKey（仅保留必要分支）
6. 如需通信：创建 `update_context` 获取 mc2context

## 参考资源

- [原始算子示例 - moe_distribute_dispatch_v2](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v2)
- [改造后示例 - dispatch_v2](https://gitcode.com/cann/ops-transformer/tree/master/examples/fast_kernel_launch_example/csrc/moe_distribute_dispatch_v2)
- [PR #2141](https://gitcode.com/cann/ops-transformer/pull/2141)