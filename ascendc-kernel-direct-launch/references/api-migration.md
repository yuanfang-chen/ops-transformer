# API 改造对照表

## Tiling 侧 API 改造

| 原始 API (传递模式) | 改造后 API (直调模式) | 说明 |
|-------------------|---------------------|------|
| `gert::TilingContext*` | 直接参数传递 | 无需 GE 上下文 |
| `context->GetInputShape(i)` | `tensor.sizes()` | PyTorch API |
| `context->GetOutputShape(i)` | 根据输入推导 | 预留创建空 tensor |
| `context->GetTilingData<T>()` | 直接构造 TilingData | 手动填充结构体 |
| `context->GetNodeName()` | 移除 | 直接使用 std::cout |
| `OP_LOGE()` / `OP_LOGI()` | `std::cout` / `TORCH_CHECK` | 标准输出或异常检查 |
| `context->SetTilingKey()` | 直接计算并传参 | 自定义 TilingKey |
| `context->SetWorkSpace()` | 在 PTA 内构造 Tensor | workspace 作为 kernel 输入 |

## TilingData 结构改造

### 传递模式（原始）

```cpp
struct MoeDistributeDispatchV2TilingData {
    Mc2InitTiling mc2InitTiling;           // 直调模式移除
    Mc2CcTiling mc2CcTiling1;              // 直调模式移除
    Mc2CcTiling mc2CcTiling2;              // 直调模式移除
    MoeDistributeDispatchV2Info moeDistributeDispatchV2Info;
};
```

### 直调模式（改造）

```cpp
struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;         // blockDim
    // ... 其他业务参数
};
```

**关键变化**：`Mc2InitTiling` 和 `Mc2CcTiling` 通信配置转移到 `update_context` 处理。

## Kernel 侧 API 改造

| 原始 API | 改造后 API | 说明 |
|---------|-----------|------|
| `REGISTER_TILING_DEFAULT(T)` | 移除 | 不需要宏注册 |
| `GET_TILING_DATA_WITH_STRUCT(T, data, gm)` | 直接参数 `TilingData data` | 传参替代宏 |
| `moe_distribute_dispatch_v2_tiling_key.h` | 移除文件 | 自定义裁剪 TilingKey |

### Kernel 入口改造对比

**传递模式**：

```cpp
template<bool HasTp, uint8_t QuantMode, ...>
__global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, ..., GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
    op.Init(mc2Context, x, expertIds, ..., &pipe, &tilingData); 
    op.Process();
}
```

**直调模式**：

```cpp
template<typename XType, typename ExpandxType, int32_t QuantMode, bool IsSmoothScaleExist>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, ..., MoeDistributeDispatchV2Info tilingData)  // 直接传参
{
    MoeDistributeDispatchV2<XType, ExpandxType, QuantMode, IsSmoothScaleExist> op;
    op.Init(mc2Context, x, expertIds, ..., &pipe, tilingData);
    op.Process();
}

// 通用入口，根据 TilingKey 选择模板
extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_generic(
    int32_t tilingKey,
    GM_ADDR x, GM_ADDR expertIds, ..., 
    MoeDistributeDispatchV2Info tilingData)
{
    switch (tilingKey) {
        case 10000:
            moe_distribute_dispatch_v2<float16_t, float16_t, UNQUANT, false>(
                x, expertIds, ..., tilingData);
            break;
        // ... 更多 case
    }
}
```

## 数据类型映射

| PyTorch 类型 | Ascend C 类型 |
|-------------|--------------|
| `torch.float16` | `float16_t` |
| `torch.bfloat16` | `bfloat16_t` |
| `torch.int32` | `int32_t` |
| `torch.int8` | `int8_t` |
| `torch.float32` | `float32_t` |

## NPU 架构配置

| 架构 | 编译参数 |
|-----|---------|
| A2 | `--npu-arch=dav-2201` |
| A3 | `--npu-arch=dav-2201` |
| A5 | `--npu-arch=dav-3510` |