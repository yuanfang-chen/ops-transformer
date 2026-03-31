# TilingKey 设计

## 传递模式 TilingKey

使用 `GET_TPL_TILING_KEY` 宏计算，编码多个模板参数：

```cpp
// 原始 Host 侧计算
uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
uint32_t tilingKeyQuantMode = quantMode;
tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, 
                    templateDispatch, commMode, TILINGKEY_TPL_A3);
context->SetTilingKey(tilingKey);

// 原始 Kernel 侧模板
template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, ..., GM_ADDR tilingGM)
```

参考：[Tiling模板编程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)

## 直调模式 TilingKey

使用简化的自定义编码，仅保留必要分支：

```cpp
/*
 * A3 TilingKey 说明（5位十进制数）
 * 第1位（个位）：quantMode: 0=不量化, 1=静态量化, 2=动态量化
 * 第2位（十位）：x输入类型: 0=float16, 1=bfloat16
 * 第3位（百位）：是否有smoothScale: 0=无, 1=有
 * 第4位（千位）：是否走fullmesh_v2: 0=不做, 1=做
 * 第5位（万位）：无实际含义
 */

// 示例 TilingKey 分支
case 10000:  // float16, 不量化, 无smoothScale, 非fullmesh
case 10002:  // float16, 动态量化, 无smoothScale, 非fullmesh
case 11000:  // float16, 不量化, 无smoothScale, fullmesh
case 11012:  // bfloat16, 动态量化, 无smoothScale, fullmesh
case 10110:  // bfloat16, 不量化, 有smoothScale, 非fullmesh
```

## TilingKey 计算示例

```cpp
// PTA 侧计算 TilingKey
int32_t CalculateTilingKey(
    int quant_mode, 
    bool is_bfloat16, 
    bool has_smooth_scale, 
    bool is_fullmesh_v2)
{
    int32_t tilingKey = 10000;  // 基础值
    tilingKey += quant_mode;                    // 个位：quantMode
    tilingKey += (is_bfloat16 ? 10 : 0);        // 十位：dtype
    tilingKey += (has_smooth_scale ? 100 : 0);  // 百位：smoothScale
    tilingKey += (is_fullmesh_v2 ? 1000 : 0);   // 千位：fullmesh
    return tilingKey;
}

// Kernel 侧 switch 分发
extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_generic(
    int32_t tilingKey, GM_ADDR x, ..., MoeDistributeDispatchV2Info tilingData)
{
    switch (tilingKey) {
        case 10000:  // float16, UNQUANT, no smoothScale, no fullmesh
            moe_distribute_dispatch_v2<float16_t, float16_t, UNQUANT, false>(
                x, ..., tilingData);
            break;
        case 11012:  // bfloat16, DYNAMIC_QUANT, no smoothScale, fullmesh
            moe_distribute_dispatch_v2<bfloat16_t, bfloat16_t, DYNAMIC_QUANT, false>(
                x, ..., tilingData);
            break;
        default:
            // 不支持的 TilingKey，可添加日志或默认处理
            break;
    }
}
```

## 设计原则

1. **精简原则**：仅保留实际业务需要的分支组合
2. **可读性**：使用清晰的编码规则，便于维护
3. **扩展性**：预留位数空间，方便后续添加新分支
4. **一致性**：与 Kernel 模板参数对应，避免遗漏