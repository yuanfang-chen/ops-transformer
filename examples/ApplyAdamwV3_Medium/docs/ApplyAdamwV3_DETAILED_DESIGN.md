# ApplyAdamwV3 设计文档

## 1. 概述

### 1.1 算子功能
ApplyAdamwV3算子实现AdamW优化器功能，用于神经网络训练中的权重更新。该算子融合了一阶动量更新、二阶动量更新和权重衰减，支持AMSGrad变体和梯度最大化模式。

### 1.2 数学公式
参见需求文档中的完整数学公式。

## 2. 架构设计（4+1 视图）

### 2.1 逻辑视图（Logical View）

```
┌─────────────────────────────────────────────────────────────┐
│                      ApplyAdamwV3 系统架构                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  op_graph   │    │   op_host   │    │  op_kernel  │     │
│  │ (图模式适配) │    │ (Host侧逻辑) │    │ (Kernel实现) │     │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤     │
│  │ • proto.h   │    │ • _def.cpp  │    │ • .cpp      │     │
│  │             │    │ • _tiling   │    │ • _dag.h    │     │
│  │             │    │             │    │ • _tiling_  │     │
│  │             │    │             │    │   struct.h │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                   ┌─────────────┐                          │
│                   │   Tiling    │                          │
│                   │    Data     │                          │
│                   └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**模块职责说明**：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_graph** | 图模式适配，定义算子 IR 原型 | `apply_adamw_v3_proto.h` |
| **op_host** | Host 侧逻辑：算子定义、Tiling 切分 | `_def.cpp`, `_tiling.cpp` |
| **op_kernel** | Kernel 侧实现：NPU 计算逻辑 | `.cpp`, `_dag.h` |

### 2.2 开发视图（Development View）

```
apply_adamw_v3/
├── CMakeLists.txt                    # 构建配置
├── README.md                         # 算子说明
├── docs/                             # 设计文档
│   ├── ApplyAdamwV3_REQUIREMENT_ANALYSIS.md
│   ├── ApplyAdamwV3_DETAILED_DESIGN.md
│   └── ApplyAdamwV3_TEST_DESIGN.md
├── examples/                         # 调用示例
│   └── test_aclnn_apply_adamw_v3.cpp
├── op_api/                           # ACLNN 接口
│   ├── aclnn_apply_adamw_v3.h
│   ├── aclnn_apply_adamw_v3.cpp
│   ├── apply_adamw_v3.h
│   └── apply_adamw_v3.cpp
├── op_host/                          # Host侧实现
│   ├── apply_adamw_v3_def.cpp      # 算子信息库
│   ├── apply_adamw_v3_infershape.cpp # Shape推导
│   └── arch32/                       # Ascend910B架构
│       ├── apply_adamw_v3_tiling.h
│       └── apply_adamw_v3_tiling.cpp
├── op_kernel/                        # Kernel侧实现
│   ├── apply_adamw_v3.cpp          # Kernel入口
│   └── arch32/                       # Ascend910B实现
│       ├── apply_adamw_v3_dag.h
│       └── apply_adamw_v3_tiling_struct.h
├── op_graph/                         # 图模式适配
│   └── apply_adamw_v3_proto.h
└── tests/                            # 测试代码
    ├── ut/                           # 单元测试
    │   └── op_host/
    │       └── test_aclnn_apply_adamw_v3.cpp
    └── st/                           # 集成测试
        └── aclnn/
            ├── CMakeLists.txt
            └── test_aclnn_apply_adamw_v3.cpp
```

### 2.3 运行视图（Process View）

**ACLNN 调用流程**：

```
用户代码
   │
   ▼
aclnnApplyAdamwV3GetWorkspaceSize()
   │
   ├──▶ op_host/_def.cpp: 获取算子定义
   ├──▶ op_host/_tiling.cpp: 计算 Tiling 参数
   │        │
   │        └──▶ 生成 TilingData
   │
   ▼
aclnnApplyAdamwV3()
   │
   ├──▶ 加载 Kernel 二进制
   ├──▶ 传递 TilingData 到 Device
   │
   ▼
┌─────────────────────────────────────┐
│         NPU Device 执行              │
│                                     │
│  op_kernel/apply_adamw_v3.cpp      │
│     │                               │
│     ├──▶ 读取 TilingData            │
│     ├──▶ GM → UB 数据搬运           │
│     ├──▶ 执行 AdamW 计算逻辑        │
│     └──▶ UB → GM 结果写回           │
│                                     │
└─────────────────────────────────────┘
```

### 2.4 物理视图（Physical View）

**支持的芯片型号**：

| 芯片型号 | 架构宏 | 目录标识 | 说明 |
|---------|--------|---------|------|
| Ascend910B | DAV_2201 | arch32/ | Atlas A2 训练/推理 |

### 2.5 用户视图（Use Case View）

| 调用方式 | 说明 | 适用场景 |
|---------|------|---------|
| **ACLNN 调用** | 通过 aclnn 接口直接调用 | 单算子验证、离线推理 |
| **图模式调用** | 通过 GE 图引擎调用 | 模型训练、在线推理 |

## 3. 实现方案

### 3.0 实现方式概述

**实现方式**：原生 AscendC API 自定义 Kernel

**选择理由**：
1. 直接控制计算流程，便于调试和优化
2. 清晰的数据流和内存管理
3. 灵活支持 FP16/BF16/FP32 多种数据类型
4. 易于理解 AdamW 算法的计算过程

### 3.1 TilingKey 设计

**定义位置**: `op_kernel/apply_adamw_v3.cpp` (直接定义在 cpp 文件中)

| TilingKey | 数据类型 | AMSGrad | 说明 |
|-----------|---------|---------|------|
| 1 | FP16 | false | FP16 基础模式 |
| 2 | BF16 | false | BF16 基础模式 |
| 3 | FP32 | false | FP32 基础模式 |
| 11 | FP16 | true | FP16 AMSGrad模式 |
| 12 | BF16 | true | BF16 AMSGrad模式 |
| 13 | FP32 | true | FP32 AMSGrad模式 |

### 3.2 Kernel 类设计

**文件**: `op_kernel/arch32/apply_adamw_v3_dag.h`

```cpp
template <typename T>
class ApplyAdamwV3Kernel {
public:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;   // 输入队列（双缓冲）
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue; // 输出队列（双缓冲）
    TBuf<QuePosition::VECCALC> tmpBuf;              // 临时计算缓冲区
    
    // 核心方法
    void Init(...);     // 初始化 GM 地址、缓冲区、参数
    void Process();     // 主处理循环
    
private:
    void CopyIn(uint32_t loopIdx);   // GM → UB
    void Compute(uint32_t loopIdx);  // 计算逻辑
    void CopyOut(uint32_t loopIdx);  // UB → GM
};
```

### 3.3 数据流设计

```
GM(var, m, v, grad) 
    │
    ▼ CopyIn (DataCopy)
UB(varLocal, mLocal, vLocal, gradLocal) [原始类型 T]
    │
    ▼ Cast to float
UB(varFloat, mFloat, vFloat, gradFloat) [float]
    │
    ▼ Compute AdamW
UB(mOutFloat, vOutFloat, varOutFloat) [float]
    │
    ▼ Cast to T
UB(varOutLocal, mOutLocal, vOutLocal) [原始类型 T]
    │
    ▼ CopyOut (DataCopy)
GM(var_out, m_out, v_out)
```

### 3.4 API 映射（Compute 函数）

| 计算步骤 | API | 说明 |
|---------|-----|------|
| 梯度处理 | `Muls` | `gt = maximizeFactor * grad` |
| 一阶动量更新 | `Muls`, `Sub` | `m_out = m * beta1 - (beta1 - 1) * gt` |
| 二阶动量更新 | `Mul`, `Muls`, `Sub` | `v_out = v * beta2 - (beta2 - 1) * gt²` |
| 权重更新 | `Muls`, `Sqrt`, `Adds`, `Div`, `Add` | 组合计算最终权重更新 |

### 3.5 内存管理

| 缓冲区 | 用途 | 大小 |
|--------|------|------|
| **inQueue** | 输入数据 (var, m, v, grad) | 4 * tileLength * sizeof(T) |
| **outQueue** | 输出数据 (var_out, m_out, v_out) | 3 * tileLength * sizeof(T) |
| **tmpBuf** | 中间计算 (float 精度) | 12 * tileLength * 4 |

### 3.6 多核切分策略

```cpp
// 每个 Core 处理的数据量
uint32_t usedCoreNum = min(coreNum, totalLength);
uint32_t avgElementsPerCore = totalLength / usedCoreNum;
uint32_t tileNumPerCore = avgElementsPerCore / tileLength;
```

**切分原则**：
1. 按元素数量均匀切分到各核
2. 每个 Core 内按 tile 大小循环处理
3. 自动对齐到 32 字节边界

## 4. Host 侧实现

### 4.1 Tiling 类设计

**文件**: `op_host/arch32/apply_adamw_v3_tiling.h` / `.cpp`

```cpp
class ApplyAdamwV3Tiling {
public:
    ge::graphStatus RunTiling();
    
private:
    ge::graphStatus CheckShapeAndType();    // 参数校验
    ge::graphStatus ReadScalarInputs();     // 读取标量
    ge::graphStatus CalcTilingParams();     // 计算切分参数
    ge::graphStatus SetTilingData();        // 设置 TilingKey
};
```

### 4.2 参数校验

| 校验项 | 条件 | 错误码 |
|--------|------|--------|
| 标量输入 shape | shape.size == 1 | GRAPH_FAILED |
| 张量输入 shape | 与 var shape 相同 | GRAPH_FAILED |
| 数据类型一致性 | 与 var dtype 相同 | GRAPH_FAILED |

### 4.3 TilingData 结构体

**文件**: `op_kernel/arch32/apply_adamw_v3_tiling_struct.h`

```cpp
struct ApplyAdamwV3TilingData {
    uint32_t tilingKey;       // 模板选择
    uint32_t usedCoreNum;     // 使用的核数
    uint32_t totalLength;     // 总元素数
    uint32_t tileNumPerCore;  // 每核 tile 数量
    uint32_t tileLength;      // 每个 tile 元素数
    uint32_t alignNum;        // 对齐元素数
    float beta1Power;         // β1^t
    float beta2Power;         // β2^t
    float lr;                 // 学习率
    float weightDecay;        // 权重衰减
    float beta1;              // β1
    float beta2;              // β2
    float epsilon;            // ε
    float maximizeFactor;     // maximize 系数 (1 或 -1)
};
```

## 5. Kernel 侧实现

### 5.1 入口函数

**文件**: `op_kernel/apply_adamw_v3.cpp`

```cpp
extern "C" __global__ __aicore__ void apply_adamw_v3(
    GM_ADDR var, GM_ADDR m, GM_ADDR v,
    GM_ADDR beta1_power, GM_ADDR beta2_power,
    GM_ADDR lr, GM_ADDR weight_decay,
    GM_ADDR beta1, GM_ADDR beta2,
    GM_ADDR epsilon, GM_ADDR grad,
    GM_ADDR max_grad_norm,
    GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamwV3TilingData, tilingData, tiling);
    
    switch (tilingData.tilingKey) {
        case FP16_TILING_KEY:
            ProcessAdamW<half>(...);
            break;
        // ... 其他 case
    }
}
```

## 6. 性能优化

### 6.1 当前优化
- ✅ 多核并行：使用 `usedCoreNum` 动态核数
- ✅ 双缓冲：`BUFFER_NUM = 2` 支持 CopyIn/Compute 重叠
- ✅ float 中间精度：FP16/BF16 转换为 float 计算

### 6.2 后续优化方向
- 🔲 流水线优化：CopyIn/Compute/CopyOut 三级流水
- 🔲 AMSGrad 路径实现

## 7. 构建命令

### 7.1 编译算子
```bash
cd ops-math
bash build.sh --ops=apply_adamw_v3 --soc=Ascend910B -j8
```

### 7.2 编译并运行 UT
```bash
bash build.sh -u --ops=apply_adamw_v3
```

## 8. 风险评估

### 8.1 API 风险
- 原生 AscendC API 在 ascend910b 上可用性已验证

### 8.2 精度风险
- FP16/BF16 计算使用 float 中间精度
- 精度验证标准：FP16 双千分之一，FP32 双万分之一

### 8.3 应对措施
1. Phase 1 先使用基础实现验证功能
2. Phase 2 验证所有数据类型和边界场景
3. Phase 3 考虑性能优化

## 9. 交付件清单

| 模块 | 文件 | 必需 | 说明 |
|------|------|------|------|
| **算子定义** | `op_host/apply_adamw_v3_def.cpp` | ✅ | 算子信息库 |
| **Tiling** | `op_host/arch32/apply_adamw_v3_tiling.cpp` | ✅ | Tiling切分逻辑 |
| **Tiling** | `op_host/arch32/apply_adamw_v3_tiling.h` | ✅ | Tiling头文件 |
| **Tiling** | `op_kernel/arch32/apply_adamw_v3_tiling_struct.h` | ✅ | TilingData结构体 |
| **Kernel** | `op_kernel/apply_adamw_v3.cpp` | ✅ | Kernel入口函数 |
| **Kernel** | `op_kernel/arch32/apply_adamw_v3_dag.h` | ✅ | Kernel类定义 |
| **ACLNN** | `op_api/aclnn_apply_adamw_v3.h` | ✅ | ACLNN头文件 |
| **ACLNN** | `op_api/aclnn_apply_adamw_v3.cpp` | ✅ | ACLNN实现 |
| **ACLNN** | `op_api/apply_adamw_v3.h` | ✅ | L0接口头文件 |
| **ACLNN** | `op_api/apply_adamw_v3.cpp` | ✅ | L0接口实现 |
| **图模式** | `op_graph/apply_adamw_v3_proto.h` | 推荐 | IR定义 |
| **图模式** | `op_host/apply_adamw_v3_infershape.cpp` | 推荐 | Shape推导 |
| **构建** | `CMakeLists.txt` | ✅ | 构建配置 |
