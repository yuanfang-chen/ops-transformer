# FlashAttentionScore 算子架构文档

## 概述

FlashAttentionScore 是华为昇腾AI处理器上的高效注意力机制算子实现，基于Flash Attention算法优化，用于Transformer模型的注意力计算。本文档详细说明了该算子的架构设计、模块组织及调用关系。

## 目录结构

```
flash_attention_score/
├── op_api/                    # API层实现
│   ├── flash_attention_score.h
│   ├── flash_attention_score.cpp
│   └── aclnn_flash_attention_score.h
├── op_host/                   # Host端实现
│   ├── flash_attention_score_def.cpp        # 算子定义
│   ├── flash_attention_score_infershape.cpp # 形状推理
│   ├── flash_attention_score_tiling.cpp      # Tiling计算
│   ├── flash_attention_score_tiling_common.h
│   ├── arch32/                            # Ascend910B架构
│   │   ├── flash_attention_score_tiling_general.cpp
│   │   ├── flash_attention_score_tiling_varlen.cpp
│   │   └── flash_attention_score_tiling_regbase.h
│   └── arch35/                            # Ascend950架构
│       ├── flash_attention_score_tiling_basic.cpp
│       ├── flash_attention_score_tiling_varlen.cpp
│       ├── flash_attention_score_tiling_dropmask.cpp
│       └── flash_attention_score_tiling_regbase.h
├── op_kernel/                  # Kernel端实现
│   ├── flash_attention_score.cpp           # 主入口
│   ├── flash_attention_score_apt.cpp       # APT版本
│   ├── arch32/                           # Ascend910B架构
│   │   ├── flash_attention_score_common.h
│   │   ├── flash_attention_score_tiling.h
│   │   ├── flash_attention_score_s1_bn2gs1.h
│   │   ├── flash_attention_score_s1s2_bn2gs1.h
│   │   ├── flash_attention_score_empty_tensor.h
│   │   ├── flash_attention_score_drop_mask_adapter.h
│   │   ├── flash_attention_var_len_score.h
│   │   └── basic_modules/
│   └── arch35/                           # Ascend950架构
│       ├── flash_attention_score_kernel_base.h
│       ├── flash_attention_score_kernel_train.h
│       ├── flash_attention_score_kernel_infer.h
│       ├── flash_attention_score_block_cube.h
│       ├── flash_attention_score_block_vec_base.h
│       ├── flash_attention_score_block_vec_train.h
│       ├── flash_attention_score_block_vec_infer.h
│       ├── flash_attention_score_entry_regbase.h
│       ├── flash_attention_score_template_tiling_key.h
│       └── vf/  # Vector算子实现
├── op_graph/                   # 图相关定义
│   └── flash_attention_score_proto.h
├── framework/                  # 框架集成
│   └── npu_flash_attention_score_onnx_plugin.cpp
├── tests/                      # 测试代码
│   ├── ut/
│   └── pytest/
├── docs/                       # 文档
└── examples/                   # 示例代码

common/                        # 通用基础设施
├── op_host/                    # Host端通用组件
│   ├── fia_tiling_base.h              # Tiling基础类
│   ├── fia_tiling_info.h              # Tiling信息结构
│   ├── fia_tiling_shape.h             # 形状处理
│   ├── fia_tiling_shape.cpp
│   ├── fia_tiling_templates_registry.h # 模板注册
│   └── split_core.h                  # 分核算法
└── op_kernel/                   # Kernel端通用组件
    ├── fia_public_define.h           # 公共定义
    ├── buffer_manager.h              # 缓冲区管理
    ├── buffer.h                     # 缓冲区定义
    ├── matmul.h                     # 矩阵乘法
    └── arch35/                     # Ascend950架构通用组件
        ├── flash_attention_score_kernel_base.h
        ├── flash_attention_score_tiling_regbase.h
        ├── flash_attention_score_common_regbase.h
        └── infer_flash_attention_comm.h
```

## 核心架构层次

### 1. API层 (op_api)

**文件**: `flash_attention_score.cpp`

**职责**:
- 提供高级API接口 `FlashAttentionScore()`
- 处理输入参数验证和转换
- 分配中间输出张量
- 调用形状推理和Kernel启动

**关键函数**:
```cpp
const std::array<const aclTensor *, 4> FlashAttentionScore(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, 
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    // ... 其他可选参数
    double scaleValue, double keepProb, int64_t preTockens,
    int64_t nextTockens, int64_t headNum, const char *inputLayout,
    // ... 其他属性
    aclOpExecutor *executor)
```

**输出**: 返回4个张量数组 [softmaxMax, softmaxSum, softmaxOut, attentionOut]

### 2. Host端实现层 (op_host)

#### 2.1 算子定义 (flash_attention_score_def.cpp)

**职责**:
- 定义算子输入输出规格
- 设置数据类型和格式约束
- 配置AI Core参数

**关键类**:
```cpp
class FlashAttentionScore : public OpDef {
public:
    explicit FlashAttentionScore(const char* name) : OpDef(name) {
        // 定义必需输入
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ...})
            .Format({ge::FORMAT_ND});
        
        // 定义可选输入
        this->Input("real_shift").ParamType(OPTIONAL);
        this->Input("drop_mask").ParamType(OPTIONAL);
        // ...
        
        // 定义输出
        this->Output("softmax_max").ParamType(REQUIRED);
        this->Output("softmax_sum").ParamType(REQUIRED);
        this->Output("softmax_out").ParamType(REQUIRED);
        this->Output("attention_out").ParamType(REQUIRED);
        
        // 定义属性
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("head_num").AttrType(REQUIRED).Int();
        // ...
    }
};
```

#### 2.2 形状推理 (flash_attention_score_infershape.cpp)

**职责**:
- 推理输出张量的形状和数据类型
- 支持多种布局格式 (BSH, SBH, BSND, BNSD, TND)

**关键函数**:
```cpp
ge::graphStatus InferShapeFlashAttentionScore(gert::InferShapeContext *context)
ge::graphStatus InferDataTypeFlashAttentionScore(gert::InferDataTypeContext *context)
```

**输出形状规则**:
- `softmax_max`: (B, N, S, 8) 或 (T, N, 8) [FP32]
- `softmax_sum`: 同softmax_max [FP32]
- `softmax_out`: (B, N, S, S) [与输入同类型]
- `attention_out`: 与query同形状，但D维度可能不同

#### 2.3 Tiling计算 (flash_attention_score_tiling.cpp)

**职责**:
- 计算数据分块策略
- 确定多核任务分配
- 生成Tiling数据传递给Kernel

**关键函数**:
```cpp
ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context)
ge::graphStatus TilingPrepareForFlashAttentionScore(gert::TilingParseContext *context)
```

**Tiling流程**:
1. 检查输入参数有效性
2. 处理空输入特殊情况
3. 调用架构特定的Tiling实现
4. 设置BlockDim和Workspace大小

### 3. Kernel端实现层 (op_kernel)

#### 3.1 主入口 (flash_attention_score.cpp)

**职责**:
- Kernel主函数入口
- 根据TilingKey选择实现模板
- 处理不同数据类型和布局组合

**关键宏定义**:
```cpp
#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)
#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)
#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)
```

**模板实例化**:
- 支持多种数据类型: FP16, BF16, FP32, FP8
- 支持多种布局: BSH, BSND, BNSD, SBH, TND
- 支持多种优化模式: 高精度、高性能、稀疏计算

#### 3.2 架构特定实现

**Arch32 (Ascend910B)**:
- `flash_attention_score_s1_bn2gs1.h`: S1切分模式
- `flash_attention_score_s1s2_bn2gs1.h`: S1S2切分模式
- `flash_attention_var_len_score.h`: 变长序列支持

**Arch35 (Ascend950)**:
- `flash_attention_score_kernel_base.h`: Kernel基础类
- `flash_attention_score_kernel_train.h`: 训练模式
- `flash_attention_score_kernel_infer.h`: 推理模式
- `flash_attention_score_block_cube.h`: Cube计算单元
- `flash_attention_score_block_vec_base.h`: Vector计算单元基础

### 4. 通用基础设施层 (common)

#### 4.1 Tiling基础框架 (op_host/fia_tiling_base.h)

**职责**:
- 提供Tiling计算的基础框架
- 定义统一的Tiling接口

**关键类**:
```cpp
class FiaTilingBase {
public:
    virtual ~FiaTilingBase() = default;
    
    ge::graphStatus DoTiling(TilingInfo *tilingInfo) {
        InitTilingInfo(tilingInfo);
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        if (DoOpTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

protected:
    virtual void InitTilingInfo(TilingInfo *tilingInfo) = 0;
    virtual bool IsCapable() = 0;
    virtual ge::graphStatus DoOpTiling() = 0;
};
```

#### 4.2 Tiling信息结构 (op_host/fia_tiling_info.h)

**职责**:
- 定义Tiling计算所需的所有参数和状态
- 支持多种配置模式

**关键结构**:
```cpp
struct FIARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FIAParaInfo {
    FIARequiredParaInfo query;
    FIARequiredParaInfo key;
    FIARequiredParaInfo value;
    // ... 可选参数
};

class FiaTilingInfo : public TilingInfo {
public:
    // 基础参数
    uint32_t bSize, n1Size, n2Size, s1Size, s2Size;
    uint32_t qkHeadDim, vHeadDim;
    float scaleValue;
    
    // 标志位
    bool emptyTensorFlag;
    bool pageAttentionFlag;
    bool antiQuantFlag;
    bool pseShiftFlag;
    bool attenMaskFlag;
    
    // 布局和数据类型
    FiaLayout qLayout, kvLayout, outLayout;
    FiaTilingInOutMode inOutMode;
    ge::DataType inputQType, inputKvType, outputType;
};
```

#### 4.3 分核算法 (op_host/split_core.h)

**职责**:
- 实现多核任务分配算法
- 支持负载均衡优化

**关键函数**:
```cpp
void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, 
              const SplitParam &splitParam, SplitResult &result);
void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, 
                  const SplitContext &splitContext, SplitResult &result);
```

**分核策略**:
- 按Batch分配
- 按行(S1G)分配
- 按块分配
- Flash Decode优化分配

#### 4.4 Kernel公共定义 (op_kernel/fia_public_define.h)

**职责**:
- 定义Kernel端公共常量和类型
- 提供工具函数

**关键定义**:
```cpp
namespace AttentionCommon {
    enum class FIA_LAYOUT : uint32_t {
        BSH = 0, BSND = 0, BNSD = 1, NZ = 2, TND = 3, NBSD = 4, NTD = 5
    };
    
    template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, 
              const bool PAGE_ATTENTION = false, const bool FLASH_DECODE = false, 
              FIA_LAYOUT LAYOUT_T = FIA_LAYOUT::BSH, ...>
    struct FIAType {
        using queryType = Q_T;
        using kvType = KV_T;
        using outputType = OUT_T;
        static constexpr bool pageAttention = PAGE_ATTENTION;
        static constexpr bool flashDecode = FLASH_DECODE;
        static constexpr FIA_LAYOUT layout = LAYOUT_T;
    };
    
    struct RunInfo {
        uint32_t bIdx, n2Idx, gS1Idx, s2Idx;
        uint64_t actS1Size, actS2Size;
        uint64_t tensorAOffset, tensorBOffset;
        // ...
    };
    
    struct ConstInfo {
        uint64_t batchSize, gSize, qHeadNum, kvHeadNum;
        uint64_t headDim, kvSeqSize, qSeqSize;
        float scaleValue;
        bool attenMaskFlag, batchContinuous;
        // ...
    };
}
```

#### 4.5 缓冲区管理 (op_kernel/buffer_manager.h)

**职责**:
- 管理不同存储层级的缓冲区
- 提供统一的内存分配接口

**关键类**:
```cpp
namespace fa_base_matmul {
    template<BufferType bufferType>
    class BufferManager {
    public:
        __aicore__ inline void Init(TPipe *pipe, uint32_t size);
        __aicore__ inline void Init(__gm__ uint8_t* workspace);
        
        template<SyncType syncType = SyncType::INNER_CORE_SYNC>
        __aicore__ inline Buffer<bufferType, syncType> AllocBuffer(uint32_t size);
    };
}
```

## 完整调用链路

### 1. 用户调用流程

```
用户代码
  ↓
FlashAttentionScore() API函数 (op_api/flash_attention_score.cpp)
  ↓
INFER_SHAPE() - 形状推理 (op_host/flash_attention_score_infershape.cpp)
  ↓
ADD_TO_LAUNCHER_LIST_AICORE() - 添加到启动列表
  ↓
FlashAttentionScore Kernel (op_kernel/flash_attention_score.cpp)
```

### 2. Host端Tiling流程

```
TilingFlashAttentionScore() (op_host/flash_attention_score_tiling.cpp)
  ↓
CheckParams() - 参数检查
  ↓
IsEmptyInput() / IsEmptyInputRegbase() - 空输入处理
  ↓
TilingRegistryArch::GetInstance().DoTilingImpl() - 架构特定Tiling
  ↓
FlashAttentionScoreTilingRegbase::DoTiling() (arch35/flash_attention_score_tiling_regbase.h)
  ↓
  ├─ GetPlatformInfo() - 获取平台信息
  ├─ GetShapeAttrsInfo() - 获取形状和属性信息
  ├─ DoOpTiling() - 执行Tiling计算
  │   ├─ CalcS1S2BasicBlock() - 计算S1S2基本块
  │   ├─ CalcDBasicBlock() - 计算D基本块
  │   ├─ SetMultiCoreParamsRegbase() - 设置多核参数
  │   └─ SetSparseParamsRegbase() - 设置稀疏参数
  ├─ GetTilingKey() - 生成TilingKey
  ├─ GetWorkspaceSize() - 计算Workspace大小
  └─ PostTiling() - 保存Tiling数据
```

### 3. Kernel端执行流程

```
flash_attention_score() Kernel入口 (op_kernel/flash_attention_score.cpp)
  ↓
根据TilingKey选择模板实例化
  ↓
  ├─ 空张量处理: FlashAttentionScoreEmptyTensor
  ├─ TND变长: FlashAttentionVarLenScore
  ├─ SameAB模式: FlashAttentionScoreS1s2Bn2gs1SameAB
  ├─ S1S2模式: FlashAttentionScoreS1s2Bn2gs1
  └─ S1模式: FlashAttentionScoreS1Bn2gs1
  ↓
FlashAttentionScoreKernelBase::InitBaseAPI() (common/op_kernel/arch35/flash_attention_score_kernel_base.h)
  ↓
  ├─ InitGlobalBuffer() - 初始化全局缓冲区
  ├─ InitLocalBuffer() - 初始化本地缓冲区
  ├─ ComputeConstexpr() - 计算编译期常量
  └─ InitMMResBuf() - 初始化矩阵乘法结果缓冲区
  ↓
FlashAttentionScoreKernelBase::Process()
  ↓
  ├─ Cube计算单元: FlashAttentionScoreBlockCube
  │   ├─ BMM1: Query × Key^T
  │   ├─ Vector1: Softmax + Mask处理
  │   └─ BMM2: Attention × Value
  └─ Vector计算单元: FlashAttentionScoreBlockVec
      ├─ 数据加载和预处理
      ├─ 累加和更新
      └─ 结果写回
```

## 数据流转

### 输入数据

1. **必需输入**:
   - `query`: 查询张量 [B, S, N, D] 或其他布局
   - `key`: 键张量 [B, S, N, D]
   - `value`: 值张量 [B, S, N, D]

2. **可选输入**:
   - `real_shift`: 实数偏移
   - `drop_mask`: Dropout掩码
   - `padding_mask`: 填充掩码
   - `atten_mask`: 注意力掩码
   - `prefix`: 前缀信息
   - `actual_seq_qlen`: Query实际序列长度
   - `actual_seq_kvlen`: KV实际序列长度
   - `query_rope`, `key_rope`: RoPE参数
   - `d_scale_q`, `d_scale_k`, `d_scale_v`: 反量化缩放因子

### 输出数据

1. **softmax_max**: Softmax最大值 [B, N, S, 8] [FP32]
2. **softmax_sum**: Softmax求和值 [B, N, S, 8] [FP32]
3. **softmax_out**: Softmax输出 [B, N, S, S] [输入类型]
4. **attention_out**: 注意力输出 [B, S, N, D] [输入类型]

### 计算流程

```
Query (B,S,N,D) × Key^T (B,D,N,S) 
  ↓ BMM1 (矩阵乘法)
Attention Score (B,S,N,S)
  ↓ Scale + Mask + Softmax
Attention Weight (B,S,N,S)
  ↓ BMM2 (矩阵乘法)
Attention Weight (B,S,N,S) × Value (B,S,N,D)
  ↓
Attention Output (B,S,N,D)
```

## 优化策略

### 1. 内存优化

- **双缓冲技术**: 减少数据搬运等待
- **L1/L0缓存复用**: 提高数据局部性
- **NZ格式**: 优化矩阵乘法内存访问
- **Workspace复用**: 减少内存分配

### 2. 计算优化

- **Flash Attention算法**: 在线Softmax，减少内存访问
- **分块计算**: 大矩阵分块处理，适配缓存
- **向量化**: 利用SIMD指令并行计算
- **多核并行**: 任务分配到多个AI Core

### 3. 特殊场景优化

- **空张量处理**: 快速返回零填充结果
- **变长序列**: 支持实际序列长度掩码
- **稀疏计算**: 支持因果掩码和带状掩码
- **量化计算**: 支持FP8输入和反量化

## 架构适配

### Ascend910B (Arch32)

- 支持数据类型: FP16, BF16, FP32
- 优化模式: 基础性能优化
- 特殊支持: 变长序列、Dropout

### Ascend950 (Arch35)

- 支持数据类型: FP16, BF16, FP32, FP8
- 优化模式: 高性能、高精度
- 特殊支持: 
  - Flash Attention V2
  - 稀疏计算优化
  - 量化计算
  - RoPE位置编码
  - Page Attention

## 配置参数

### 必需属性

- `head_num`: 注意力头数
- `input_layout`: 输入数据布局 (BSH/SBH/BSND/BNSD/TND)

### 可选属性

- `scale_value`: 缩放因子 (默认1.0)
- `keep_prob`: Dropout保留概率 (默认1.0)
- `pre_tokens`: 前缀token数
- `next_tokens`: 下一个token数
- `inner_precise`: 内部计算精度 (默认0)
- `sparse_mode`: 稀疏模式 (默认0)
- `pse_type`: PSE类型 (默认1)
- `seed`: 随机种子
- `offset`: 偏移量
- `out_dtype`: 输出数据类型
- `softmax_out_layout`: Softmax输出布局

## 错误处理

### 常见错误码

- `ACLNN_ERR_PARAM_INVALID`: 参数无效
- `ACLNN_ERR_INTERNAL`: 内部错误
- `GRAPH_FAILED`: 图处理失败
- `GRAPH_PARAM_INVALID`: 参数无效

### 错误检查点

1. API层: 输入张量验证
2. Host层: 形状和数据类型检查
3. Tiling层: 参数合法性验证
4. Kernel层: 运行时边界检查

## 性能调优建议

### 1. 数据布局选择

- **BSH**: 适合Batch维度较大的场景
- **SBH**: 适合序列长度较大的场景
- **BSND/BNSD**: 适合头数较多的场景
- **TND**: 适合变长序列和推理场景

### 2. 数据类型选择

- **FP16**: 平衡精度和性能
- **BF16**: 更好的数值稳定性
- **FP32**: 最高精度，性能较低
- **FP8**: 最低精度，最高性能（需支持）

### 3. 批处理大小

- 选择合适的Batch大小以充分利用AI Core
- 考虑内存带宽限制
- 避免频繁的核间同步

### 4. 序列长度

- 对齐到16或32的倍数
- 避免过小的序列长度
- 考虑填充对性能的影响

## 测试和验证

### 单元测试

- 位置: `tests/ut/`
- 覆盖: 形状推理、Tiling计算、Kernel功能

### 集成测试

- 位置: `tests/pytest/`
- 覆盖: 端到端功能、性能测试

### 示例代码

- 位置: `examples/`
- 提供: C++和Python调用示例

## 依赖关系

### 外部依赖

- CANN框架: 算子注册和执行
- AscendC: AI Core编程接口
- GE (Graph Engine): 图构建和优化

### 内部依赖

- common模块: 基础设施和工具
- 矩阵乘法库: BMM1和BMM2计算
- Softmax库: 注意力权重计算

## 扩展性设计

### 1. 新架构支持

通过继承`FiaTilingBase`和实现架构特定的Tiling类来支持新硬件架构。

### 2. 新数据类型

在算子定义和Kernel模板中添加新的数据类型支持。

### 3. 新布局格式

扩展`FiaLayout`枚举和相应的形状处理逻辑。

### 4. 新优化模式

通过TilingKey机制选择不同的Kernel实现模板。

## 总结

FlashAttentionScore算子采用了分层架构设计，从API层到Kernel层职责清晰，通过Tiling机制实现了灵活的数据分块和多核并行。common模块提供了可复用的基础设施，支持多种架构和数据类型。该实现充分利用了昇腾AI处理器的硬件特性，通过多种优化策略实现了高效的注意力计算。

---

**文档版本**: 1.0  
**最后更新**: 2025年  
**维护者**: Huawei Ascend Team