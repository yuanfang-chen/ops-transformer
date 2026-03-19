# KvQuantSparseFlashAttentionPioneer 算子代码梳理

## 1. 算子概述

### 1.1 功能定位
KvQuantSparseFlashAttentionPioneer 是面向长上下文推理场景的 Sparse Flash Attention 算子，基于 MLA（Multi-head Latent Attention）架构，支持 Per-Token-Head-Tile-128 量化输入。核心特点：
- 通过 `sparse_indices` 选择性计算关键 KV 块，大幅降低计算量
- 支持 FP8（E4M3FN / HiFloat8）量化 KV 输入，在 AIV 侧完成反量化
- AIC/AIV 异构三级流水架构，Cube 负责 BMM1/BMM2，Vector 负责数据搬运、反量化、Softmax
- 支持 PageAttention（PA_BSND）、BSND、TND 三种布局

### 1.2 计算公式
$$
Attention = \text{softmax}\left(\frac{Q \cdot \text{Dequant}(\tilde{K}^{FP8}, Scale_K)^T}{\sqrt{d_k}}\right) \cdot \text{Dequant}(\tilde{V}^{FP8}, Scale_V)
$$
- $Q$: Query 张量，BF16，D=576（nope 512 + rope 64）
- $\tilde{K}, \tilde{V}$: 经稀疏选择的 Key/Value，FP8 量化，D=656（nope 512 + rope 128 + scale 16）
- $Scale_K, Scale_V$: Per-Tile-128 量化参数，FP32，与 KV 数据 combine 存放
- $d_k$: Head Dimension，用于缩放

### 1.3 支持平台
| SoC | 架构 | 数据类型 |
|-----|------|---------|
| ascend950 (Atlas A5) | DAV_C310 | Q: BF16, KV: FP8_E4M3FN / HiFloat8, Out: BF16 |

---

## 2. 目录结构

```
kv_quant_sparse_flash_attention_pioneer/
├── CMakeLists.txt                                          # 顶层构建入口，递归添加子目录
├── README.md                                               # API 文档与调用示例
│
├── docs/                                                   # 文档目录
│
├── op_host/                                                # ═══ Host 侧（CPU 端）═══
│   ├── CMakeLists.txt                                      # Host 构建配置
│   ├── kv_quant_sparse_flash_attention_pioneer_def.cpp     # OpDef：算子接口定义（11输入1输出13属性）
│   ├── kv_quant_sparse_flash_attention_pioneer_infershape.cpp  # InferShape：输出 shape 推导
│   ├── kv_quant_sparse_flash_attention_pioneer_tiling.h    # Tiling 头文件：TilingData、Parser、Check、Tiling 类
│   └── kv_quant_sparse_flash_attention_pioneer_tiling.cpp  # Tiling 实现：切分策略、workspace 计算
│
└── op_kernel/                                              # ═══ Kernel 侧（NPU 端）═══
    ├── kv_quant_sparse_flash_attention_pioneer.cpp          # Kernel 入口：__global__ 函数
    ├── kv_quant_sparse_flash_attention_pioneer_kernel_mla.h # 主 Kernel 类：Init + 主循环 + 分核
    ├── kv_quant_sparse_flash_attention_pioneer_common.h     # 公共常量、枚举、对齐工具、模板宏
    ├── kv_quant_sparse_flash_attention_pioneer_service_cube_mla.h   # Cube 服务：AIC 核 BMM1/BMM2
    ├── kv_quant_sparse_flash_attention_pioneer_service_vector_mla.h # Vector 服务：AIV 核 Vec0/Vec1/Vec2
    ├── kv_quant_sparse_flash_attention_pioneer_kvcache.h    # KV Cache 参数计算、循环控制
    ├── kv_quant_sparse_flash_attention_pioneer_template_tiling_key.h # 模板参数组合声明
    ├── util_regbase.h                                      # 运行时参数结构体：RunInfo、ConstInfo、CVSharedParams
    └── vf/                                                 # 微核函数（VF）
        ├── vf_mul_sel_softmaxflashv2_cast_nz_qsfa.h       # Softmax + Cast 路径
        ├── vf_flashupdate_new_qsfa.h                       # Flash Update（累加状态更新）
        ├── vf_basic_block_utils.h                          # 基本块工具函数
        ├── vf_basic_block_aligned128_update_qsfa.h         # 对齐128 + update 路径
        ├── vf_basic_block_aligned128_no_update_qsfa.h      # 对齐128 + 首次（无update）路径
        ├── vf_basic_block_unaligned64_update_qsfa.h        # 非对齐64 + update 路径
        ├── vf_basic_block_unaligned64_no_update_qsfa.h     # 非对齐64 + 首次路径
        ├── vf_basic_block_unaligned128_update_qsfa.h       # 非对齐128 + update 路径
        └── vf_basic_block_unaligned128_no_update_qsfa.h    # 非对齐128 + 首次路径
```

---

## 3. 输入/输出/属性规格

### 3.1 输入张量
| 序号 | 名称 | 必选 | 数据类型 | Shape | 说明 |
|------|------|------|---------|-------|------|
| 0 | query | 是 | BF16 | BSND:[B,S1,N1,576] / TND:[T,N1,576] | Q=q_nope(512)+q_rope(64) |
| 1 | key | 是 | FP8_E4M3FN / HiFloat8 | PA:[Bn,Bs,1,656] / BSND:[B,S2,1,656] / TND:[T,1,656] | K=nope(512)+rope(128)+scale(16) |
| 2 | value | 是 | FP8_E4M3FN / HiFloat8 | 同 key | V 与 K 共享同一份数据 |
| 3 | sparse_indices | 是 | INT32 | BSND:[B,S1,1,K] / TND:[T,1,K] | K=sparse_block_count，离散取 KV 的索引 |
| 4 | key_dequant_scale | 否 | FLOAT | - | 预留参数，当前未使用 |
| 5 | value_dequant_scale | 否 | FLOAT | - | 预留参数，当前未使用 |
| 6 | block_table | 否 | INT32 | [B, max_block_num] | PA 场景的 block 映射表 |
| 7 | actual_seq_lengths_query | 否 | INT32 | [B] | 各 batch 的 Q 有效 token 数（TND 为前缀和） |
| 8 | actual_seq_lengths_kv | 否 | INT32 | [B] | 各 batch 的 KV 有效 token 数 |
| 9 | key_sink | 否 | BF16 | - | **预留输入**，Param Sink 的 Key 数据（当前传 nullptr） |
| 10 | value_sink | 否 | BF16 | - | **预留输入**，Param Sink 的 Value 数据（当前传 nullptr） |

### 3.2 输出张量
| 名称 | 数据类型 | Shape | 说明 |
|------|---------|-------|------|
| attention_out | BF16 | BSND:[B,S1,N1,512] / TND:[T,N1,512] | D=query_D - rope_head_dim |

### 3.3 关键属性
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| scale_value | float | 1.0 | softmax 缩放系数，= 1/√d_k |
| key_quant_mode | int64 | 1 | 量化模式，仅支持 2（per_tile） |
| value_quant_mode | int64 | 1 | 量化模式，仅支持 2（per_tile） |
| sparse_block_size | int64 | 1 | sparse 阶段 block 大小，[1,16] 且为 2 的幂 |
| layout_query | string | "BSND" | Q 的布局，BSND / TND |
| layout_kv | string | "BSND" | KV 的布局，BSND / TND / PA_BSND |
| sparse_mode | int64 | 3 | 0=全算，3=rightDownCausal |
| pre_tokens | int64 | INT64_MAX | 左窗口大小 |
| next_tokens | int64 | INT64_MAX | 右窗口大小 |
| attention_mode | int64 | 0 | 仅支持 2（MLA-absorb） |
| quant_scale_repo_mode | int64 | 1 | 量化参数存放模式，1=combine |
| tile_size | int64 | 128 | per_tile 块大小 |
| rope_head_dim | int64 | 64 | MLA 架构下 rope head dim |

### 3.4 布局支持
| 布局 | 维度 | 适用张量 | 说明 |
|------|------|---------|------|
| BSND | 4D [B,S,N,D] | Q, K, V, Out | 标准 Batch 模式 |
| TND | 3D [T,N,D] | Q, K, V, Out | 累积序列模式，T=ΣS |
| PA_BSND | 4D [Bn,Bs,N,D] | K, V | PageAttention 模式 |

---

## 4. Host 侧逻辑

### 4.1 OpDef（kv_quant_sparse_flash_attention_pioneer_def.cpp）
- 注册 11 个输入（query/key/value/sparse_indices 为 REQUIRED，其余 OPTIONAL）
- 注册 1 个输出（attention_out）
- 注册 13 个属性
- 支持 2 组数据类型组合：
  - Q:BF16 + KV:FP8_E4M3FN + Out:BF16
  - Q:BF16 + KV:HiFloat8 + Out:BF16
- 仅配置 ascend950 SoC

### 4.2 InferShape（kv_quant_sparse_flash_attention_pioneer_infershape.cpp）
- 输出 shape = query shape，D 维度减去 rope_head_dim
  - BSND: `[B, S1, N1, D - rope_head_dim]` → `[B, S1, N1, 512]`
  - TND: `[T, N1, D - rope_head_dim]` → `[T, N1, 512]`
- 输出 DataType = query DataType

### 4.3 Tiling 策略

#### 4.3.1 Tiling 主流程
```
TilingKvQuantSparseFlashAttentionPioneer()
  ├── QSFAPInfoParser::Parse()         # 解析所有输入参数
  │   ├── CheckRequiredParaExistence()  # 校验必选参数存在
  │   ├── GetOpParaInfo()               # 获取所有输入/输出/属性
  │   ├── GetInOutDataType()            # 获取数据类型
  │   ├── Get*Size()                    # 获取各维度大小
  │   └── GenerateInfo()                # 汇总到 QSFATilingInfo
  │
  ├── QSFAPTilingCheck::Process()      # 参数校验
  │   ├── CheckParaExistence()          # 校验参数组合
  │   ├── CheckSinglePara()             # 校验单个参数
  │   ├── CheckKV()                     # 校验 KV 形状/类型
  │   └── CheckFeature()                # 校验特性组合
  │
  └── QSFAPMlaTiling::DoOpTiling()     # 执行 Tiling
      ├── GetPlatformInfo()             # 获取 AIC/AIV 核数
      ├── InitParams()                  # 初始化 perfMode、headDimAlign
      ├── Split()                       # 切分策略
      │   └── SplitBalanced()           # 均衡切分
      │       └── CalcInnerSize()       # 计算 S2 内切大小
      ├── FillTiling()                  # 填充 TilingData
      ├── CalcBlockDim()                # 计算 block dim
      ├── GetWorkspaceSize()            # 计算 workspace 大小
      └── GenTilingKey()                # 生成 TilingKey
```

#### 4.3.2 关键切分参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| mBaseSize_ | 128 | M 方向基本块大小（S1×G） |
| s1BaseSize | 64 | S1 方向基本块大小 |
| s2BaseSize | 128 | S2 方向基本块大小 |
| sInnerSize_ | 512 | S2 方向内切大小 |
| perfMode_ | V_TEMPLATE_MODE | 性能模式 |

#### 4.3.3 TilingData 结构体
```
KvQuantSparseFlashAttentionPioneerTilingDataMla
  ├── baseParams (KvQuantSparseFlashAttentionPioneerBaseParamsMla)
  │   ├── batchSize, seqSize, qSeqSize           # B, S2, S1
  │   ├── blockSize, maxBlockNumPerBatch          # PA 参数
  │   ├── scaleValue                              # softmax 缩放
  │   ├── nNumOfQInOneGroup                       # GQA 分组数 (N1/N2)
  │   ├── actualLenDimsQ, actualLenDimsKV         # actual seq len 维度
  │   ├── outputLayout, sparseMode                # 布局和稀疏模式
  │   ├── sparseBlockSize, sparseBlockCount       # 稀疏参数
  │   └── dSizeVInput                             # KV 输入 D 大小 (656)
  │
  ├── splitKVParams (KvQuantSparseFlashAttentionPioneerSplitKVParamsMla)
  │   ├── s2                                      # S2 切分份数
  │   ├── accumOutSize                            # Flash Decode workspace
  │   └── logSumExpSize                           # Flash Decode workspace
  │
  ├── singleCoreParams (KvQuantSparseFlashAttentionPioneerSingleCoreParamsMla)
  │   └── usedCoreNum                             # 实际使用核数
  │
  ├── singleCoreTensorSize (KvQuantSparseFlashAttentionPioneerSingleCoreTensorSizeMla)
  │   ├── mmResUbSize                             # BMM1 结果 UB 大小
  │   └── bmm2ResUbSize                           # BMM2 结果 UB 大小
  │
  └── innerSplitParams (KvQuantSparseFlashAttentionPioneerInnerSplitParams)
      ├── mBaseSize                               # S1G 方向基本块
      └── s2BaseSize                              # S2 方向基本块
```

#### 4.3.4 Workspace 组成
Workspace 包含多个部分：
- libapi 基础空间
- BMM1 结果缓存：`preLoadNum × mmResUbSize × coreNum × 4B(fp32)`
- Vec1 结果缓存：`preLoadNum × mmResUbSize × coreNum × 2B(bf16)`
- BMM2 结果缓存：`preLoadNum × bmm2ResUbSize × coreNum × 4B(fp32)`
- Softmax 状态缓存：`preLoadNum × mBaseSize × coreNum × 4B`
- Vec2 结果缓存：`preLoadNum × bmm2ResUbSize × coreNum × 4B`
- 离散聚合缓存：`4 × 512 × 576 × 2 × coreNum` bytes
- MTE2 size 缓存：`4 × 128 × 4 × 2 × coreNum` bytes

#### 4.3.5 参数校验
- 数据类型：Q 仅支持 BF16/FP16，KV 仅支持 INT8/FP8_E4M3FN/HiFloat8
- 布局：Q 支持 BSND/TND，KV 支持 BSND/TND/PA_BSND
- sparse_mode：仅支持 0（全算）和 3（rightDownCausal）
- sparse_block_size：[1,16]，必须为 2 的幂
- attention_mode：仅支持 2（MLA-absorb）
- quant_scale_repo_mode：仅支持 1（combine）
- 非 PA 场景：Q 和 KV 布局必须一致

---

## 5. Kernel 侧逻辑

### 5.1 整体架构：AIC/AIV 异构协同

```
┌───────────────────────────────────────────────────────┐
│                    NPU AI Core                         │
│                                                       │
│  ┌──────────────┐    SSBuf     ┌──────────────┐      │
│  │  AIC (Cube)  │◄───────────►│ AIV (Vector)  │      │
│  │  BMM1/BMM2   │  CVSharedParams │ Vec0/Vec1/Vec2│  │
│  └──────┬───────┘              └──────┬───────┘      │
│         │ L0A/L0B/L0C                 │ UB           │
│         ▼                             ▼              │
│  ┌────────────────────────────────────────────┐      │
│  │              L1 Buffer (512KB)              │      │
│  │  ┌─────────┐ ┌─────────┐ ┌───────────────┐│      │
│  │  │  Q(L1)  │ │  K(L1)  │ │ KV_NZ(L1)     ││      │
│  │  │ SingleBuf│ │ 3-Buf   │ │ 3-Buf(CrossCore)│    │
│  │  └─────────┘ └─────────┘ └───────────────┘│      │
│  └────────────────────────────────────────────┘      │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐      │
│  │          Global Memory (GM/HBM)            │      │
│  └────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────┘

数据流：
  AIV: GM(KV_FP8) →[Vec0]→ UB(反量化) →[DataCopy]→ L1(NZ) ──┐
  AIC: GM(Q_BF16) →[DataCopy]→ L1(Q) ─────────────────────────┤
       L1(Q,KV) →[BMM1]→ L0C → UB(scores) ←───────────────────┤
  AIV: UB(scores) →[Vec1/Softmax]→ L1(P_NZ) ─────────────────┤
  AIC: L1(P,V) →[BMM2]→ L0C → UB(result) ←───────────────────┘
  AIV: UB(result) →[Vec2/FlashUpdate]→ UB → GM(attentionOut)
```

### 5.2 Kernel 入口（kv_quant_sparse_flash_attention_pioneer.cpp）

**函数签名**：
```cpp
template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void kv_quant_sparse_flash_attention_pioneer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *key_sink, __gm__ uint8_t *value_sink,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
```

**模板参数**：
| 参数 | 含义 | 取值 |
|------|------|------|
| FLASH_DECODE | 是否 Flash Decode | 0/1 |
| LAYOUT_T | Q 布局 | 0(BSND)/1(TND) |
| KV_LAYOUT_T | KV 布局 | 0(BSND)/1(TND)/2(PA_BSND) |
| TEMPLATE_MODE | 性能模式 | 0(C)/1(V) |

**数据类型组合**：
- BF16 + FP8_E4M3FN → `<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t>`
- BF16 + HiFloat8 → `<bfloat16_t, hifloat8_t, float, bfloat16_t>`

**AIC/AIV 实例化**：通过 `QSFA_OP_IMPL` 宏，AIC 侧实例化 `QSFAMatmulService` + `QSFAVectorServiceDummy`，AIV 侧实例化 `QSFAMatmulServiceDummy` + `QSFAVectorService`。

**注意**：`key_sink` 和 `value_sink` 参数已在入口函数签名中预留，但当前 `QSFA_OP_IMPL` 宏传递 `nullptr` 给 Init 函数。

### 5.3 主 Kernel 类（kv_quant_sparse_flash_attention_pioneer_kernel_mla.h）

#### 5.3.1 内存层级与 Buffer 布局

| 内存层级 | Buffer | 类型 | 大小 | 用途 |
|----------|--------|------|------|------|
| **GM** | gmBufferManager | - | - | 全局内存管理 |
| **L1** | l1BufferManager | BufferManager | 512KB | L1 总管理器 |
| **L1** | l1RightBuffers | 3-Buf, CrossCore | s2Base×576×sizeof(Q_T) | BMM1 右矩阵 KV（核间共享） |
| **UB** | ubBufferManager | BufferManager | mm1Res×2 + mm2Res | UB 总管理器 |
| **UB** | bmm1Buffers | DB, CrossCore | s1Base/2 × s2Base × sizeof(T) | BMM1 结果（FP32） |
| **UB** | bmm2Buffers | Single, CrossCore | s1Base/2 × 512 × sizeof(T) | BMM2 结果（FP32） |

#### 5.3.2 初始化流程
```
Init()
  ├── [AIV] InitVecBlock()                  # 解析 TilingData → CVSharedParams
  │        └── InitCubeVecSharedParams()    # 填充 sharedParams 并写入 SSBuf
  │             └── CrossCoreSetFlag(15)    # 通知 AIC
  ├── [AIV] CleanOutput()                   # 初始化 attentionOut 为 0（needInit时）
  ├── InitMMResBuf()                        # 分配 L1/UB buffer
  ├── [AIC] InitCubeBlock()                 # 初始化 Cube 侧 L0A/L0B/L0C
  │        └── CrossCoreWaitFlag(15)        # 等待 AIV 共享参数
  │        └── 从 SSBuf 读取 CVSharedParams
  ├── ComputeConstexpr()                    # 计算各种维度乘积、偏移常量
  ├── InitGlobalBuffer()                    # 设置 GM 指针
  ├── InitCalcParamsEach()                  # 均衡分核计算
  └── InitLocalBuffer()                     # 初始化 AIV 侧 UB buffer
```

**分核策略（InitCalcParamsEach）**：
- 统计所有 batch 的总基本块数 `totalBaseNum = Σ(actualS1[b])`
- 均分到每个核：`avgBaseNum = ceil(totalBaseNum / coreNum)`
- 每个核获得 `[bN2Start, gS1Start]` 到 `[bN2End, gS1End]` 的范围

#### 5.3.3 主循环（ProcessMainLoop）

**三级流水编排**：

```
时间 →    T0          T1          T2          T3          ...
AIC:   BMM1[0]     BMM1[1]     BMM1[2]     BMM1[3]     ...
                    BMM2[0]     BMM2[1]     BMM2[2]     ...
AIV:   Vec0[0]     Vec0[1]     Vec0[2]     Vec0[3]     ...
                    Vec1[0]     Vec1[1]     Vec1[2]     ...
                                Vec2[0]     Vec2[1]     ...
```

**循环结构**：
```
for bnIdx in [bN2Start, bN2End):              # BN2 外层循环
  ComputeParamBatch()                          # 计算 batch 参数
  ComputeS1LoopInfo()                          # 计算 S1 循环范围
  for gS1Index in [gs1Start, gs1End+PRELOAD]:  # GS1 循环（含 2 轮 preload 排空）
    ComputeAxisIdxByBnAndGs1()                 # 计算轴索引
    ComputeParamS1()                           # 计算 S1 方向参数
    ComputeS2LoopInfo()                        # 计算 S2 循环范围
    for s2LoopCount in [0, s2LoopLimit]:       # S2 内层循环
      // 三级流水
      if (notLastTwoLoop):
        SetRunInfo(runInfo[taskId%3])           # 设置当前 task 参数
        [AIC] cubeBlock.IterateBmm1()           # Stage0-Cube: Q×K^T
        [AIV] vecBlock.ProcessVec0()            # Stage0-Vec: 搬运+反量化 KV→L1
      if (taskId > 0 && notLast):
        [AIV] vecBlock.ProcessVec1()            # Stage1-Vec: Softmax(scores)→P→L1
        [AIC] cubeBlock.IterateBmm2()           # Stage1-Cube: P×V
      if (taskId > 1):
        [AIV] vecBlock.ProcessVec2()            # Stage2-Vec: FlashUpdate + CopyOut
      ++taskId
```

**关键设计点**：
- 使用 `RunInfo[3]` 环形数组存储三级流水的参数，`taskId % 3` 索引
- PRELOAD_NUM=2：最后 2 个 gS1 循环不执行 Stage0（BMM1/Vec0），仅排空 Stage1 和 Stage2
- `multiCoreInnerIdx` 跟踪 GS1 循环次数，用于 Softmax 状态管理（pingpong）

### 5.4 Cube 服务（kv_quant_sparse_flash_attention_pioneer_service_cube_mla.h）

**核心常量**：
| 常量 | 值 | 说明 |
|------|-----|------|
| s1BaseSize | 64 | M 方向基本块 |
| s2BaseSize | 128 | N 方向基本块 |
| dBaseSize | 576 | K 方向总大小 |
| dBaseMatmulSize | 128 | K 方向 Matmul 切分大小 |

**Buffer 分配**：
| Buffer | 层级 | 类型 | 大小 | 用途 |
|--------|------|------|------|------|
| l1QBuffers | L1 | SingleBuffer | 64×576×2B | BMM1 左矩阵 Q，S2 循环内复用 |
| l1KBuffers | L1 | 3-Buf | 576×128×2B | BMM1 右矩阵 K |
| mmL0ABuffers | L0A | DB | 16KB | Matmul L0A |
| mmL0BBuffers | L0B | DB | 32KB | Matmul L0B |
| mmL0CBuffers | L0C | DB | 128KB | Matmul L0C |

**IterateBmm1（Q × K^T → scores）**：
1. S2 第一次循环：搬运 Q 到 L1（CopyToL1Nd2Nz），后续复用
2. WaitCrossCore 等待 AIV 的 KV 数据就绪
3. MatmulK：Q(M=mReal, K=dSize) × K^T(K=dSize, N=s2Real)，K 方向按 128 切分
4. Fixpipe：L0C → UB（RowMajor，dualDstCtl=1 双目标模式）
5. SetCrossCore 通知 AIV

**IterateBmm2（P × V → partial_output）**：
1. WaitCrossCore 等待 AIV 的 P 和 V 数据
2. MatmulN：P(M=mReal, K=s2Real) × V(K=s2Real, N=dSizeNope=512)，左矩阵 P 来自 l1RightBuffers 的 rope 偏移位置，右矩阵 V 来自同一 buffer 的 nope 位置
3. Fixpipe：L0C → UB（RowMajor）
4. SetCrossCore 通知 AIV

### 5.5 Vector 服务（kv_quant_sparse_flash_attention_pioneer_service_vector_mla.h）

**三阶段处理**：

**Vec0（ProcessVec0）—— KV 搬运与反量化**：
1. WaitCrossCore 等待 L1 buffer 可用
2. 设置 PA 参数（blockSize, maxBlockNumPerBatch）
3. 调用 ProcessSparseKv：
   - 按 subBlockIdx（0/1）划分 S2 范围，双 subBlock 并行
   - 每次处理 16 行：
     a. GetRealCmpS2Idx：从 sparse_indices 获取真实 token 索引
     b. CopyInKvSparse：GM→UB，支持两行合并搬运（stride 合法时）或单行搬运
     c. DequantKv：FP8→BF16 反量化（VF 实现）
        - AntiquantVFFp8D448：nope 512 维度的反量化（per-tile-128 scale）
        - Copy rope 64 维度：直接从 BF16 拷贝
     d. CopyOutKvUb2L1：UB→L1，NZ 格式写出
4. SetCrossCore 通知 AIC

**Vec1（ProcessVec1）—— Softmax + Cast + P→L1**：
1. WaitCrossCore 等待 BMM1 结果
2. ProcessVec1Vf：根据 s2RealSize 分档（≤64, <128, =128）
   - 首次循环（s2LoopCount==0）：初始化 max/sum，执行 Softmax
   - 后续循环：Flash Softmax（online softmax），更新 max/sum/exp
3. Cast FP32→BF16，DataCopy 到 L1 的 P 区域
4. SetCrossCore 通知 AIC
5. 更新 ExpSum 和 ExpMax（非首次循环）

**Vec2（ProcessVec2）—— Flash Update + Output**：
1. WaitCrossCore 等待 BMM2 结果
2. 首次循环：直接 DataCopy
3. 中间循环：FlashUpdateNew（乘 exp 系数累加）
4. 最后循环：FlashUpdateLastNew（除以 sum 归一化）
5. 最后循环且仅一次 S2：LastDivNew
6. CopyOutAttentionOut：Cast FP32→BF16，DataCopy 到 GM

### 5.6 微核函数（vf/ 目录）

VF 路径选择基于 s2RealSize 的对齐情况：
| s2RealSize | 路径 | 文件 |
|------------|------|------|
| = 128（对齐） | aligned128 | vf_basic_block_aligned128_*.h |
| (64, 128) | unaligned128 | vf_basic_block_unaligned128_*.h |
| ≤ 64 | unaligned64 | vf_basic_block_unaligned64_*.h |

每个路径又分 update（非首次）和 no_update（首次）两种。

核心 VF 函数：
- `ProcessVec1Vf`：Muls(scale) → SelSoftmax → Cast(FP32→BF16) → NZ排布
- `FlashUpdateNew`：`O = O * exp + bmm2Res`
- `FlashUpdateLastNew`：`O = (O * exp + bmm2Res) / sum`
- `LastDivNew`：`O = O / sum`（仅 s2LoopCount==0 且最后一次）

### 5.7 KV Cache 参数计算（kv_quant_sparse_flash_attention_pioneer_kvcache.h）

关键函数：
- `ComputeParamBatch`：计算当前 batch 的 actualS1/S2、preTokens/nextTokens、queryOffset
- `ComputeS1LoopInfo`：计算 GS1 循环范围 [gs1LoopStart, gs1LoopEnd]
- `ComputeParamS1`：计算 S1 方向参数（s1RealSize, mRealSize, halfMRealSize 等）
- `ComputeS2LoopInfo`：计算 S2 循环范围，考虑 sparse 裁剪
- `LoopSOuterOffsetInit`：计算 attentionOutOffset 和 tensorQOffset
- `InitTaskParamByRun`：将 RunParam 参数拷贝到 RunInfo

稀疏索引映射：
- `GetRealCmpS2Idx`：从 sparse_indices GM 读取真实 token 索引
- `GetkeyOffset`：根据 PA/非PA 计算 KV 在 GM 中的实际偏移

---

## 6. 运行时参数结构体（util_regbase.h）

### 6.1 RunParamStr（分核与切块参数）
| 字段 | 说明 |
|------|------|
| boIdx, s1oIdx, n2oIdx, goIdx | 各轴的循环索引 |
| s2LoopEndIdx | S2 方向循环终止位置 |
| s2LineStartIdx, s2LineEndIdx | S2 按行的起止位置 |
| s1RealSize, halfS1RealSize | S1 方向真实大小和半分大小 |
| mRealSize, halfMRealSize | M 方向（S1×G）真实大小和半分大小 |
| attentionOutOffset | 输出偏移 |
| actualS1Size, actualS2Size | 当前 batch 的有效序列长度 |
| preTokensPerBatch, nextTokensPerBatch | 窗口参数 |
| qSNumInOneBlock | 每个基本块处理的 Q 行数 |
| oriKvLoopEndIdx | 原始 KV 循环终止 |

### 6.2 RunInfo（运行时信息）
在 RunParamStr 基础上增加：
| 字段 | 说明 |
|------|------|
| s2StartIdx, s2EndIdx | 当前 S2 的起止位置 |
| s2LoopCount, s2LoopLimit | 当前循环计数和上限 |
| taskId, taskIdMod2, taskIdMod3 | 流水线任务 ID |
| multiCoreInnerIdx, multiCoreIdxMod2/3 | GS1 循环计数 |
| s2RealSize, s2AlignedSize | S2 方向真实/对齐大小 |
| vec2S1BaseSize, vec2MBaseSize | Vec2 阶段的基本块大小 |

### 6.3 ConstInfo（初始化后不变的常量）
| 字段组 | 说明 |
|--------|------|
| bSize, s1Size, s2Size, gSize, n2Size | 各轴大小 |
| dSize(576), dSizeV(512), dSizeRope(64), dSizeNope(448→512) | D 方向维度 |
| s1BaseSize(64), s2BaseSize(128) | 基本块大小 |
| 各种轴乘积（s1Dv, gS1Dv, n2GDv 等） | 预计算的偏移乘积 |
| mm1Ka | BMM1 的 K 方向参数 |
| attentionOutStride | 输出步长 |
| 分核范围（bN2Start/End, gS1Start/End, s2Start/End） | 当前核的工作范围 |
| sparseBlockCount, sparseBlockSize | 稀疏参数 |
| oriBlockSize, cmpBlockSize, oriMaxBlockNumPerBatch | PA 参数 |
| softmaxScale | Softmax 缩放系数 |

### 6.4 CVSharedParams（核间共享参数）
通过 SSBuf 从 AIV 传递到 AIC 的共享参数，使用位域压缩：
| 字段 | 位宽 | 说明 |
|------|------|------|
| s1BaseSize, s2BaseSize | 32b each | 基本块大小 |
| bSize, n2Size, gSize, s1Size, s2Size | 32b each | 轴大小 |
| dSize | 10b | D 方向大小 |
| dSizeVInput | 12b | KV 输入 D 大小 |
| needInit | 4b | 是否需要初始化输出 |
| layoutType | 4b | 布局类型 |
| isActualSeqLengthsNull/KVNull | 1b each | actual seq len 标志 |
| sparseBlockCount | 32b | 稀疏块数 |
| softmaxScale | 32b(float) | Softmax 缩放 |
| cmpRatio, dSizeRope | 9b, 11b | 压缩比、rope D |
| oriWinLeft, oriWinRight | 32b each | 窗口参数 |
| tileSize, oriBlockSize, cmpBlockSize | 8b, 12b, 12b | tile/block 大小 |
| oriMaxBlockNumPerBatch, cmpMaxBlockNumPerBatch | 32b each | PA 参数 |
| usedCoreNum | 32b | 实际使用核数 |

---

## 7. 端到端数据流

```
         Host (CPU)                         NPU Kernel
    ┌──────────────────┐          ┌──────────────────────────────┐
    │ OpDef 校验        │          │                              │
    │ InferShape 推导   │          │ ┌──── Stage0 (Vec0+BMM1) ───┐│
    │   out.D = Q.D-64 │          │ │ AIV: GM(KV_FP8) → UB      ││
    │                  │          │ │      反量化 → NZ → L1      ││
    │ Tiling 切分       │          │ │ AIC: GM(Q_BF16) → L1      ││
    │   均衡分核        │ ───────► │ │      L1(Q,K) → BMM1 → UB  ││
    │   计算workspace  │          │ └────────────────────────────┘│
    │                  │          │ ┌──── Stage1 (Vec1+BMM2) ───┐│
    │ 下发 TilingData   │          │ │ AIV: UB(scores) → Softmax ││
    │   via SSBuf      │          │ │      → Cast → P_NZ → L1   ││
    └──────────────────┘          │ │ AIC: L1(P,V) → BMM2 → UB  ││
                                  │ └────────────────────────────┘│
                                  │ ┌──── Stage2 (Vec2) ────────┐│
                                  │ │ AIV: UB(bmm2Res)          ││
                                  │ │   → FlashUpdate(累加)      ││
                                  │ │   → Cast → GM(attenOut)   ││
                                  │ └────────────────────────────┘│
                                  └──────────────────────────────┘
```

---

## 8. 关键设计总结

| 设计要点 | 实现方式 |
|---------|---------|
| 稀疏计算 | sparse_indices 索引离散取 KV，每行独立查 topK 块 |
| FP8 反量化 | AIV 侧 VF 微核（Per-Tile-128 scale × FP8 → BF16） |
| 三级流水 | Vec0+BMM1 / Vec1+BMM2 / Vec2 交错执行，RunInfo[3] 环形 |
| AIC/AIV 核间通信 | SSBuf 传递 CVSharedParams；L1 CrossCore buffer 共享 KV/P |
| 在线 Softmax | Flash Attention 范式：逐块更新 max/sum/exp |
| 均衡分核 | 按 totalBaseNum 均分到各核，支持变长 actual seq len |
| PA 支持 | block_table 映射 + blockSize 切分 |
| D 方向 MLA | Q:nope(512)+rope(64)=576, KV:nope(512)+rope(128)+scale(16)=656 |
| key_sink/value_sink 预留 | OpDef 已注册（index 9/10），入口函数已有参数，当前传 nullptr |
