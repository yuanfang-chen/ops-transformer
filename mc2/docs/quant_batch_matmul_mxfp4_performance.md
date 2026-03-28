# MXFP4 Quant BatchMatMul 算子性能优化指南

## 概述

本文档在《MXFP4 量化矩阵乘算子性能优化指南》（参考：`e:\算子代码\quant_matmul_mxfp4_performance.md`）的基础上，针对 **Quant BatchMatMul（量化批矩阵乘）** 补充 **batch 维、MC2 融合实现与调优要点**。算子级数据搬运、MMAD、性能建模公式与 **单矩阵乘 MXFP4** 一致；差异主要体现在 **多 batch 的 tiling、负载均衡与通信–计算融合**。

**图示说明**：下文 **嵌入式 SVG** 与参考文档中 `figures/image23.png`～`image34.png` **语义一一对应**。若你本地已有原版 PNG，可将同名文件放入本目录 `figures/` 替换或并存；当前仓库提供 `quant_bmm_figure*.svg` 便于审阅与版本管理。

| 参考文档插图 | 本文档插图（SVG） |
|-------------|------------------|
| image23 数据流 | `figures/quant_bmm_figure1_dataflow.svg` |
| image28 SWAT | `figures/quant_bmm_figure2_swat_zorder.svg` |
| image29 L1 Bank | `figures/quant_bmm_figure5_l1_bank.svg` |
| image30 Scale 缓存 | `figures/quant_bmm_figure6_scale_cache.svg` |
| image31 全载 | `figures/quant_bmm_figure8_full_load.svg` |
| image32 尾轮均衡 | `figures/quant_bmm_figure7_balance.svg` |
| image33 Double Buffer | `figures/quant_bmm_figure3_double_buffer.svg` |
| image34 UnitFlag | `figures/quant_bmm_figure4_unitflag.svg` |

---

## 算子实现原理

### 算子功能说明

- **算子功能**：在 **batch 维**上执行多组 MXFP4 矩阵乘。MX 量化等价于 **GroupSize=32**、Scale 为 **float8_e8m0** 的 per-group 量化；数学形式与单 MatMul 相同，仅在 **A/B 的 batch 维**上重复或向量化调度。

- **应用场景**：LLM 推理中 **按 batch 的线性层 / 融合通信算子**（如 AllGather + Quant BMM），在保持精度的同时提升吞吐。

- **计算公式**（单 batch 切片与参考文档一致；batch 维独立求和或并行）：

$$
c_{b,i,j} = \sum^{K/G-1}_{g=0}\left(scaleA_{b,g,i} \cdot scaleB_{b,g,j} \cdot \sum^{G-1}_{k'=0} (a_{b,i,gG+k'} \cdot b_{b,gG + k', j}) \right)
$$

- **参数说明（Quant BMM 典型形状）**：

| **变量名** | **描述** | **Dtype** | **Layout** | **Shape（示例）** |
|------------|----------|-----------|------------|-------------------|
| a | 左矩阵 | `float4_e2m1` 等 | ND | `(b, m, k)` |
| b | 右矩阵 | `float4_e2m1` 等 | ND | `(b, n, k)` |
| scaleA | 左 scale | `float8_e8m0` | ND | `(b, m, ceil(k/64), 2)` 等（以 ACL 为准） |
| scaleB | 右 scale | `float8_e8m0` | ND | `(b, n, ceil(k/64), 2)` 等 |
| c | 输出 | `float32` / `float16` / `bfloat16` | ND | `(b, m, n)` |

### 算子实现说明

与单 MatMul MXFP4 相同：**scaleA / scaleB** 进入 **L0A_MX / L0B_MX**，**MMAD** 自动完成 MXFP4 乘加，**Fixpipe** 输出目标 dtype；**主路径仅依赖 CUBE 核**。Quant BMM 在实现上通常通过 **MatmulWithScale** 将 scale 与矩阵乘绑定在同一 Cube 流水，避免先反量化再乘。

**Batch 维处理**：在 `Mc2QuantBatchMatmulASWKernel` 中，当 `batchC == 1` 时走简化偏移；多 batch 时在 **GM 地址与 tiling 的 round** 上遍历各 batch 切片（见 `ComputeBasicOptiLoop`）。

<div align="center">
  <img src="figures/quant_bmm_figure1_dataflow.svg" width="820" alt="Quant BMM MXFP4 数据流" />
</div>

**Tensor 搬运与 Layout**（a、b、scaleA、scaleB）：与参考文档中 **GM→L1→L0**、**ND2NZ**、**Load2D / Load2D_MX** 的描述一致；仅 **Shape 最左维增加 batch**。下列插图沿用与 **image24～image27** 相同的逻辑，不再逐张贴图；需细节时请对照 `e:\算子代码\quant_matmul_mxfp4_performance.md` 原图。

### 算子实现约束

与参考文档一致，Quant BMM 额外建议：

1. **batch 维**：各 batch 内 **K 对齐、内轴偶数** 等约束与单 MatMul 相同；**不同 batch 形状一致** 时最易发挥多核均衡。
2. **融合算子**：AllGather + Quant BMM 等场景下，**通信侧输出排布**需与 MatMul 输入对齐，避免多余格式转换增加 MTE2。

---

## 算子性能建模

### 性能瓶颈分析

与单 MatMul 相同，分为 **Cube Bound** 与 **Memory Bound**（MTE2 / MTE1 / FIXPIPE）。Quant BMM 额外关注：

- **batch 放大 FIXPIPE**：输出 `(b, m, n)` 时，Fixpipe 搬运量随 **b** 线性增加，小 **K** 时更易出现 **FIXPIPE Bound**（与参考文档 FIXPIPE 分析一致）。
- **多核尾轮**：总 tile 数 = **batch ×** 单 batch 在 M×N 上的切分数；尾轮负载均衡需覆盖 **batch 维**（见下文插图）。

### 性能建模公式

**基本原理**（与参考文档一致）：

理论计算时间 = max(MMAD 时间, MTE2 时间, MTE1 时间, FIXPIPE 时间)

**MMAD、MTE2、MTE1、FIXPIPE** 的公式与参考文档 **§ 算子性能建模** 相同；将 **M、N** 理解为 **单 batch 切片**时，对 **每个 batch** 重复；总 MMAD 时间近似按 **batch 数** 线性扩展（实际以 tiling 融合 batch 与否为准）。

**Quant BMM 补充**：融合通信时，MTE2 搬运量需加上 **gather 输出** 与 **可选 scale 重复项**，建议用 Profiler 区分 **DDR / L2** 占比。

---

## 算子优化实践

本章与参考文档结构对齐：**搬运效率**、**计算效率**、**指令并行度**；下列插图与 **image28～image34** 对应。

### 搬运效率优化

#### SWAT（自适应滑动窗口模板）

- **原理**：通过 **M×N 面 Z 型滑动** 提高 L2 命中率，使 MMAD 不断流；Quant BMM 在 **每个 batch 的 M×N 面** 上重复相同策略，或由 tiling 将 **batch 与 M、N** 统一分块。

<div align="center">
  <img src="figures/quant_bmm_figure2_swat_zorder.svg" width="900" alt="SWAT Z 型顺序" />
</div>

- **适用**：大规模 batch、多核 Cube Bound 场景。

#### L1 Bank 冲突优化

<div align="center">
  <img src="figures/quant_bmm_figure5_l1_bank.svg" width="720" alt="L1 Bank" />
</div>

#### Scale 缓存优化

<div align="center">
  <img src="figures/quant_bmm_figure6_scale_cache.svg" width="760" alt="Scale 缓存" />
</div>

#### 全载优化

<div align="center">
  <img src="figures/quant_bmm_figure8_full_load.svg" width="760" alt="全载 L1" />
</div>

### 计算效率优化

#### 尾轮负载均衡

<div align="center">
  <img src="figures/quant_bmm_figure7_balance.svg" width="800" alt="尾轮负载均衡" />
</div>

**本仓库对应**：`Mc2QuantBatchMatmulASWKernel::ComputeBasicOptiLoop` 中对 `round`、`doTailTile_` 与 `usedCoreNum` 的处理，用于减少尾块空转。

### 指令并行度优化

#### Double Buffer

<div align="center">
  <img src="figures/quant_bmm_figure3_double_buffer.svg" width="820" alt="Double Buffer" />
</div>

#### UnitFlag

<div align="center">
  <img src="figures/quant_bmm_figure4_unitflag.svg" width="780" alt="UnitFlag" />
</div>

---

## 与本仓库实现的对应关系

| 优化点 | 参考文档 | 本仓库（示例） |
|--------|----------|----------------|
| Cube 内 MX scale | MMAD + L0_MX | `mc2/common/op_kernel/mc2_quant_batch_matmul.h`：`IsMxType`、`MatmulWithScalePolicy` |
| 分块与尾轮 | 尾轮负载均衡 | 同文件：`ComputeBasicOptiLoop`、`SetSingleShape` |
| 融合 MX 入口 | — | `mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_v2_apt.cpp`：`INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL` |

---

## 算子模板归纳

| **模板** | **特点** | **Quant BMM 说明** |
|----------|----------|-------------------|
| **SWAT** | 通用大形状，Cube Bound 友好 | 对每个 batch 的 M×N 调度；首选 |
| **FullLoad** | 小形状驻留 L1 | 小 batch Decode、MTE2 Bound |
| **A 全载** | 单侧全载 | 与参考文档 `quant_matmul_mxfp4_a_full_load` 思想一致 |

---

## 优化策略选择指南

| **Bound 类型** | **推荐策略** | **Quant BMM 备注** |
|----------------|-------------|-------------------|
| Cube Bound | SWAT + 尾轮均衡 | 注意 **batch×tile** 总数 |
| MTE2 Bound | 全载 + Scale 缓存 | 融合场景计入 **通信输出** |
| MTE1 Bound | L1 Bank 优化 | 与单 MatMul 相同 |
| FIXPIPE Bound | UnitFlag | **batch 大、K 小** 时优先排查 |
| 流水停顿 | Double Buffer | batch 与 K 轮次叠加时更有效 |

---

## 性能调优实践步骤

1. **Profiling**：判定 Cube / MTE2 / MTE1 / FIXPIPE Bound（含 batch 维）。
2. **瓶颈识别**：是否尾轮不均、是否小 K 导致 FIXPIPE。
3. **策略选择**：对照上表。
4. **模板选择**：大矩阵 SWAT，小矩阵 FullLoad。
5. **参数调优**：`baseM/baseN`、batch 合并、K 对齐。
6. **验证**：相对 BF16 MatMul / 非 MX 量化的吞吐与延迟。

---

## 总结

MXFP4 **Quant BatchMatMul** 的算子内核与 **MXFP4 单 MatMul** 共享同一套 **Cube + L0_MX + MMAD + Fixpipe** 路径；性能优化仍以 **SWAT、全载、Scale 缓存、Double Buffer、UnitFlag、尾轮均衡** 为主，并额外关注 **batch 维 tiling** 与 **融合通信下的 MTE2**。图示与 `e:\算子代码\quant_matmul_mxfp4_performance.md` 中插图一一对应，便于两边文档对照阅读。

---

## 附录：原始 PNG 使用方式

若已在 `e:\算子代码\figures\` 下保留 **image23.png～image34.png**，可复制到本目录：

`ops-transformer/mc2/docs/figures/`

并将本文中对应 `<img src="figures/quant_bmm_figure*.svg">` 替换为 `imageXX.png` 即可获得与 CANN 样例完全一致的像素级配图。
