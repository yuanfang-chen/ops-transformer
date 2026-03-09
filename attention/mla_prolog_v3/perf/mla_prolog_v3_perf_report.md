# MlaPrologV3 理论性能分析报告

## 1. Kernel 分解与分析边界

本报告基于 `kernel_mla_prolog_split_n.h` 与 `mla_prolog_tiling.cpp` 的当前实现，按真实的 CV 同步关系拆成 `copy_global_params`、`copy_sincos`、`mm1_cq`、`rmsnorm_cq`、`mm2_ckvkr`、`rmsnorm_ckv_scatter`、`rope_kr_scatter`、`mm3_qcqr`、`dequant_or_cast_qc`、`rope_qr`、`mm4_qn`、`dynamic_quant_qn_mul_qr` 等阶段。

分析假设:
- 外部输入、最终输出与 cache scatter 走 GM 带宽模型。
- kernel 内部 workspace 与 weight 复用优先按 L2 带宽模型估算。
- `enableDequantOpt` 场景下，`mm3_qcqr -> dequant_or_cast_qc -> mm4_qn` 按 split-N producer/consumer pipeline 估算暴露时长。
- 这是理论模型，不包含 runtime profile、DMA 冲突、bank conflict 与指令级泡沫的实测修正。

## 2. Ascend950 硬件模型

- SoC: `Ascend950`, AIC=32, AIV=64, CV 模式=`1:2`
- Cube 峰值: BF16=432.538 TFLOPS, INT8=865.075 TOPS, FP8=865.075 TOPS
- Vector 峰值: FP32=432.538 TFLOPS, BF16=865.075 TFLOPS, INT8=1730.150 TOPS, FP8=1730.150 TOPS
- 带宽: GM=1.600 TB/s, L2=5.200 TB/s
- 本地存储: UB=248.0 KB, L1=512.0 KB, L0A/L0B/L0C=64.0/64.0/256.0 KB

需要特别指出的当前 kernel 约束:
- `stepBatchSize` 仍然被 host 固定为 `min(128, T)`。
- vector path 仍然使用 `MAX_UB_SIZE = 192KB` 的静态上限，而不是 Ascend950 的 248KB。
- cube path 仍然按 `L1_A_SIZE = 128KB`, `L1_B_SIZE = 128KB`, `L0A/L0B/L0C = 32/32/64KB` 的旧静态常量组织双 buffer。

## 3. 当前 Host Tiling 特征

以 canonical 形状 `He=7168, Hcq=1536, Hckv=512, D=128, Dr=64, N=128` 为例，Ascend950 当前 host 规则得到:
- `stepBatchSize = 128`, `vectorBlockNum = 64`
- `mm1BlockNum = 24`，说明 `MatmulCq` 只激活 24 个 cube block
- `mm2BlockNum = 9`，说明 `MatmulCkvKr` 只激活 9 个 cube block
- `mm3BlockNum = 32`, `mm4BlockNum = 32`，`MatmulQcQr` 与 `MatmulQn` 才能基本吃满 32 个 AIC

这意味着当前 tiling 在 Ascend950 上的主要问题不是 `mm3/mm4`，而是前两段 cube matmul 天然欠并行。

## 4. 分场景瓶颈结论

- `bf16_small`: steady-step `22.065 us`, critical side `cube_side`, top stage `mm3_qcqr` (l2_bandwidth, 14.529 us)
- `bf16_large`: steady-step `45.683 us`, critical side `cube_side`, top stage `mm3_qcqr` (cube_compute, 22.342 us)
- `int8_tensor`: steady-step `26.588 us`, critical side `cube_side`, top stage `mm3_qcqr` (cube_compute, 11.171 us)
- `fp8_tensor`: steady-step `26.586 us`, critical side `cube_side`, top stage `mm3_qcqr` (cube_compute, 11.171 us)
- `pa_blk_stress`: steady-step `24.624 us`, critical side `cube_side`, top stage `mm3_qcqr` (cube_compute, 11.171 us)

从模型结果归纳:
- BF16 / 非量化路径: 主瓶颈仍落在 cube 侧，核心压力集中在 `mm1_cq` 与 `mm3_qcqr`。其中 `mm1_cq` 经常先撞到 L2 供数，而不是 vector 侧。
- INT8 全量化 + per-tensor kv + query quant: 在当前 canonical 大形状上仍是 cube-side bound，但 `dequant_or_cast_qc`、`rope_qr`、`dynamic_quant_qn_mul_qr` 与 cache/update 使 vector 暴露时长明显上升，最容易先把瓶颈推向 vector + memory。
- MXFP8-like 路径: 当前 canonical 大形状下仍由 `mm3_qcqr` 主导，但 `float` 累加输出、split-N dequant/cast、rope 和最终 query post-process 使后半段的 vector/L2 压力更突出。
- PA/PA_BLK cache 模式: scatter efficiency 降低后，GM/L2 在 `rmsnorm_ckv_scatter` 与 `rope_kr_scatter` 上更容易主导总时长，尤其是 `PA_BLK_*`。

### Regime Map

- BF16 canonical path: 预计为 cube-side bound，热点集中在 `MatmulCq` 与 `MatmulQcQr`。
- INT8 / FP8 full-quant path: 在当前 canonical `T=144` case 上仍是 cube-side bound，但最脆弱的部分已经转到 `dequant/cast + rope + scatter + dynamic quant`，cache 更差或 shape 更偏 vector 时更容易变成 vector + memory bound。
- PA-style cache writes: 最终瓶颈取决于 scatter memory efficiency，可能是 vector 算子本身，也可能直接退化成 GM 带宽瓶颈。

## 5. `stepBatchSize = 128` 何时不优

当前 host 固定 `stepBatchSize = 128` 有两个问题:
- 当 `T` 远大于 128 且 path 已经被 vector/scatter 主导时，更大的 step 会放大 `rope_qr`、`dynamic_quant_qn_mul_qr` 与 cache scatter 的单步尾延迟。
- 当 `mm3` 允许 `baseN = 256` 的 BF16/INT8 非 FP8 场景，`stepBatchSize <= 64` 会触发更激进的 `mmQcQr.baseN`，有机会改善 `mm3` 的 L1/L0 利用率和 steady-step time。

模型上的结论不是“永远把 128 改小”，而是:
- cube-bound 且 scatter 不重时，128 仍然合理，因为它提升了 `mm1/mm3/mm4` 的 M 维利用率。
- vector/scatter-bound 时，128 往往开始拖累后半段，应该允许更小的 `stepBatchSize` 进入候选集。

## 6. 优化建议

- 优先改 `mm2SingleCoreN = 64` 的固定经验值。Ascend950 上这会把 `MatmulCkvKr` 长期锁死在 9 个 cube block，明显浪费 32 AIC。
- 重新审视 `mm1SingleCoreN >= 64` 的下界。对 `Hcq = 1536` 这类常见形状，它让 `MatmulCq` 只能启用 24 个 cube block。
- 给 host 增加可搜索的 `stepBatchSize` 与 `vectorBlockNum`，而不是把它们写死为 `min(128, T)` / `min(stepBatchSize, aiv_num)`。
- 允许 Ascend950 使用更大的 vector UB 预算。当前 `MAX_UB_SIZE = 192KB` 会直接吃掉新芯片的额外 56KB 空间。
- 让 `mmQcQr.baseN`、`mm1/mm2/mm3/mm4` split heuristic 与 Ascend950 分离配置，不再沿用旧 NPU 的经验常量。
- 对 PA / PA_BLK 路径单独做 memory-oriented tiling，重点降低 `rmsnorm_ckv_scatter` 与 `rope_kr_scatter` 的 GM 暴露时长。

## 7. Search 结果摘要

- `bf16_large_host_legal`: best objective `45.683 us`, `stepBatchSize=128`, `vectorBlockNum=64`, `mm1/mm2/mm3/mm4=64/64/768/4`
  no better host-legal change was found under current C++ tiling policy
- `bf16_large_exploratory`: best objective `22.790 us`, `stepBatchSize=16`, `vectorBlockNum=4`, `mm1/mm2/mm3/mm4=128/32/768/4`
  requires host/kernel changes: stepBatchSize host rule, vectorBlockNum host rule, stepNumHeadDequant host rule, mm1 single-core N heuristic, mm2 single-core N heuristic
- `int8_tensor_exploratory`: best objective `11.051 us`, `stepBatchSize=16`, `vectorBlockNum=8`, `mm1/mm2/mm3/mm4=128/32/3072/16`
  requires host/kernel changes: stepBatchSize host rule, vectorBlockNum host rule, stepNumHeadDequant host rule, mm1 single-core N heuristic, mm2 single-core N heuristic, mm3 single-core N heuristic, mm4 head split heuristic

## 8. BF16 大形状稳态阶段表

| Stage | Engine | Scope | Bound | Ops | GM Bytes | L2 Bytes | Duration (us) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| copy_sincos | dma | per_step | gm_bandwidth | 0.00 | 18,432.00 | 18,432.00 | 0.012 |
| mm1_cq | cube | per_step | l2_bandwidth | 2,818,572,288.00 | 2,228,224.00 | 66,060,288.00 | 12.704 |
| rmsnorm_cq | vector | per_step | l2_bandwidth | 2,359,296.00 | 0.00 | 789,504.00 | 0.152 |
| mm2_ckvkr | cube | per_step | cube_compute | 1,056,964,608.00 | 1,835,008.00 | 24,772,608.00 | 8.688 |
| rmsnorm_ckv_scatter | vector | per_step | gm_bandwidth | 851,968.00 | 131,072.00 | 131,072.00 | 0.082 |
| rope_kr_scatter | vector | per_step | gm_bandwidth | 73,728.00 | 16,384.00 | 16,384.00 | 0.010 |
| mm3_qcqr | cube | per_step | cube_compute | 9,663,676,416.00 | 0.00 | 82,182,144.00 | 22.342 |
| dequant_or_cast_qc | vector | per_step | vector_compute | 0.00 | 0.00 | 0.00 | 0.000 |
| rope_qr | vector | per_step | gm_bandwidth | 8,388,608.00 | 2,097,152.00 | 2,097,152.00 | 1.311 |
| mm4_qn | cube | per_step | gm_bandwidth | 2,147,483,648.00 | 16,777,216.00 | 20,971,520.00 | 10.486 |
| dynamic_quant_qn_mul_qr | vector | per_step | vector_compute | 0.00 | 0.00 | 0.00 | 0.000 |

## 9. INT8 全量化稳态阶段表

| Stage | Engine | Scope | Bound | Ops | GM Bytes | L2 Bytes | Duration (us) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| copy_sincos | dma | per_step | gm_bandwidth | 0.00 | 18,432.00 | 18,432.00 | 0.012 |
| mm1_cq | cube | per_step | l2_bandwidth | 2,818,572,288.00 | 1,703,936.00 | 33,030,144.00 | 6.352 |
| rmsnorm_cq | vector | per_step | l2_bandwidth | 4,718,592.00 | 0.00 | 986,112.00 | 0.190 |
| mm2_ckvkr | cube | per_step | cube_compute | 1,056,964,608.00 | 917,504.00 | 12,386,304.00 | 4.344 |
| rmsnorm_ckv_scatter | vector | per_step | l2_bandwidth | 1,638,400.00 | 65,536.00 | 262,144.00 | 0.050 |
| rope_kr_scatter | vector | per_step | gm_bandwidth | 73,728.00 | 16,384.00 | 32,768.00 | 0.010 |
| mm3_qcqr | cube | per_step | cube_compute | 9,663,676,416.00 | 0.00 | 50,528,256.00 | 11.171 |
| dequant_or_cast_qc | vector | per_step | l2_bandwidth | 4,194,304.00 | 0.00 | 12,648,448.00 | 2.432 |
| rope_qr | vector | per_step | gm_bandwidth | 8,388,608.00 | 2,097,152.00 | 4,194,304.00 | 1.311 |
| mm4_qn | cube | per_step | l2_bandwidth | 2,147,483,648.00 | 0.00 | 37,748,736.00 | 7.259 |
| dynamic_quant_qn_mul_qr | vector | per_step | gm_bandwidth | 109,051,904.00 | 10,551,296.00 | 18,874,368.00 | 6.595 |
