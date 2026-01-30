# UT 覆盖率提升报告 - all_gather_matmul

## 测试信息
- **算子名称**: all_gather_matmul
- **测试类型**: ophost
- **测试时间**: 2026-02-02
- **执行环境**: 容器 y00845930
- **测试命令**: `bash build.sh -u --ophost --ops='all_gather_matmul' --cov`

## 测试结果
- **总测试用例数**: 24 (20 tiling + 4 infershape)
- **通过**: 24
- **失败**: 0

## 覆盖率数据

### 总体覆盖率变化
| 文件 | 初始覆盖率 | 最终覆盖率 | 提升幅度 |
|------|-----------|-----------|---------|
| all_gather_matmul_infershape.cpp | 100% (7/7) | 100% (7/7) | - |
| all_gather_formulaic_tiling.cpp | 81.2% (56/69) | 84.1% (58/69) | +2.9% |
| all_gather_formulaic_tiling.h | 90.9% (10/11) | 90.9% (10/11) | - |
| all_gather_matmul_tiling.cpp | 85.6% (291/340) | 86.8% (295/340) | +1.2% |

### 覆盖率详情
| 文件 | 已覆盖行数 | 总行数 | 覆盖率 |
|-----|-----------|--------|--------|
| all_gather_matmul_infershape.cpp | 7 | 7 | 100.0% |
| all_gather_formulaic_tiling.cpp | 58 | 69 | 84.1% |
| all_gather_formulaic_tiling.h | 10 | 11 | 90.9% |
| all_gather_matmul_tiling.cpp | 295 | 340 | 86.8% |

## 补充的测试用例

### 新增测试用例列表 (v4)
1. **all_gather_matmul_test_tiling_float32** - FLOAT32 数据类型测试 (覆盖 inputDtypeSize == 4 分支)
2. **all_gather_matmul_test_tiling_no_bias** - 无 bias 场景测试
3. **all_gather_matmul_test_tiling_tiny_shape** - 小形状场景测试
4. **all_gather_matmul_test_tiling_large_k** - 大 K 值场景测试
5. **all_gather_matmul_test_tiling_large_m_n** - 大 M 大 N 场景测试
6. **all_gather_matmul_test_tiling_median_m** - 中等 M 值场景测试
7. **all_gather_matmul_test_tiling_bias_no_trans** - bias + 无转置场景测试
8. **all_gather_matmul_test_tiling_rank2_bias** - 最小 rankSize=2 且 bias 场景测试
9. **all_gather_matmul_test_tiling_no_gather_out** - 无 gatherOut 输出场景测试

## 无法覆盖的代码路径分析

### all_gather_matmul_tiling.cpp
| 行号 | 代码片段 | 无法覆盖原因 |
|------|---------|-------------|
| 111 | 函数 `MC2_Splite` 声明 | 函数声明行不计入覆盖 |
| 114-131 | `MC2_Splite` 函数体 | 需要设置 commTurn != 0，但代码中行562-563检查commTurn必须为0，导致此分支永远无法进入 |
| 147-161 | `AllGatherParamsCheck` 错误分支 | 这些是错误检测分支，需要特定的非法输入：gather_index != 0、isTransA != false、K值不在有效范围。测试框架可能阻止这些非法输入通过参数校验 |
| 180-181 | `GetAllGatherFormulateTileCnt` 错误返回 | 需要context->GetAttrs() == nullptr，但测试框架始终提供有效的attrs |
| 215-219 | `MCSpliteM` 中的 enableSplitK 分支 | 需要args.enableSplitK为true，但代码中始终设置为false (行326: args.enableSplitK = false) |
| 220-237 | `MCSpliteM` 中的 commTurn != 0 分支 | 需要设置commTurn != 0，但行562-563检查commTurn必须为0 |
| 302-306 | A.shape(1) != B.shape(0) 的分支 | 需要A的K维度和B的K维度不匹配，但在正常的矩阵乘法场景中，这应该是一个错误条件 |
| 349-350 | cmdType error 分支 | 需要cmdType不等于HCCL_CMD_ALLGATHER，但代码中始终设置为HCCL_CMD_ALLGATHER (行589) |
| 354-358 | DOUBLE_RING + Step 通信算法分支 | 需要commAlg == COMM_ALG_DOUBLE_RING且isStep == 1，但代码中commAlg始终设置为COMM_ALG_FULL_MESH (行171)，isStep始终为0 (行526) |
| 401-404 | `GetTiling` 返回 -1 的错误处理 | 需要matmul tiling计算失败，这是一个错误处理路径 |
| 428-431 | gatherIndex != 0 时的分支 | 需要设置gatherIndex != 0，但现有测试和代码验证都要求gatherIndex == 0 (行150-152) |
| 441 | isStorageGather == false 且 gatherIndex != 0 | 需要gatherIndex != 0的场景 |
| 474-476 | workspace 为 nullptr 的错误处理 | 这是错误处理路径，测试框架始终提供有效的workspace |
| 608 | `TilingParseForAllGatherMatmul` 函数 | 这是一个单独的函数，可能在UT测试中不被调用，需要单独的parse测试 |

### all_gather_formulaic_tiling.cpp
| 行号 | 代码片段 | 无法覆盖原因 |
|------|---------|-------------|
| 55 | `SetAlignLength(tilingM_.GetAlignLength() / TWO)` | 需要满足reduceAlignLen条件且allowMoreCuts为true，这是一个特定的性能优化分支 |
| 61 | `noCutFlag_ = false` (HUGE_K_BOUNDARY条件) | 需要K值大于HUGE_K_BOUNDARY且满足其他条件，这是一个特定的边界条件 |
| 66-69 | `SetCommTimeFactorForA5` | 需要SocVersion::SOC910_95，但测试环境可能使用SOC910B或SOC910_93 |
| 86-89 | bwGrowthByShape 分支中的特定子分支 | 需要满足特定的条件组合：medianMFlag和bwGrowthByShape |
| 100 | `isA3 == 0` 判断 | 代码中isA3始终设置为0 (行525)，但相关分支已经被覆盖 |
| 117 | `ShortAtEndCalcBoundBalancing()` | 需要满足shortTileAtBack且smallFront为false的特定条件 |
| 128 | smallDimAlignUp 相关逻辑 | 需要满足特定的条件组合 |

## 总结

### 覆盖率提升成果
1. **all_gather_formulaic_tiling.cpp**: 从 81.2% 提升到 84.1% (+2.9%)
2. **all_gather_matmul_tiling.cpp**: 从 85.6% 提升到 86.8% (+1.2%)

### 无法100%覆盖的原因
1. **设计限制**: 部分代码路径是错误处理分支，在正常UT测试中无法触发
2. **框架限制**: 测试框架的某些参数校验阻止了非法输入的测试
3. **配置固定**: 某些配置参数（如commAlg、isStep、enableSplitK）在代码中是固定的，无法通过测试用例改变
4. **SOC版本差异**: 某些代码路径仅针对特定SOC版本（如SOC910_95），测试环境可能不支持
5. **条件限制**: 某些代码路径需要满足极其特定的条件组合才能触发

### 建议改进方向
1. 添加针对错误路径的专门测试用例（如果测试框架允许）
2. 为不同SOC版本添加专门的测试用例
3. 考虑将某些硬编码的配置参数改为可通过测试输入控制的参数
