# UT 覆盖率增强技能指南 v2.0

本文件为 Claude Code 在 ops-transformer 仓库中进行单元测试（UT）覆盖率分析和提升提供指导。

**v2.0 更新说明**: 基于实际项目经验，新增了测试失败诊断、场景检测、容器环境执行等重要章节。

## 项目概述

本项目是 ops-transformer 算子仓库的 UT 覆盖率增强工具，用于分析和提升算子测试用例的代码覆盖率。

### 核心功能

- 运行指定算子的 UT 并计算覆盖率
- 分析未覆盖的代码路径和分支
- 生成补充测试用例以提升覆盖率
- 诊断并修复测试失败问题
- 重新运行 UT 并分析覆盖率提升情况
- 生成覆盖率提升报告

## 工作流程

```
用户输入算子名称和测试类型
        |
        ▼
┌───────────────────┐
|  需求充分性检查    | ───▶ 信息不足 ──▶ [Interview 模式]
└───────────────────┘        主动询问算子名和UT类型
        | 信息充分
        ▼
┌───────────────────┐
|  检查覆盖率配置    | ───▶ 配置错误 ──▶ 修正 test_config.yaml
└───────────────────┘        或说明无法统计原因
        | 配置正确
        ▼
┌───────────────────┐
|  运行初始UT        |
|  获取覆盖率报告    |
└───────────────────┘
        |
        ▼
┌───────────────────┐
|  分析未覆盖代码    |
|  - 场景检测逻辑    | ⭐ 新增
|  - 未覆盖函数      |
|  - 未覆盖分支      |
└───────────────────┘
        |
        ▼
┌───────────────────┐
|  设计补充测试用例  |
|  - 触发条件分析    | ⭐ 新增
|  - 参数组合设计    |
|  - 场景识别策略    |
└───────────────────┘
        |
        ▼
┌───────────────────┐
|  实现测试用例      |
|  添加到原测试文件  |
└───────────────────┘
        |
        ▼
┌───────────────────┐
|  运行并验证测试    |
|  - 检查测试结果    | ⭐ 新增
|  - 诊断失败原因    |
|  - 修复并迭代      |
└───────────────────┘
        |
        ▼
┌───────────────────┐
|  生成提升报告      |
|  - 覆盖率变化      |
|  - 新增用例说明    |
|  - 提升要点总结    |
└───────────────────┘
```

---

## 环境和前置条件

### 1. 工作目录与执行环境 ⭐ 重要

**宿主机环境**:
- **工作目录**: `/home/y00845930/ascendc-agent/ops-transformer/`
- **输出目录**: `/home/y00845930/ascendc-agent/output/`

**容器环境** (用于编译和测试):
- **容器名称**: `y00845930_dev`
- **容器内路径**: `/ascendc-agent/ops-transformer/`
- **进入方式**: `docker exec y00845930_dev bash -c "命令"`

### 2. 文件操作原则 ⭐⭐⭐

> **所有测试文件的修改都在宿主机进行，然后复制到容器执行测试**

**工作流程**:
1. 在宿主机上读取、分析、修改测试文件
2. 使用 `docker cp` 将修改后的文件复制到容器
3. 在容器内执行编译和测试命令
4. 将测试结果和覆盖率报告复制回宿主机

**示例**:
```bash
# 宿主机上修改文件
vim mc2/matmul_all_reduce/tests/ut/op_host/test_matmul_all_reduce_910_tiling.cpp

# 复制到容器
docker cp mc2/matmul_all_reduce/tests/ut/op_host/test_matmul_all_reduce_910_tiling.cpp \
  y00845930_dev:/ascendc-agent/ops-transformer/mc2/matmul_all_reduce/tests/ut/op_host/

# 在容器内执行测试
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='matmul_all_reduce' --cov"

# 复制结果回宿主机
docker cp y00845930_dev:/ascendc-agent/ops-transformer/build/cov_result/coverage.info \
  /home/y00845930/ascendc-agent/output/
```

### 3. UT 类型说明

| 类型 | 说明 | 测试内容 |
|-----|------|---------|
| **ophost** | Host 侧测试 | 算子 tiling 和 infershape 的测试用例 |
| **opapi** | API 接口测试 | 算子 ACLNN 接口的测试用例 |
| **opkernel** | Kernel 侧测试 | 算子 kernel 的测试用例 |
| **opgraph** | 图模式测试 | 暂不支持 |

### 4. SOC 版本分类与全覆盖要求 ⭐⭐⭐ v2.0 新增

> **覆盖率目标：不同芯片型号能覆盖的实现全覆盖**
>
> 在设计测试用例时，必须考虑不同 SOC 版本的差异，确保所有支持的 SOC 版本都有对应的测试覆盖。

#### 4.1 SOC 版本分类

| SOC 版本 | 架构标识 | 芯片型号 | 代码位置 | 测试文件 |
|---------|---------|---------|---------|---------|
| **ASCEND310P** | arch31 | Ascend310P | `op_tiling/arch31/` | `test_*_310_tiling.cpp` |
| **ASCEND910B** | arch32 | Ascend910B | `op_tiling/arch32/` | `test_*_910_tiling.cpp` |
| **ASCEND910_95** | arch35 | Ascend910_95 | `op_tiling/arch35/` | `test_*_950_tiling.cpp` |
| **ASCEND910A** | arch33 | Ascend910A | `op_tiling/arch33/` | `test_*_910A_tiling.cpp` |

#### 4.2 SOC 版本差异化分析

在分析未覆盖代码时，必须按 SOC 版本分别分析：

| 分析项 | 说明 | 优先级 |
|-------|------|-------|
| **SOC 特定代码** | 每个版本有独立的实现文件 | ⭐⭐⭐ |
| **版本通用代码** | base 目录中的通用代码 | ⭐⭐ |
| **版本差异逻辑** | 不同版本的条件编译代码 | ⭐⭐⭐ |
| **版本限制差异** | 不同版本的支持能力差异 | ⭐⭐ |

#### 4.3 全覆盖验证标准

**全覆盖的定义**：
- 每个支持的 SOC 版本都有对应的测试用例
- 每个 SOC 版本的特定实现代码都有测试覆盖
- SOC 通用代码（base）被所有版本测试覆盖
- 不同版本的差异逻辑都被测试覆盖

**验证方法**：
```bash
# 检查每个 SOC 版本的覆盖率
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && lcov --capture --directory build --output-file coverage.info 2>&1" | grep -E "arch31|arch32|arch35|arch33"
```

#### 4.4 SOC 版本测试用例设计策略

**策略1: 版本独立测试**
- 为每个 SOC 版本创建独立的测试文件
- 测试文件命名：`test_<算子名>_<SOC标识>_tiling.cpp`
- 例如：`test_matmul_all_reduce_910_tiling.cpp` (arch32)

**策略2: 参数化测试**
- 使用参数化测试，同一个测试用例支持多个 SOC 版本
- 通过 `TEST_P` 宏实现版本参数化
- 示例：
  ```cpp
  class MatmulAllReduceTilingUT : public testing::TestWithParam<SOCVersion> {
      // 测试代码
  };

  INSTANTIATE_TEST_SUITE_P(MatmulAllReduceTilingUT, SOCVersion,
      testing::Values(SOCVersion::ASCEND910B,
                      SOCVersion::ASCEND910_95,
                      SOCVersion::ASCEND310P));
  ```

**策略3: 场景差异化设计**
- 分析不同 SOC 版本的功能差异
- 为版本独有功能设计专门测试
- 为版本限制设计边界测试

**示例：不同 SOC 版本的权重量化支持差异**

| 功能 | arch31 | arch32 | arch35 | 测试策略 |
|-----|--------|--------|--------|---------|
| A16W8 权重量化 | 不支持 | 支持 | 支持 | 为 arch32/arch35 设计测试 |
| FP8 权重量化 | 不支持 | 不支持 | 支持 | 为 arch35 设计测试 |
| A16W4 权重量化 | 不支持 | 支持 | 支持 | 为 arch32/arch35 设计测试 |
| FP8 PER_TENSOR | 不支持 | 不支持 | 不支持 | 为 arch35 设计失败测试 |

#### 4.5 SOC 版本覆盖率统计要求

**最终报告必须包含**：

| 统计项 | 说明 |
|-------|------|
| 总体覆盖率 | 所有 SOC 版本的综合覆盖率 |
| 各 SOC 版本覆盖率 | 每个 SOC 版本的独立覆盖率 |
| SOC 特定代码覆盖率 | `arch31/`, `arch32/`, `arch35/` 等目录的覆盖率 |
| 通用代码覆盖率 | `base` 目录代码被所有版本测试的情况 |

**覆盖率表格模板**：

| 文件/目录 | arch31 | arch32 | arch35 | arch33 | 总体 |
|---------|--------|--------|--------|--------|------|
| base/ | - | ✓ | ✓ | - | ✓ |
| arch31/ | ✓ | - | - | - | ✓ |
| arch32/ | - | ✓ | - | - | ✓ |
| arch35/ | - | - | ✓ | - | ✓ |
| **综合** | **✓** | **✓** | **✓** | **-** | **✓** |

**✓ 全覆盖检查清单**：
- [ ] 每个支持的 SOC 版本都有测试用例
- [ ] SOC 特定实现代码都有测试覆盖
- [ ] 版本差异逻辑都被测试覆盖
- [ ] 最终报告包含各 SOC 版本覆盖率统计

### 5. 编译运行命令

```bash
# 在容器内执行
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --cov"

# 编译运行单类型 UT 并获取覆盖率报告
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --opapi --ops='算子名称' --cov"
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --opkernel --ops='算子名称' --cov"

# 编译运行所有类型 UT
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ops='算子名称' --cov"
```

### 5. 调试环境变量

```bash
# 在容器内设置环境变量
docker exec y00845930_dev bash -c "export ASCEND_SLOG_PRINT_TO_STDOUT=1 && export ASCEND_GLOBAL_LOG_LEVEL=0 && cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称'"
```

---

## 核心职责与原则

### 核心职责

1. 分析算子 UT 的代码覆盖率，识别未覆盖的代码路径
2. 设计并实现补充测试用例，提升覆盖率
3. **针对不同 SOC 版本设计测试用例，实现 SOC 版本全覆盖** ⭐ v2.0 新增
4. 诊断并修复测试失败问题，确保测试用例有效
5. 验证测试用例的有效性，确保新增用例能真正提升覆盖率
6. 生成详细的覆盖率提升报告，包含 SOC 版本覆盖率统计

### 核心原则 ⭐

> **UT 覆盖率提升 ≠ 盲目增加测试用例数量**
>
> 目标是通过有针对性的测试用例设计，覆盖关键代码路径和边界条件

**测试设计原则**：

| 原则 | 说明 | 示例 |
|-----|------|------|
| **边界优先** | 优先测试边界值和极值 | 最小/最大形状、0值、空tensor |
| **路径完整** | 覆盖所有代码分支 | if/else各分支、异常处理路径 |
| **场景多样** | 测试不同参数组合 | 不同数据类型、不同shape组合 |
| **独立性** | 新增用例独立可运行 | 不依赖其他用例的执行结果 |
| **可验证** | 测试结果可预期和验证 | 明确期望的返回值和tiling key |

**常见误区**：

| 误区 | 正确做法 |
|-----|---------|
| 随意增加测试数据大小 | 分析未覆盖代码，有针对性设计 |
| 只关注行覆盖率 | 同时关注分支覆盖率和条件覆盖率 |
| 忽略异常路径 | 补充错误处理和边界条件测试 |
| 修改原测试用例 | 新增用例独立添加，不破坏原有测试 |
| 期望值设置错误 | 仔细分析代码逻辑，正确设置期望结果 |

---

## 各阶段详细指南

### 阶段一：需求充分性检查

在开始 UT 覆盖率增强前，确认以下信息是否完整：

#### 必需信息（缺失则进入 Interview 模式）
- [ ] **算子名称**：明确要增强哪个算子的 UT 覆盖率
- [ ] **UT 类型**：ophost、opapi、opkernel 或全部
- [ ] **目标覆盖率**（可选）：期望达到的覆盖率目标

#### Interview 模式触发条件
当用户输入满足以下任一条件时，使用 **AskUserQuestion** 工具主动提问：

1. **缺少算子名称**：未指定要处理哪个算子
2. **缺少 UT 类型**：未指定要处理哪种类型的测试
3. **信息模糊**：如"提升覆盖率"但未说明具体算子和类型

**✓ 检查清单**：
- [ ] 算子名称已确认
- [ ] UT 类型已确认
- [ ] 目标覆盖率已明确（如需）

### 阶段二：覆盖率配置检查

在运行 UT 前，**必须检查**覆盖率配置是否正确。

#### 1. 检查 test_config.yaml

查看算子在 `tests/test_config.yaml` 中的配置，确认 `ut_cov_exclude` 没有排除需要统计覆盖率的源码目录。

**✓ 检查清单**：
- [ ] 已检查 test_config.yaml 配置
- [ ] 确认需要统计的目录未被 ut_cov_exclude 排除
- [ ] 已记录初始配置状态

### 阶段三：运行初始UT并获取覆盖率

#### 1. 运行 UT 并保存日志

```bash
# 在容器内运行 UT 并获取覆盖率
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --cov" > /home/y00845930/ascendc-agent/output/original_cov.log 2>&1
```

#### 2. 解析覆盖率报告

从日志中提取覆盖率信息：
- 总体行覆盖率
- 各文件的行覆盖率
- 函数覆盖率

**✓ 检查清单**：
- [ ] UT 运行成功
- [ ] 覆盖率数据已提取
- [ ] 已识别主要未覆盖模块
- [ ] 已记录初始覆盖率数据

### 阶段四：分析未覆盖代码 ⭐ 关键阶段

#### 1. 分析未覆盖代码路径

从 HTML 报告或覆盖率数据中分析：

| 分析维度 | 检查内容 | 优先级 |
|---------|---------|-------|
| **未覆盖函数** | 哪些函数完全没有被测试覆盖 | 高 |
| **未覆盖分支** | if/else 条件分支的覆盖情况 | 高 |
| **未覆盖行** | 具体哪些代码行未被执行 | 中 |
| **场景检测逻辑** | 如何触发特定的代码路径 | ⭐⭐⭐ 极高 |
| **边界条件** | 边界值、极值的测试是否缺失 | 中 |
| **异常路径** | 错误处理、异常情况的测试是否缺失 | 中 |

#### 2. 场景检测逻辑分析 ⭐⭐⭐ v2.0 新增

**这是覆盖率提升的核心技巧！**

要触发特定的代码路径，必须理解代码的场景检测逻辑：

```cpp
// 示例：A16W8 权重量化场景检测
bool MatmulAllReduceTilingBase::IsA16W8Scenario(
    const DataType aType, const DataType bType,
    const gert::StorageShape* antiQuantScale) const
{
    // 条件1: A16W8 (FP16/BF16 x INT8)
    if ((aType != ge::DT_INT8) && (bType == ge::DT_INT8) && (antiQuantScale != nullptr)) {
        return true;
    }
    // 条件2: FP8/HIF8
    if (((aType == ge::DT_FLOAT16) || (aType == ge::DT_BF16)) &&
        ((bType == ge::DT_FLOAT8_E4M3FN) || (bType == ge::DT_FLOAT8_E5M2) || (bType == ge::DT_HIFLOAT8)) &&
        (antiQuantScale != nullptr)) {
        return true;
    }
    return false;
}
```

**场景检测分析步骤**:

1. **定位场景检测函数**: 查找 `IsCapable()`, `GetScenario()`, `CheckInput()` 等函数
2. **分析触发条件**: 仔细分析每个 if/else 分支的条件
3. **设计参数组合**: 根据条件设计测试参数
4. **验证参数正确性**: 确认参数类型、格式等符合要求

#### 3. SOC 版本差异化分析 ⭐⭐⭐ v2.0 新增

> **按 SOC 版本分别分析未覆盖代码，确保版本全覆盖**

**分析维度**:

| 分析项 | 说明 | 优先级 |
|-------|------|-------|
| **SOC 特定代码** | 每个版本独立的实现文件 | ⭐⭐⭐ |
| **版本功能差异** | 不同版本支持的功能不同 | ⭐⭐⭐ |
| **版本限制差异** | 不同版本的限制条件不同 | ⭐⭐ |
| **条件编译代码** | #ifdef SOC_VERSION 条件编译的代码 | ⭐⭐ |

**分析步骤**:

1. **识别 SOC 版本特定文件**:
   ```bash
   # 查找各 SOC 版本的实现文件
   ls mc2/算子名/op_host/op_tiling/
   # 输出：arch31/ arch32/ arch35/ 等
   ```

2. **分析每个版本的未覆盖代码**:
   ```bash
   # 查看特定 SOC 版本的覆盖率
   grep -A 20 "arch32/" coverage.info
   ```

3. **识别版本差异功能**:
   - 阅读各版本的实现文件
   - 比对不同版本的功能支持
   - 找出版本独有功能

4. **设计版本特定测试**:
   - 为版本独有功能设计专门测试
   - 为版本限制设计边界测试

**示例：权重量化功能的 SOC 版本差异**

| 功能 | arch31 | arch32 | arch35 | 测试策略 |
|-----|--------|--------|--------|---------|
| A16W8 权重量化 | ❌ | ✅ | ✅ | 为 arch32/arch35 设计测试 |
| FP8 权重量化 | ❌ | ❌ | ✅ | 为 arch35 设计测试 |
| A16W4 权重量化 | ❌ | ✅ | ✅ | 为 arch32/arch35 设计测试 |
| FP8 PER_TENSOR | ❌ | ❌ | ❌ | arch35 设计失败测试 |
| FP8 PER_GROUP | ❌ | ❌ | ❌ | arch35 设计失败测试 |

**✓ SOC 分析检查清单**：
- [ ] 已识别所有支持的 SOC 版本
- [ ] 已分析每个 SOC 版本的未覆盖代码
- [ ] 已识别版本功能差异
- [ ] 已设计版本特定测试用例

### 5. 测试框架限制分析

| 要点 | 说明 | 常见错误 |
|-----|------|---------|
| **数据类型** | aType, bType 的组合必须符合场景要求 | 类型不匹配导致场景检测失败 |
| **指针参数** | antiQuantScale, dequantScale 等指针必须非空或为空 | 指针为空导致场景检测失败 |
| **格式选择** | ND vs FRACTAL_NZ，不同格式有不同的维度要求 | 使用 FRACTAL_NZ 但只提供 2D tensor |
| **参数约束** | groupSize, alignDim 等有最小值和对齐要求 | groupSize < 最小值导致错误 |

#### 3. 测试框架限制分析 ⭐ v2.0 新增

**测试框架限制可能导致无法触发某些代码路径**:

| 限制类型 | 说明 | 解决方案 |
|---------|------|---------|
| **维度限制** | 框架只提供 2D tensor，但某些格式需要 4D | 使用 ND 格式替代 FRACTAL_NZ |
| **数据类型限制** | 框架不支持某些数据类型 | 选择支持的数据类型进行测试 |
| **环境限制** | 需要 NPU 硬件或多卡环境 | 说明环境限制，跳过相关测试 |

**ND vs FRACTAL_NZ 格式选择**:

```cpp
// FRACTAL_NZ 格式: 需要 4D tensor (batch, M, N, K)
// 但测试框架只提供 2D tensor (M, K)

// 解决方案: 使用 ND 格式
FORMAT_ND  // 2D tensor，通过维度检查
```

**✓ 检查清单**：
- [ ] 已识别未覆盖的函数和分支
- [ ] 已分析场景检测逻辑
- [ ] 已理解触发条件
- [ ] 已设计参数组合
- [ ] 已考虑测试框架限制

### 阶段五：设计补充测试用例

#### 1. 测试场景补充策略

根据 UT 类型，采用不同的补充策略：

##### ophost 测试场景

| 场景类别 | 测试内容 | 示例 |
|---------|---------|------|
| **Tiling 参数** | 不同输入形状组合 | 小/中/大shape、不规则shape |
| **InferShape** | 各种输入形状推导 | 动态维度、rank变化、空tensor |
| **多卡场景** | 分布式参数 | 不同rank、不同通信模式 |
| **权重量化** | A16W8/A16W4/FP8 场景 | 不同量化类型、不同反量化类型 |

##### opapi 测试场景

| 场景类别 | 测试内容 | 示例 |
|---------|---------|------|
| **参数验证** | 空指针、无效参数 | nullptr检查、越界检查 |
| **Workspace** | workspace大小获取 | 最小/最大workspace |
| **正常流程** | 各种数据类型 | float16、float32、int32 |

##### opkernel 测试场景

| 场景类别 | 测试内容 | 示例 |
|---------|---------|------|
| **计算逻辑** | 不同数据类型 | fp32、fp16、int8 |
| **边界值** | 特殊值测试 | 0、NaN、Inf、最大/最小值 |
| **性能路径** | 不同数据量 | 小数据量 vs 大数据量 |

#### 2. 测试用例设计要点 ⭐⭐⭐ v2.0 更新

| 设计要点 | 说明 | 关键注意事项 |
|---------|------|-------------|
| **场景优先** | 先分析场景检测逻辑，再设计用例 | 不要盲目设计参数组合 |
| **边界优先** | 最小值、最大值、0值、空tensor | 边界值要能触发错误检查 |
| **路径完整** | 覆盖所有 if/else 分支 | 包括错误处理路径 |
| **参数组合** | 不同数据类型、shape组合 | 必须符合场景检测条件 |
| **期望正确** | 仔细设置 expectResult 和 tiling key | 期望值错误会导致测试失败 |

**期望值设置指南** ⭐⭐⭐:

```cpp
// 成功场景的期望值设置
ge::GRAPH_SUCCESS, correct_tiling_key

// 失败场景的期望值设置
ge::GRAPH_FAILED, 4294967295UL  // GRAPH_FAILED = 4294967295

// 注意: 即使触发错误路径，如果代码返回了 GRAPH_SUCCESS，测试也会失败
// 需要仔细分析代码逻辑，正确设置期望值
```

**PER_CHANNEL vs PER_GROUP 判断逻辑** ⭐⭐⭐:

```cpp
// 关键发现: 当提供 antiQuantGroupSize 参数时，代码会将其识别为 PER_GROUP 场景
// 无论测试用例名称如何

// PER_CHANNEL: 不提供 groupSize 参数
// PER_GROUP: 提供 groupSize 参数 (>= 最小值，如 32、64 等)

// 错误示例:
{"test_name", ..., 0, 0,  // groupSize = 0, 期望 PER_CHANNEL 但实际被识别为 PER_GROUP
 ge::GRAPH_SUCCESS, PER_CHANNEL_tiling_key}  // 错误！会使用 PER_GROUP key

// 正确示例:
{"test_name_per_channel", ...,  // 不设置 groupSize 参数
 ge::GRAPH_SUCCESS, PER_CHANNEL_tiling_key}

{"test_name_per_group", ..., 64,  // 设置 groupSize = 64
 ge::GRAPH_SUCCESS, PER_GROUP_tiling_key}
```

**✓ 检查清单**：
- [ ] 已分析场景检测逻辑
- [ ] 已设计针对性的测试场景
- [ ] 已明确每个测试用例的目的
- [ ] 期望值设置正确
- [ ] tiling key 设置正确

#### 3. SOC 版本差异化设计 ⭐⭐⭐ v2.0 新增

> **为不同 SOC 版本设计差异化的测试用例，实现版本全覆盖**

**差异化设计策略**:

| 策略 | 说明 | 示例 |
|-----|------|------|
| **版本独立功能测试** | 为版本独有功能设计专门测试 | arch35 的 FP8 支持 |
| **版本限制边界测试** | 为不同版本的限制条件设计测试 | arch31 不支持权重量化 |
| **版本对齐要求测试** | 测试不同版本的对齐要求差异 | A16W4 需要 64 对齐 (arch35) |
| **版本错误检查测试** | 测试版本特定的错误检查 | FP8 不支持 PER_TENSOR (arch35) |

**SOC 版本功能差异识别方法**:

1. **查看版本特定实现文件**:
   ```bash
   # 查看各版本的实现
   ls mc2/算子名/op_host/op_tiling/arch*/
   # arch31/, arch32/, arch35/, arch33/
   ```

2. **分析版本功能支持**:
   ```cpp
   // 示例：查找 FP8 支持的版本
   grep -r "DT_FLOAT8_E4M3FN\|FP8" mc2/算子名/op_host/op_tiling/
   ```

3. **识别条件编译差异**:
   ```cpp
   // 查找条件编译代码
   grep -r "#ifdef.*ASCEND.*910\|#ifndef.*ASCEND.*910" mc2/算子名/
   ```

**版本差异测试用例设计示例**:

```cpp
// Arch31 (ASCEND310P) - 不支持权重量化
{"a16w8_weight_quant_not_supported_310", ..., ge::DT_FLOAT16, ge::FORMAT_ND),
 TD({...}, ge::DT_INT8, ge::FORMAT_ND), ...,
 ge::GRAPH_FAILED, 4294967295UL},  // 期望失败

// Arch32 (ASCEND910B) - 支持 A16W8 但不支持 FP8
{"a16w8_per_tensor_nd_success_910", ..., ge::GRAPH_SUCCESS, 910_tiling_key},

// Arch35 (ASCEND910_95) - 支持 A16W8 和 FP8
{"a16w8_per_tensor_nd_success_950", ..., ge::GRAPH_SUCCESS, 950_tiling_key},
{"fp8_per_tensor_not_supported_950", ..., ge::GRAPH_FAILED, 4294967295UL},  // FP8 不支持
```

**版本对齐要求差异测试**:

```cpp
// A16W4 PER_GROUP 对齐要求
// arch32: 需要对齐，但检查较宽松
{"a16w4_per_group_910", TD({{256, 64}, {256, 64}}, ...),  // K,N=64，满足对齐
 ge::GRAPH_SUCCESS, 910_tiling_key},

// arch35: 必须严格 64 对齐
{"a16w4_per_group_950_aligned", TD({{256, 64}, {256, 64}}, ...),  // K,N=64，满足对齐
 ge::GRAPH_SUCCESS, 950_tiling_key},

{"a16w4_per_group_950_not_aligned", TD({{256, 65}, {256, 65}}, ...),  // K,N=65，不满足对齐
 ge::GRAPH_FAILED, 4294967295UL},  // 期望失败
```

**版本特定错误检查测试**:

```cpp
// arch35 特有：FP8 PER_TENSOR 不支持
{"fp8_per_tensor_not_supported_950",
 TD({{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND),
 TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND),
 ..., // 提供 antiQuantScale
 ge::GRAPH_FAILED, 4294967295UL},
```

**✓ SOC 版本检查清单**：
- [ ] 已识别所有支持的 SOC 版本
- [ ] 已分析各版本功能差异
- [ ] 已为版本独有功能设计测试
- [ ] 已为版本限制设计边界测试
- [ ] 已为版本差异设计差异化测试

### 阶段六：实现测试用例

#### 1. 添加测试用例

在原始测试文件中添加补充的测试用例，使用注释标识：

```cpp
// ============================================================================
// 以下为补充的测试用例 - 用于提升 weight_quant_matmul_all_reduce_tiling 覆盖率
// 添加时间: 2026-01-29
// 目的: 使用 ND 格式触发 A16W8 权重量化场景
// ============================================================================

{"a16w8_per_tensor_nd_success", TD({{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND),
 TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND),
 std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND),
 std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND),
 "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2293772UL},
```

#### 2. 测试用例命名规范

```cpp
// 命名格式：{场景描述}_{格式/状态}_{期望结果}
a16w8_per_tensor_nd_success      // A16W8 PER_TENSOR + ND 格式 + 期望成功
a16w8_per_group_not_aligned_fail // A16W8 PER_GROUP + 未对齐 + 期望失败
```

**✓ 检查清单**：
- [ ] 新增测试用例已添加到原测试文件
- [ ] 测试用例已添加明确注释
- [ ] 测试用例命名符合规范
- [ ] 测试用例可独立运行

### 阶段七：运行并验证测试 ⭐⭐⭐ v2.0 新增

#### 1. 编译运行更新后的 UT

```bash
# 在容器内执行测试
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --cov"
```

#### 2. 测试失败诊断 ⭐⭐⭐

**常见失败原因及处理方法**:

| 失败类型 | 错误信息 | 原因分析 | 处理方法 |
|---------|---------|---------|---------|
| **Tiling key 不匹配** | `Expected: 4390924, Actual: 6488076` | 使用的 tiling key 与场景不符 | 从测试输出提取实际 key |
| **期望值错误** | `Expected: GRAPH_FAILED, Actual: GRAPH_SUCCESS` | 期望值设置错误 | 分析代码逻辑，修正期望值 |
| **场景未触发** | 测试通过但未覆盖目标代码 | 参数组合未触发目标场景 | 检查场景检测条件，调整参数 |
| **参数校验失败** | `CheckInput failed` | 参数不符合校验规则 | 修正参数或设计为失败测试 |

**Tiling Key 提取方法**:

```bash
# 从测试输出中提取实际的 tiling key
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' 2>&1" | grep "actualTilingKey\|Which is: [0-9]"
```

**期望值判断方法**:

```cpp
// 1. 阅读源代码，分析该场景应该返回什么
// 2. 检查是否触发错误检查 (OP_TILING_CHECK)
// 3. 确认是否有提前返回 return ge::GRAPH_FAILED

// 如果代码中有 OP_TILING_CHECK 且条件满足，则期望 GRAPH_FAILED
// 如果所有检查都通过，则期望 GRAPH_SUCCESS
```

#### 3. 测试迭代优化流程

```
运行测试
    |
    ▼
测试通过？
    |
    ├── 是 ──▶ 验证覆盖率是否提升 ──▶ 覆盖率已提升 ──▶ 完成
    │
    └── 否 ──▶ 分析失败原因
                |
                ▼
        ┌───────────────┐
        │ 失败类型判断 │
        └───────────────┘
                |
        ├── Tiling key 不匹配 ──▶ 提取实际 key，更新测试用例
        ├── 期望值错误 ──────────▶ 分析代码逻辑，修正期望值
        ├── 场景未触发 ──────────▶ 检查参数组合，调整测试用例
        └── 其他错误 ──────────────▶ 分析日志，修正代码或数据
                |
                ▼
        重新运行测试
```

**✓ 检查清单**：
- [ ] UT 运行成功
- [ ] 新增测试用例通过
- [ ] 已验证覆盖率提升
- [ ] 已分析失败原因并修复

### 阶段八：生成提升报告

#### 1. 报告内容要求

完整的覆盖率提升报告应包含：

##### 1.1 测试信息
- 算子名称和测试类型
- 测试时间和执行环境
- 测试命令

##### 1.2 测试结果
- 通过/失败的测试用例数量
- 失败测试用例的处理说明

##### 1.3 覆盖率数据
- 初始覆盖率百分比
- 最终覆盖率百分比
- 覆盖率提升幅度
- 各文件的覆盖率对比

##### 1.4 补充的测试用例
- 新增测试用例列表
- 每个测试用例覆盖的场景说明

##### 1.5 提升要点总结
- 关键突破点说明
- 技术难点和解决方案
- 经验教训和最佳实践

#### 2. 报告格式模板

```
## UT 覆盖率提升报告 - [算子名称]

### 测试结果
- 总测试用例数: X
- 通过: Y
- 失败: Z

### 覆盖率数据
| 文件 | 初始覆盖率 | 最终覆盖率 | 提升幅度 |
|-----|-----------|-----------|---------|
| xxx.cpp | X% | Y% | +Z% |

### 补充测试用例
1. 用例名称: xxx
   - 覆盖场景: xxx
   - 对应代码: xxx.cpp 第 Y 行

### 提升要点
1. 关键突破: xxx
2. 技术难点: xxx
```

**✓ 检查清单**：
- [ ] 报告包含测试结果
- [ ] 报告包含覆盖率数据
- [ ] 报告包含补充用例说明
- [ ] 报告包含提升要点或限制说明

---

## 重要注意事项

### 1. 场景检测分析优先原则 ⭐⭐⭐

> **始终先分析场景检测逻辑，再设计测试用例**
>
> 在设计测试用例前，必须理解代码的场景检测逻辑，分析触发条件，然后设计相应的参数组合。

**分析步骤**:
1. 查找场景检测相关函数
2. 分析每个 if/else 分支的条件
3. 理解参数约束和限制
4. 设计符合要求的参数组合

### 2. 测试框架限制原则 ⭐⭐

> **了解测试框架的限制，避免设计无法执行的测试用例**
>
> 测试框架可能只支持特定的数据格式、维度范围等，需要在设计用例时考虑这些限制。

**常见限制**:
- 只提供 2D tensor，不支持 4D FRACTAL_NZ
- 不支持某些数据类型
- 环境、硬件依赖

**解决方案**:
- 使用 ND 格式替代 FRACTAL_NZ
- 选择支持的数据类型
- 说明环境限制

### 3. 期望值设置谨慎原则 ⭐⭐⭐

> **仔细分析代码逻辑，正确设置期望值**
>
> 期望值错误会导致测试失败，即使测试用例已经触发了正确的代码路径。

**设置要点**:
- 阅读源代码，理解返回值逻辑
- 检查是否有 OP_TILING_CHECK
- 确认是否提前返回
- 提取实际的 tiling key

### 4. 增量开发与迭代原则

> **先实现一个测试场景，逐步验证和迭代**
>
> 不要一次性添加大量测试用例，容易导致调试困难。

**迭代流程**:
1. 实现一个测试场景
2. 运行测试验证
3. 分析失败原因
4. 修复问题
5. 继续下一个场景

### 5. 文件操作安全原则 ⭐⭐⭐

> **在宿主机上修改文件，然后复制到容器执行测试**
>
> 不要在容器内直接修改文件，容易导致权限和同步问题。

---

## 常见问题与处理 ⭐ v2.0 扩展

### 问题1：测试用例失败 - Tiling Key 不匹配

**错误信息**:
```
Expected equality of these values:
  tilingKey
    Which is: 6488076
  expectTilingKey
    Which is: 4390924
```

**原因分析**:
- 使用的 tiling key 与实际场景不符
- 可能是 PER_CHANNEL vs PER_GROUP 的 key 混淆

**处理方法**:
1. 从测试输出中提取实际的 tiling key
2. 理解不同场景对应的 tiling key
3. 更新测试用例中的 tiling key

**提取方法**:
```bash
# 从测试输出提取 tiling key
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' 2>&1" | grep -E "actualTilingKey|Which is: [0-9]"
```

### 问题2：测试用例失败 - 期望值错误

**错误信息**:
```
Expected equality of these values:
  tilingRet (0)
  expectResult (4294967295)
```

**原因分析**:
- 期望值设置错误
- 代码实际返回了 GRAPH_SUCCESS，但期望的是 GRAPH_FAILED

**处理方法**:
1. 阅读源代码，分析该场景应该返回什么
2. 检查是否触发错误检查
3. 确认是否有提前返回
4. 修正期望值

**判断方法**:
- 如果代码中有 `OP_TILING_CHECK` 且条件满足 → 期望 GRAPH_FAILED
- 如果所有检查都通过 → 期望 GRAPH_SUCCESS

### 问题3：测试用例通过但覆盖率未提升

**原因分析**:
- 参数组合未触发目标场景
- 场景检测条件不满足
- 代码路径不可达（死代码）

**处理方法**:
1. 检查参数是否满足场景检测条件
2. 使用调试日志验证场景是否触发
3. 分析代码是否可达

**验证方法**:
```bash
# 使用调试日志
docker exec y00845930_dev bash -c "export ASCEND_SLOG_PRINT_TO_STDOUT=1 && export ASCEND_GLOBAL_LOG_LEVEL=0 && cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称'" | grep "场景名称"
```

### 问题4：测试框架限制 - 维度不匹配

**错误信息**:
```
Expect x2 dim to be 4, but got x2 dim:[2]
```

**原因分析**:
- FRACTAL_NZ 格式需要 4D tensor
- 测试框架只提供 2D tensor

**处理方法**:
- 使用 ND 格式替代 FRACTAL_NZ
- 或说明无法测试该场景

### 问题5：PER_GROUP vs PER_CHANNEL 混淆

**错误信息**:
```
In the per-group scenario, antiquantGroupSize should be in range [32, ...], Actual is 16
```

**原因分析**:
- 提供了 groupSize 参数，代码将其识别为 PER_GROUP
- groupSize < 最小值限制

**处理方法**:
- 理解 PER_GROUP vs PER_CHANNEL 的判断逻辑
- PER_CHANNEL: 不提供 groupSize 参数
- PER_GROUP: 提供 groupSize 参数 (>= 最小值)

---

## 覆盖率提升技巧与最佳实践 ⭐⭐⭐ v2.0 新增

### 1. 场景驱动测试设计

**核心思想**: 从场景检测逻辑出发，设计能触发特定代码路径的测试用例

**步骤**:
1. **定位场景检测函数**: 找到判断使用哪种实现方式的代码
2. **分析触发条件**: 理解 if/else 的判断逻辑
3. **设计参数组合**: 根据条件设计测试参数
4. **验证触发效果**: 确认测试用例能触发目标路径

**示例**:
```cpp
// 场景检测逻辑
bool IsA16W8Scenario(DataType aType, DataType bType, const StorageShape* antiQuantScale) {
    return (aType != DT_INT8) && (bType == DT_INT8) && (antiQuantScale != nullptr);
}

// 参数设计
aType = DT_FLOAT16    // 不是 INT8 ✓
bType = DT_INT8       // 是 INT8 ✓
antiQuantScale != 0  // 非空 ✓
```

### 2. 参数组合设计技巧

| 技巧 | 说明 | 示例 |
|-----|------|------|
| **格式选择** | 优先使用 ND 格式，避免维度限制 | FORMAT_ND vs FORMAT_FRACTAL_NZ |
| **指针参数** | 必须满足非空/为空的要求 | antiQuantScale != nullptr |
| **数值范围** | 边界值要能触发检查 | groupSize < 32 触发错误 |
| **类型匹配** | 数据类型必须符合场景要求 | FP16 + INT8 触发 A16W8 |

### 3. 调试技巧

**使用日志验证场景触发**:
```bash
# 设置环境变量启用日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0

# 运行测试并过滤日志
bash build.sh -u --ophost --ops='算子名称' 2>&1 | grep "场景关键词"
```

**分析代码流程**:
1. 从测试入口开始跟踪
2. 记录每个函数调用的参数
3. 检查每个 if/else 分支的判断结果
4. 确认最终进入哪个代码路径

### 4. 迭代优化流程

```
初步设计测试用例
    |
    ▼
运行测试
    |
    ├── 成功 ──▶ 验证覆盖率 ──▶ 覆盖率提升 ──▶ 完成
    │
    └── 失败 ──▶ 分析失败原因
                  |
                  ▼
          ┌─────────────────┐
          │ 失败类型判断   │
          └─────────────────┘
                  |
    ├── Tiling key 错 ──▶ 提取实际 key
    ├── 期望值 错 ─────▶ 分析代码逻辑
    ├── 场景未触发 ────▶ 检查参数组合
    └── 参数错误 ──────▶ 修正测试数据
                  |
                  ▼
          修正测试用例
                  |
                  ▼
          重新运行测试
```

### 5. 最佳实践总结

| 实践 | 说明 | 效果 |
|-----|------|------|
| **场景优先** | 先分析场景检测逻辑 | 提高测试用例针对性 |
| **ND 格式优先** | 使用 ND 格式避免维度限制 | 触发更多代码路径 |
| **参数验证** | 仔细检查参数组合 | 确保场景正确触发 |
| **期望值谨慎** | 分析代码逻辑设置期望值 | 避免期望值错误 |
| **迭代优化** | 逐步验证和修正 | 提高测试用例质量 |
| **日志调试** | 使用日志验证场景 | 快速定位问题 |

---

## 示例命令序列

```bash
# 1. 查看现有测试文件
ls mc2/matmul_all_reduce/tests/ut/op_host/

# 2. 分析场景检测逻辑
grep -n "IsA16W8Scenario\|IsCapable" mc2/matmul_all_reduce/op_host/op_tiling/*.cpp

# 3. 设计测试参数（在宿主机上）
vim mc2/matmul_all_reduce/tests/ut/op_host/test_matmul_all_reduce_910_tiling.cpp

# 4. 复制到容器
docker cp mc2/matmul_all_reduce/tests/ut/op_host/test_matmul_all_reduce_910_tiling.cpp \
  y00845930_dev:/ascendc-agent/ops-transformer/mc2/matmul_all_reduce/tests/ut/op_host/

# 5. 运行测试（在容器内）
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='matmul_all_reduce' --cov"

# 6. 提取 tiling key（如果需要）
docker exec y00845930_dev bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='matmul_all_reduce' 2>&1" | grep -E "actualTilingKey|Which is: [0-9]"

# 7. 复制结果回宿主机
docker cp y00845930_dev:/ascendc-agent/ops-transformer/build/cov_result/coverage.info \
  /home/y00845930/ascendc-agent/output/

# 8. 生成报告
# 手动编写或使用工具生成覆盖率提升报告
```

---

## 开发失败处理

当遇到无法解决的覆盖率提升问题时：

1. **记录失败原因**：
   - 明确说明哪些代码无法覆盖
   - 记录已尝试的测试场景和结果
   - 提供相关配置或环境限制说明

2. **诚实报告**：
   - 不要强行增加无效测试用例
   - 明确标注"覆盖率无法提升"及原因
   - 说明是配置限制、技术限制还是代码本身问题

3. **提供改进建议**：
   - 建议可能的配置调整
   - 建议代码优化方向
   - 建议测试环境改进

4. **保存工作成果**：
   - 保存所有尝试过的测试用例
   - 保存覆盖率数据和分析结果
   - 生成详细的问题报告
