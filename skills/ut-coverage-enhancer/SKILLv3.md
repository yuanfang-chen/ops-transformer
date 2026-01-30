# UT 覆盖率增强技能指南 v3.0

本文件为 Claude Code 在 ops-transformer 仓库中进行单元测试（UT）覆盖率分析和提升提供指导。

**v3.0 更新说明**: 基于24个算子批量覆盖率增强经验，新增批量处理模式、环境自适应处理、进度跟踪等功能，优化文档结构，提升易用性。

---

# 第一部分：快速开始

## 5分钟快速入门

### 基本使用流程

```bash
# 步骤1: 运行初始UT获取覆盖率
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && \
    bash build.sh -u --ophost --ops='算子名' --soc='支持的SOC' --cov"

# 步骤2: 分析未覆盖代码
grep -rn "IsCapable\|IsSupported\|CheckInput" 算子名/op_host/

# 步骤3: 设计并添加测试用例（在宿主机编辑）
vim 算子名/tests/ut/op_host/test_算子名_910_tiling.cpp

# 步骤4: 复制到容器并验证
docker cp 测试文件 y00845930:/ascendc-agent/ops-transformer/路径/
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && \
    bash build.sh -u --ophost --ops='算子名' --soc='支持的SOC' --cov"
```

## 常用命令速查

| 功能 | 命令 |
|------|------|
| 单个算子处理 | `ut-coverage-enhancer --operator=算子名 --ut-type=ophost` |
| 批量处理 | `ut-coverage-enhancer --batch --dir=mc2/` |
| 生成汇总报告 | `ut-coverage-enhancer --summary --output=report.md` |
| 检查环境 | `docker ps \| grep y00845930` |
| 提取tiling key | `grep -oP "actualTilingKey: \K[0-9]+" 测试日志` |
| 查找场景检测 | `grep -rn "IsCapable\|IsSupported" op_host/` |

## 快速参考

### 覆盖率目标

| 算子类型 | 目标覆盖率 | 关键点 |
|---------|-----------|--------|
| 简单算子 | > 90% | 基础场景全覆盖 |
| 复杂算子 | > 80% | 主要分支覆盖 |
| 依赖算子 | > 70% | 核心逻辑覆盖 |

### 快速诊断

| 问题 | 检查命令 | 解决方法 |
|------|---------|----------|
| 编译失败 | `bash build.sh -u --ophost --ops='op'` | 检查依赖 |
| 测试失败 | 查看测试日志 | 提取实际key |
| 覆盖率不提升 | 检查 `ut_cov_exclude` | 修改配置 |

### 环境检测

```bash
# 检查容器状态
docker ps | grep y00845930

# 检查编译环境
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && \
    bash build.sh -u --ophost --ops='test_op' 2>&1 | head -20"
```

---

# 第二部分：完整指南

## 项目概述

本项目是 ops-transformer 算子仓库的 UT 覆盖率增强工具，用于分析和提升算子测试用例的代码覆盖率。

### 核心功能

- 运行指定算子的 UT 并计算覆盖率
- 分析未覆盖的代码路径和分支
- 生成补充测试用例以提升覆盖率
- 诊断并修复测试失败问题
- 重新运行 UT 并分析覆盖率提升情况
- 生成覆盖率提升报告
- **批量处理多个算子** ⭐ v3.0 新增
- **环境自适应处理** ⭐ v3.0 新增

## 工作流程

```
用户输入算子名称和测试类型
        |
        ▼
┌───────────────────┐
│  需求充分性检查    │ ───▶ 信息不足 ──▶ [Interview 模式]
└───────────────────┘        主动询问算子名和UT类型
        | 信息充分
        ▼
┌───────────────────┐
│  环境检测与模式选择│ ⭐ v3.0 新增
│  - 容器状态检查    │ ───▶ 环境完整 ──▶ [实际测试模式]
│  - 编译环境验证    │                    运行测试获取实际覆盖率
│  - 模式决策        │ ───▶ 环境缺失 ──▶ [静态分析模式]
└───────────────────┘                    静态分析预估覆盖率
        |
        ▼
┌───────────────────┐
│  检查覆盖率配置    │ ───▶ 配置错误 ──▶ 修正 test_config.yaml
└───────────────────┘        或说明无法统计原因
        | 配置正确
        ▼
┌───────────────────┐
│  运行初始UT        │
│  获取覆盖率报告    │
└───────────────────┘
        |
        ▼
┌───────────────────┐
│  分析未覆盖代码    │
│  - 场景检测逻辑    │
│  - 未覆盖函数      │
│  - 未覆盖分支      │
└───────────────────┘
        |
        ▼
┌───────────────────┐
│  设计补充测试用例  │
│  - 触发条件分析    │
│  - 参数组合设计    │
│  - 场景识别策略    │
└───────────────────┘
        |
        ▼
┌───────────────────┐
│  实现测试用例      │
│  添加到原测试文件  │
└───────────────────┘
        |
        ▼
┌───────────────────┐
│  运行并验证测试    │
│  - 检查测试结果    │
│  - 诊断失败原因    │
│  - 修复并迭代      │
└───────────────────┘
        |
        ▼
┌───────────────────┐
│  生成提升报告      │
│  - 覆盖率变化      │
│  - 新增用例说明    │
│  - 提升要点总结    │
└───────────────────┘
```

## 环境和前置条件

### 1. 工作目录与执行环境

**宿主机环境**:

- **工作目录**: `/home/y00845930/ascendc-agent/ops-transformer/`
- **输出目录**: `/home/y00845930/ascendc-agent/output/`

**容器环境** (用于编译和测试):

- **容器名称**: `y00845930`
- **容器内路径**: `/ascendc-agent/ops-transformer/`
- **进入方式**: `docker exec y00845930 bash -c "命令"`

### 2. 环境检测 ⭐ v3.0 新增

在开始任务前，自动检测环境状态：

```bash
# 检查容器状态
docker ps | grep y00845930

# 检查编译环境
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && \
    bash build.sh -u --ophost --ops='test_op' 2>&1 | head -20"
```

根据检测结果，自动选择处理模式：
- **实际测试模式**: 环境完整时使用
- **静态分析模式**: 环境缺失时使用

### 3. 容器资源管理 ⭐ v3.0 新增

| 资源 | 建议 | 说明 |
|------|------|------|
| 磁盘空间 | 定期清理 build/ 目录 | 覆盖率数据文件较大 |
| 内存使用 | 每5-10个算子重启容器 | 避免内存泄漏累积 |
| CPU占用 | 串行处理降低负载 | 避免并行编译冲突 |

### 4. 文件操作原则 ⭐⭐⭐

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
  y00845930:/ascendc-agent/ops-transformer/mc2/matmul_all_reduce/tests/ut/op_host/

# 在容器内执行测试
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='matmul_all_reduce' --soc='支持的SOC' --cov"

# 复制结果回宿主机
docker cp y00845930:/ascendc-agent/ops-transformer/build/cov_result/coverage.info \
  /home/y00845930/ascendc-agent/output/
```

### 5. UT 类型说明

| 类型         | 说明          | 测试内容                             |
| ------------ | ------------- | ------------------------------------ |
| **ophost**   | Host 侧测试   | 算子 tiling 和 infershape 的测试用例 |
| **opapi**    | API 接口测试  | 算子 ACLNN 接口的测试用例            |
| **opkernel** | Kernel 侧测试 | 算子 kernel 的测试用例               |
| **opgraph**  | 图模式测试    | 暂不支持                             |

### 6. SOC 版本分类与全覆盖要求 ⭐⭐⭐

> **覆盖率目标：不同芯片型号能覆盖的实现全覆盖**

#### 6.1 SOC 版本分类

| SOC 版本         | 架构标识 | 芯片型号     | 代码位置            | 测试文件                 |
| ---------------- | -------- | ------------ | ------------------- | ------------------------ |
| **ASCEND310P**   | arch31   | Ascend310P   | `op_tiling/arch31/` | `test_*_310_tiling.cpp`  |
| **ASCEND910B**   | arch32   | Ascend910B   | `op_tiling/arch32/` | `test_*_910_tiling.cpp`  |
| **ASCEND910_95** | arch35   | Ascend910_95 | `op_tiling/arch35/` | `test_*_950_tiling.cpp`  |
| **ASCEND910C**   | arch33   | Ascend910C   | `op_tiling/arch33/` | `test_*_910C_tiling.cpp` |

#### 6.2 SOC 版本差异化分析

| 分析项           | 说明                     | 优先级 |
| ---------------- | ------------------------ | ------ |
| **SOC 特定代码** | 每个版本有独立的实现文件 | ⭐⭐⭐ |
| **版本通用代码** | base 目录中的通用代码    | ⭐⭐   |
| **版本差异逻辑** | 不同版本的条件编译代码   | ⭐⭐⭐ |
| **版本限制差异** | 不同版本的支持能力差异   | ⭐⭐   |

#### 6.3 全覆盖验证标准

- 每个支持的 SOC 版本都有对应的测试用例
- 每个 SOC 版本的特定实现代码都有测试覆盖
- SOC 通用代码（base）被所有版本测试覆盖
- 不同版本的差异逻辑都被测试覆盖

#### 6.4 SOC 版本功能差异示例

| 功能           | arch31 | arch32 | arch35 | 测试策略                  |
| -------------- | ------ | ------ | ------ | ------------------------- |
| A16W8 权重量化 | 不支持 | 支持   | 支持   | 为 arch32/arch35 设计测试 |
| FP8 权重量化   | 不支持 | 不支持 | 支持   | 为 arch35 设计测试        |
| A16W4 权重量化 | 不支持 | 支持   | 支持   | 为 arch32/arch35 设计测试 |
| FP8 PER_TENSOR | 不支持 | 不支持 | 不支持 | 为 arch35 设计失败测试    |

### 7. 编译运行命令

```bash
# 在容器内执行
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --soc='支持的SOC' --cov"

# 编译运行单类型 UT 并获取覆盖率报告
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --opapi --ops='算子名称' --soc='支持的SOC' --cov"
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --opkernel --ops='算子名称' --soc='支持的SOC' --cov"

# 编译运行所有类型 UT
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ops='算子名称' --soc='支持的SOC' --cov"
```

## 核心职责与原则

### 核心职责

1. 分析算子 UT 的代码覆盖率，识别未覆盖的代码路径
2. 设计并实现补充测试用例，提升覆盖率
3. **针对不同 SOC 版本设计测试用例，测试用例全覆盖, 如果无法实现100%覆盖率，详细说明具体函数/分支/文件无法覆盖原因**
4. 诊断并修复测试失败问题，确保测试用例有效
5. **验证测试用例的有效性，确保新增用例能编译通过且真正提升覆盖率**
6. 生成详细的覆盖率提升报告，包含 SOC 版本覆盖率统计

### 核心原则 ⭐

> **UT 覆盖率提升 ≠ 盲目增加测试用例数量**
>
> 目标是通过有针对性的测试用例设计，覆盖关键代码路径和边界条件

**测试设计原则**:

| 原则         | 说明                 | 示例                         |
| ------------ | -------------------- | ---------------------------- |
| **边界优先** | 优先测试边界值和极值 | 最小/最大形状、0值、空tensor |
| **路径完整** | 覆盖所有代码分支     | if/else各分支、异常处理路径  |
| **场景多样** | 测试不同参数组合     | 不同数据类型、不同shape组合  |
| **独立性**   | 新增用例独立可运行   | 不依赖其他用例的执行结果     |
| **可验证**   | 测试结果可预期和验证 | 明确期望的返回值和tiling key |

**常见误区**:

| 误区                 | 正确做法                           |
| -------------------- | ---------------------------------- |
| 随意增加测试数据大小 | 分析未覆盖代码，有针对性设计       |
| 只关注行覆盖率       | 同时关注分支覆盖率和条件覆盖率     |
| 忽略异常路径         | 补充错误处理和边界条件测试         |
| 修改原测试用例       | 新增用例独立添加，不破坏原有测试   |
| 期望值设置错误       | 仔细分析代码逻辑，正确设置期望结果 |

## 各阶段详细指南

### 阶段一：需求充分性检查

在开始 UT 覆盖率增强前，确认以下信息是否完整：

#### 必需信息（缺失则进入 Interview 模式）

- [ ]  **算子名称**：明确要增强哪个算子的 UT 覆盖率
- [ ]  **UT 类型**：ophost、opapi、opkernel 或全部
- [ ]  **目标覆盖率**（可选）：期望达到的覆盖率目标

#### Interview 模式触发条件

当用户输入满足以下任一条件时，使用 **AskUserQuestion** 工具主动提问：

1. **缺少算子名称**：未指定要处理哪个算子
2. **缺少 UT 类型**：未指定要处理哪种类型的测试
3. **信息模糊**：如"提升覆盖率"但未说明具体算子和类型

### 阶段二：覆盖率配置检查

在运行 UT 前，**必须检查**覆盖率配置是否正确。

#### 检查 test_config.yaml

查看算子在 `tests/test_config.yaml` 中的配置，确认 `ut_cov_exclude` 没有排除需要统计覆盖率的源码目录。

### 阶段三：运行初始UT并获取覆盖率

#### 运行 UT 并保存日志

```bash
# 在容器内运行 UT 并获取覆盖率
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --cov" > /home/y00845930/ascendc-agent/output/算子名/original_cov.log 2>&1
```

#### 解析覆盖率报告

从日志中提取覆盖率信息：
- 总体行覆盖率
- 各文件的行覆盖率
- 函数覆盖率

### 阶段四：分析未覆盖代码 ⭐ 关键阶段

#### 快速分析方法 ⭐ v3.0 更新

**方法1: 覆盖率报告直接分析**
```bash
# 提取未覆盖的行
lcov --list coverage.info | grep ":0"
```

**方法2: 源代码静态扫描**
```bash
# 查找场景检测函数
grep -rn "IsCapable\|IsSupported\|CheckInput\|Validate" op_host/

# 查找条件分支
grep -rn "if.*return\|#ifdef.*SOC" op_host/
```

**方法3: Tiling Key 分析**
```bash
# 提取所有 tiling key
grep -rn "return.*UL\|return.*ULL" op_host/ | grep -v "return false"
```

#### 未覆盖代码分类

| 类别 | 优先级 | 处理策略 |
|------|--------|----------|
| 场景检测分支 | 高 | 设计参数触发该场景 |
| 边界条件分支 | 中 | 使用边界值测试 |
| 错误处理分支 | 中 | 构造非法输入 |
| 硬件依赖分支 | 低 | 记录为无法覆盖 |
| 死代码/预留代码 | 低 | 记录为设计限制 |

#### 场景检测逻辑分析 ⭐⭐⭐

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

#### SOC 版本差异化分析 ⭐⭐⭐

**分析维度**:

| 分析项           | 说明                              | 优先级 |
| ---------------- | --------------------------------- | ------ |
| **SOC 特定代码** | 每个版本独立的实现文件            | ⭐⭐⭐ |
| **版本功能差异** | 不同版本支持的功能不同            | ⭐⭐⭐ |
| **版本限制差异** | 不同版本的限制条件不同            | ⭐⭐   |
| **条件编译代码** | #ifdef SOC_VERSION 条件编译的代码 | ⭐⭐   |

**分析步骤**:

1. **识别 SOC 版本特定文件**:
   ```bash
   ls mc2/算子名/op_host/op_tiling/
   # 输出：arch31/ arch32/ arch35/ 等
   ```

2. **分析每个版本的未覆盖代码**:
   ```bash
   grep -A 20 "arch32/" coverage.info
   ```

3. **识别版本差异功能**:
   - 阅读各版本的实现文件
   - 比对不同版本的功能支持
   - 找出版本独有功能

4. **设计版本特定测试**:
   - 为版本独有功能设计专门测试
   - 为版本限制设计边界测试

### 阶段五：设计补充测试用例

#### 测试场景补充策略

根据 UT 类型，采用不同的补充策略：

##### ophost 测试场景

| 场景类别        | 测试内容             | 示例                         |
| --------------- | -------------------- | ---------------------------- |
| **Tiling 参数** | 不同输入形状组合     | 小/中/大shape、不规则shape   |
| **InferShape**  | 各种输入形状推导     | 动态维度、rank变化、空tensor |
| **多卡场景**    | 分布式参数           | 不同rank、不同通信模式       |
| **权重量化**    | A16W8/A16W4/FP8 场景 | 不同量化类型、不同反量化类型 |

##### opapi 测试场景

| 场景类别      | 测试内容          | 示例                    |
| ------------- | ----------------- | ----------------------- |
| **参数验证**  | 空指针、无效参数  | nullptr检查、越界检查   |
| **Workspace** | workspace大小获取 | 最小/最大workspace      |
| **正常流程**  | 各种数据类型      | float16、float32、int32 |

##### opkernel 测试场景

| 场景类别     | 测试内容     | 示例                     |
| ------------ | ------------ | ------------------------ |
| **计算逻辑** | 不同数据类型 | fp32、fp16、int8         |
| **边界值**   | 特殊值测试   | 0、NaN、Inf、最大/最小值 |
| **性能路径** | 不同数据量   | 小数据量 vs 大数据量     |

#### 测试用例设计模板 ⭐ v3.0 新增

**模板1: 基础场景**
```cpp
{"算子名_基础_数据类型",
 TD({{M, K}, {M, K}}, aType, FORMAT_ND),
 TD({{K, N}, {K, N}}, bType, FORMAT_ND),
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 ge::GRAPH_SUCCESS, tiling_key}
```

**模板2: 量化场景**
```cpp
{"算子名_量化_A16W8_PER_TENSOR",
 TD({{M, K}, {M, K}}, DT_FLOAT16, FORMAT_ND),
 TD({{K, N}, {K, N}}, DT_INT8, FORMAT_ND),
 TD({{N}, {N}}, DT_FLOAT16, FORMAT_ND),  // antiQuantScale
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 std::nullopt, 0,  // groupSize=0 表示 PER_TENSOR
 ge::GRAPH_SUCCESS, quant_tiling_key}
```

**模板3: 异常场景**
```cpp
{"算子名_异常_维度不匹配",
 TD({{1, 1}, {1, 1}}, DT_FLOAT16, FORMAT_ND),
 TD({{2, 2}, {2, 2}}, DT_FLOAT16, FORMAT_ND),
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 ge::GRAPH_FAILED, 4294967295UL}
```

#### 参数设计工具箱 ⭐ v3.0 新增

**数据类型组合**
```cpp
// FP16/BF16 + FP32 输出
aType = DT_FLOAT16, bType = DT_FLOAT32, cType = DT_FLOAT32

// 量化组合
aType = DT_FLOAT16, bType = DT_INT8 (A16W8)
aType = DT_INT8, bType = DT_INT8 (A8W8)
```

**维度对齐**
```cpp
// 不满足对齐 (触发对齐检查)
K = 65, N = 17 (对齐要求为 16)

// 满足对齐
K = 64, N = 16
```

**通信参数**
```cpp
// Rank Size 选项
rankSize = 2  // 最小多卡
rankSize = 4  // 中等规模
rankSize = 8  // 大规模
```

#### Tiling Key 推断方法 ⭐ v3.0 新增

**方法1: 从测试输出提取**
```bash
bash build.sh -u --ophost --ops='算子名' 2>&1 | \
    grep -oP "actualTilingKey: \K[0-9]+"
```

**方法2: 从源码分析**
```bash
# 查找 tiling key 定义
grep -rn "return [0-9]*UL" op_tiling/ | grep -v "return false"
```

**方法3: 启发式规则**
```cpp
// 非量化场景
非量化_tiling_key = 基础值 + 格式偏移 + SOC偏移

// 量化场景
量化_tiling_key = 基础值 + 量化模式偏移 + SOC偏移
```

#### 期望值设置指南 ⭐⭐⭐

```cpp
// 成功场景的期望值设置
ge::GRAPH_SUCCESS, correct_tiling_key

// 失败场景的期望值设置
ge::GRAPH_FAILED, 4294967295UL  // GRAPH_FAILED = 4294967295

// 注意: 即使触发错误路径，如果代码返回了 GRAPH_SUCCESS，测试也会失败
// 需要仔细分析代码逻辑，正确设置期望值
```

#### PER_CHANNEL vs PER_GROUP 判断逻辑 ⭐⭐⭐

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

### 阶段六：实现测试用例

#### 添加测试用例

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

#### 测试用例命名规范

```cpp
// 命名格式：{场景描述}_{格式/状态}_{期望结果}
a16w8_per_tensor_nd_success      // A16W8 PER_TENSOR + ND 格式 + 期望成功
a16w8_per_group_not_aligned_fail // A16W8 PER_GROUP + 未对齐 + 期望失败
```

### 阶段七：运行并验证测试 ⭐⭐⭐

#### 编译运行更新后的 UT

```bash
# 在容器内执行测试
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --soc='支持的SOC' --cov"
```

#### 测试失败诊断 ⭐⭐⭐

**常见失败原因及处理方法**:

| 失败类型              | 错误信息                                        | 原因分析                     | 处理方法                   |
| --------------------- | ----------------------------------------------- | ---------------------------- | -------------------------- |
| **Tiling key 不匹配** | `Expected: 4390924, Actual: 6488076`            | 使用的 tiling key 与场景不符 | 从测试输出提取实际 key     |
| **期望值错误**        | `Expected: GRAPH_FAILED, Actual: GRAPH_SUCCESS` | 期望值设置错误               | 分析代码逻辑，修正期望值   |
| **场景未触发**        | 测试通过但未覆盖目标代码                        | 参数组合未触发目标场景       | 检查场景检测条件，调整参数 |
| **参数校验失败**      | `CheckInput failed`                             | 参数不符合校验规则           | 修正参数或设计为失败测试   |

**Tiling Key 提取方法**:

```bash
# 从测试输出中提取实际的 tiling key
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='算子名称' --soc='支持的SOC' 2>&1" | grep "actualTilingKey\|Which is: [0-9]"
```

**测试迭代优化流程**:

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

### 阶段八：生成覆盖率提升报告

#### 报告内容要求

完整的覆盖率提升报告应包含：

##### 1. 测试信息

- 算子名称和测试类型
- 测试时间和执行环境
- 测试命令

##### 2. 测试结果

- 通过/失败的测试用例数量
- 失败测试用例的处理说明

##### 3. 覆盖率数据

- 初始覆盖率百分比
- 最终覆盖率百分比
- 覆盖率提升幅度
- 各文件的覆盖率对比

##### 4. 补充的测试用例

- 新增测试用例列表
- 每个测试用例覆盖的场景说明

##### 5. 提升要点总结

- 关键突破点说明
- 技术难点和解决方案
- 经验教训和最佳实践

#### 报告格式模板

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

---

# 第三部分：高级主题

## 批量处理模式 ⭐ v3.0 新增

### 适用场景

- 需要同时处理多个算子的覆盖率增强
- 需要生成整体覆盖率汇总报告
- 需要跟踪批量处理进度

### 使用方法

#### 方式1: 指定算子列表

```bash
ut-coverage-enhancer --batch --operators="op1,op2,op3"
```

#### 方式2: 指定算子目录

```bash
ut-coverage-enhancer --batch --dir=mc2/
```

#### 方式3: 使用配置文件

```bash
ut-coverage-enhancer --batch --config=operators.yaml
```

### 执行流程

```
批量处理流程:
├── 预检查阶段
│   ├── 检查所有算子目录是否存在
│   ├── 检查测试文件完整性
│   └── 估算处理时间
│
├── 处理阶段
│   ├── 串行处理每个算子
│   ├── 显示当前进度 (x/总数)
│   └── 保存中间结果
│
└── 汇总阶段
    ├── 生成汇总报告
    ├── 生成覆盖率对比表
    └── 生成批量处理日志
```

### 输出结果

```
output/
├── batch_summary.md         # 批量处理汇总报告
├── coverage_comparison.md    # 覆盖率对比表格
├── batch_log.txt           # 批量处理日志
└── [算子名]/v1/            # 各算子的详细报告
```

## 环境自适应处理 ⭐ v3.0 新增

### 环境检测流程

```
环境检测:
├── 检查1: 容器是否运行
│   └── docker ps | grep y00845930
│
├── 检查2: 编译环境是否完整
│   ├── 检查 CANN 版本
│   ├── 检查依赖库 (OPBASE)
│   └── 尝试编译测试用例
│
└── 决策点
    ├── 环境完整 → 实际测试模式
    │   ├── 运行测试获取实际覆盖率
    │   ├── 诊断并修复测试失败
    │   └── 生成准确覆盖率报告
    │
    └── 环境缺失 → 静态分析模式
        ├── 分析源代码识别场景
        ├── 设计测试用例
        ├── 预估覆盖率
        └── 生成预估报告
```

### 静态分析模式

当编译环境不可用时，使用静态代码分析：

| 分析项 | 方法 | 产出 |
|--------|------|------|
| 场景识别 | 查找 IsCapable/CheckInput 等函数 | 场景列表 |
| 未覆盖行 | 分析 if/else 分支 | 未覆盖代码列表 |
| 参数设计 | 解析触发条件 | 测试参数组合 |
| 覆盖率预估 | 启发式计算 | 预期覆盖率 |

### 模式切换标志

在报告中明确标识使用的模式：
- `[实际测试]` - 环境完整，有实际测试数据
- `[静态分析]` - 环境缺失，基于代码分析的预估

## 进度跟踪与状态显示 ⭐ v3.0 新增

### 进度显示格式

```
[UT Coverage Enhancement]
正在处理: 12/24 (50%)
当前算子: matmul_all_reduce
状态: 分析未覆盖代码...
预计剩余时间: 15 分钟
```

### 状态定义

| 状态 | 描述 | 颜色 |
|------|------|------|
| 🔵 待处理 | 在队列中等待 | 蓝色 |
| 🟡 处理中 | 正在处理 | 黄色 |
| 🟢 已完成 | 处理成功 | 绿色 |
| 🔴 已失败 | 处理失败 | 红色 |
| ⚪ 已跳过 | 算子不存在或无需处理 | 灰色 |

### 断点续传

```
如果批量处理中断，可以从中断处继续：

ut-coverage-enhancer --batch --resume

将从上次的进度继续处理已完成的算子会被跳过
```

---

# 第四部分：经验总结

## 覆盖率提升技巧与最佳实践 ⭐⭐⭐

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
antiQuantScale != 0   // 非空 ✓
```

### 2. 参数组合设计技巧

| 技巧         | 说明                           | 示例                           |
| ------------ | ------------------------------ | ------------------------------ |
| **格式选择** | 优先使用 ND 格式，避免维度限制 | FORMAT_ND vs FORMAT_FRACTAL_NZ |
| **指针参数** | 必须满足非空/为空的要求        | antiQuantScale != nullptr      |
| **数值范围** | 边界值要能触发检查             | groupSize < 32 触发错误        |
| **类型匹配** | 数据类型必须符合场景要求       | FP16 + INT8 触发 A16W8         |

### 3. 调试技巧

**使用日志验证场景触发**:

```bash
# 设置环境变量启用日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0

# 运行测试并过滤日志
bash build.sh -u --ophost --ops='算子名称' --soc='支持的SOC' 2>&1 | grep "场景关键词"
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

| 实践            | 说明                     | 效果               |
| --------------- | ------------------------ | ------------------ |
| **场景优先**    | 先分析场景检测逻辑       | 提高测试用例针对性 |
| **ND 格式优先** | 使用 ND 格式避免维度限制 | 触发更多代码路径   |
| **参数验证**    | 仔细检查参数组合         | 确保场景正确触发   |
| **期望值谨慎**  | 分析代码逻辑设置期望值   | 避免期望值错误     |
| **迭代优化**    | 逐步验证和修正           | 提高测试用例质量   |
| **日志调试**    | 使用日志验证场景         | 快速定位问题       |

## 常见问题与处理

### 问题1：测试用例失败 - Tiling Key 不匹配

**错误信息**:

```
Expected equality of these values:
  tilingKey
    Which is: 6488076
  expectTilingKey
    Which is: 4390924
```

**处理方法**:

1. 从测试输出中提取实际的 tiling key
2. 理解不同场景对应的 tiling key
3. 更新测试用例中的 tiling key

### 问题2：测试用例失败 - 期望值错误

**错误信息**:

```
Expected equality of these values:
  tilingRet (0)
  expectResult (4294967295)
```

**处理方法**:

1. 阅读源代码，分析该场景应该返回什么
2. 检查是否触发错误检查
3. 确认是否有提前返回
4. 修正期望值

**判断方法**:

- 如果代码中有 `OP_TILING_CHECK` 且条件满足 → 期望 GRAPH_FAILED
- 如果所有检查都通过 → 期望 GRAPH_SUCCESS

### 问题3：测试用例通过但覆盖率未提升

**处理方法**:

1. 检查参数是否满足场景检测条件
2. 使用调试日志验证场景是否触发
3. 分析代码是否可达

### 问题4：测试框架限制 - 维度不匹配

**错误信息**:

```
Expect x2 dim to be 4, but got x2 dim:[2]
```

**处理方法**:

- 使用 ND 格式替代 FRACTAL_NZ
- 或说明无法测试该场景

## 重要注意事项

### 1. 场景检测分析优先原则 ⭐⭐⭐

> **始终先分析场景检测逻辑，再设计测试用例**

### 2. 测试框架限制原则 ⭐⭐

> **了解测试框架的限制，避免设计无法执行的测试用例**

### 3. 期望值设置谨慎原则 ⭐⭐⭐

> **仔细分析代码逻辑，正确设置期望值**

### 4. 增量开发与迭代原则

> **先实现一个测试场景，逐步验证和迭代**

### 5. 文件操作安全原则 ⭐⭐⭐

> **在宿主机上修改文件，然后复制到容器执行测试**

---

# 第五部分：参考资料

## 测试用例模板库

### 模板目录结构

```
templates/
├── basic_test.cpp.template       # 基础测试模板
├── quant_test.cpp.template       # 量化测试模板
├── error_test.cpp.template       # 异常测试模板
├── soc_specific_test.cpp.template # SOC特定测试模板
└── boundary_test.cpp.template    # 边界测试模板
```

### 基础测试模板

```cpp
{"算子名_基础_数据类型",
 TD({{M, K}, {M, K}}, aType, FORMAT_ND),
 TD({{K, N}, {K, N}}, bType, FORMAT_ND),
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 ge::GRAPH_SUCCESS, tiling_key}
```

### 量化测试模板

```cpp
{"算子名_量化_A16W8_PER_TENSOR",
 TD({{M, K}, {M, K}}, DT_FLOAT16, FORMAT_ND),
 TD({{K, N}, {K, N}}, DT_INT8, FORMAT_ND),
 TD({{N}, {N}}, DT_FLOAT16, FORMAT_ND),  // antiQuantScale
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 std::nullopt, 0,  // groupSize=0 表示 PER_TENSOR
 ge::GRAPH_SUCCESS, quant_tiling_key}
```

### 异常测试模板

```cpp
{"算子名_异常_维度不匹配",
 TD({{1, 1}, {1, 1}}, DT_FLOAT16, FORMAT_ND),
 TD({{2, 2}, {2, 2}}, DT_FLOAT16, FORMAT_ND),
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
 ge::GRAPH_FAILED, 4294967295UL}
```

## 辅助脚本集合

### 环境检查脚本

```bash
#!/bin/bash
# check_env.sh - 环境检查脚本

echo "=== UT Coverage Enhancement 环境检查 ==="

# 检查容器
if ! docker ps | grep -q y00845930; then
    echo "❌ 容器未运行"
    exit 1
fi
echo "✅ 容器状态: 运行中"

# 检查编译环境
if docker exec y00845930 bash -c "
    cd /ascendc-agent/ops-transformer
    bash build.sh -u --ophost --ops='test' 2>&1 | head -1
" | grep -q "error"; then
    echo "⚠️  编译环境不完整，将使用静态分析模式"
    exit 0
fi
echo "✅ 编译环境: 完整"

echo "=== 环境检查完成 ==="
```

### 覆盖率提取脚本

```bash
#!/bin/bash
# extract_coverage.sh - 提取覆盖率数据

#!/bin/bash
# extract_coverage.sh - 提取覆盖率数据

echo "=== 提取覆盖率数据 ==="

# 提取总体覆盖率
echo "总体覆盖率:"
lcov --summary coverage.info | grep "lines"

# 提取未覆盖的行
echo "未覆盖代码行:"
lcov --list coverage.info | grep ":0"

# 按文件提取覆盖率
echo "各文件覆盖率:"
lcov --list coverage.info | grep -E "\.cpp:"

echo "=== 提取完成 ==="
```

### 批量处理脚本

```bash
#!/bin/bash
# batch_process.sh - 批量处理脚本

OPERATORS=$1  # 算子列表，逗号分隔
OUTPUT_DIR=$2 # 输出目录

IFS=',' read -ra OP_ARRAY <<< "$OPERATORS"
TOTAL=${#OP_ARRAY[@]}

for i in "${!OP_ARRAY[@]}"; do
    OP=${OP_ARRAY[$i]}
    echo "[$((i+1))/$TOTAL] 处理算子: $OP"

    # 处理算子
    ut-coverage-enhancer --operator="$OP" --output="$OUTPUT_DIR/$OP"

    echo "完成: $OP"
done

echo "批量处理完成"
```

## 常用命令参考

### 容器操作

```bash
# 进入容器
docker exec -it y00845930 bash

# 复制文件到容器
docker cp 本地文件 y00845930:/容器路径/

# 复制文件从容器
docker cp y00845930:/容器路径/ 本地路径

# 查看容器日志
docker logs y00845930
```

### 编译测试

```bash
# 单个算子 ophost 测试
bash build.sh -u --ophost --ops='算子名' --soc='SOC版本' --cov

# 单个算子 opapi 测试
bash build.sh -u --opapi --ops='算子名' --soc='SOC版本' --cov

# 单个算子 opkernel 测试
bash build.sh -u --opkernel --ops='算子名' --soc='SOC版本' --cov
```

### 覆盖率分析

```bash
# 生成覆盖率报告
lcov --capture --directory build --output-file coverage.info

# 查看覆盖率摘要
lcov --summary coverage.info

# 生成HTML报告
genhtml coverage.info --output-directory coverage_html

# 查找未覆盖代码
lcov --list coverage.info | grep ":0"
```

### 调试命令

```bash
# 设置日志环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0

# 提取 tiling key
grep -oP "actualTilingKey: \K[0-9+" 测试日志

# 查找场景检测函数
grep -rn "IsCapable\|IsSupported\|CheckInput" op_host/

# 查找条件分支
grep -rn "if.*return\|#ifdef.*SOC" op_host/
```

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
  y00845930:/ascendc-agent/ops-transformer/mc2/matmul_all_reduce/tests/ut/op_host/

# 5. 运行测试（在容器内）
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='matmul_all_reduce' --soc='支持的SOC' --cov"

# 6. 提取 tiling key（如果需要）
docker exec y00845930 bash -c "cd /ascendc-agent/ops-transformer && bash build.sh -u --ophost --ops='matmul_all_reduce' --soc='支持的SOC' 2>&1" | grep -E "actualTilingKey|Which is: [0-9]"

# 7. 复制结果回宿主机
docker cp y00845930:/ascendc-agent/ops-transformer/build/cov_result/coverage.info \
  /home/y00845930/ascendc-agent/output/

# 8. 生成报告
# 手动编写或使用工具生成覆盖率提升报告
```

---

**文档维护**: UT Coverage Enhancer Skill Team
**最后更新**: 2026-02-02
**版本**: v3.0
