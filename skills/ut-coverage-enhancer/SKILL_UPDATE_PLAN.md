# SKILL v3.0 更新建议

**基于项目**: ops-transformer mc2 算子批量覆盖率增强 (24个算子)
**更新时间**: 2026-02-02

---

## 一、SKILL.md 需要更新的章节

### 1.1 新增章节建议

#### 新增1: 批量处理模式

```markdown
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
```

#### 新增2: 环境自适应模式

```markdown
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
```

#### 新增3: 进度跟踪

```markdown
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
```

### 1.2 需要更新的现有章节

#### 更新1: 环境和前置条件

```markdown
## 环境和前置条件

### 1. 工作目录与执行环境 ⭐ v3.0 更新

**宿主机环境**:
- **工作目录**: `/home/y00845930/ascendc-agent/ops-transformer/`
- **输出目录**: `/home/y00845930/ascendc-agent/output/`

**容器环境** (用于编译和测试):
- **容器名称**: `y00845930`
- **容器内路径**: `/ascendc-agent/ops-transformer/`
- **进入方式**: `docker exec y00845930 bash -c "命令"`

### 环境检测 ⭐ v3.0 新增

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

### 容器资源管理 ⭐ v3.0 新增

| 资源 | 建议 | 说明 |
|------|------|------|
| 磁盘空间 | 定期清理 build/ 目录 | 覆盖率数据文件较大 |
| 内存使用 | 每5-10个算子重启容器 | 避免内存泄漏累积 |
| CPU占用 | 串行处理降低负载 | 避免并行编译冲突 |
```

#### 更新2: 各阶段详细指南

```markdown
### 阶段四：分析未覆盖代码 ⭐ v3.0 更新

#### 快速分析方法

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
```

#### 更新3: 设计补充测试用例

```markdown
### 阶段五：设计补充测试用例 ⭐ v3.0 更新

#### 测试用例设计模板 ⭐ 新增

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

#### 参数设计工具箱 ⭐ 新增

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

#### Tiling Key 推断方法 ⭐ 新增

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
```

---

## 二、SKILL.md 结构优化建议

### 2.1 当前结构问题

| 问题 | 描述 | 影响 |
|------|------|------|
| 文档过长 | SKILL.md 42KB+ | 难以快速定位信息 |
| 重点不突出 | 核心经验分散 | 学习曲线陡峭 |
| 缺少快速入门 | 没有快速参考 | 新手上手慢 |

### 2.2 优化后的结构建议

```markdown
# UT 覆盖率增强技能指南 v3.0

## 第一部分: 快速开始 (新增)
- 5分钟快速入门
- 常用命令速查
- 典型问题快速索引

## 第二部分: 完整指南 (原有内容重组)
- 工作流程
- 环境和前置条件
- 各阶段详细指南

## 第三部分: 高级主题 (新增)
- 批量处理模式
- 环境自适应处理
- 性能优化技巧

## 第四部分: 经验总结 (原有内容)
- 覆盖率提升技巧
- 常见问题与处理
- 最佳实践

## 第五部分: 参考资料 (新增)
- 模板代码库
- 工具脚本集合
- 相关文档链接
```

### 2.3 快速参考卡片

```markdown
## 快速参考 ⭐ 新增

### 常用命令

```bash
# 单个算子处理
ut-coverage-enhancer --operator=算子名 --ut-type=ophost

# 批量处理
ut-coverage-enhancer --batch --dir=mc2/

# 生成汇总报告
ut-coverage-enhancer --summary --output=report.md
```

### 覆盖率目标

| 算子类型 | 目标覆盖率 | 关键点 |
|---------|-----------|--------|
| 简单算子 | > 90% | 基础场景全覆盖 |
| 复杂算子 | > 80% | 主要分支覆盖 |
| 依赖算子 | > 70% | 核心逻辑覆盖 |

### 快速诊断

| 问题 | 检查命令 | 解决方法 |
|------|---------|----------|
| 编译失败 | bash build.sh -u --ophost --ops='op' | 检查依赖 |
| 测试失败 | 查看测试日志 | 提取实际 key |
| 覆盖率不提升 | 检查 ut_cov_exclude | 修改配置 |
```

---

## 三、模板和脚本建议

### 3.1 测试用例模板库

创建 `templates/` 目录，存放各种测试用例模板：

```
templates/
├── basic_test.cpp.template       # 基础测试模板
├── quant_test.cpp.template       # 量化测试模板
├── error_test.cpp.template       # 异常测试模板
├── soc_specific_test.cpp.template # SOC特定测试模板
└── boundary_test.cpp.template    # 边界测试模板
```

### 3.2 脚本工具集

创建 `scripts/` 目录，存放辅助脚本：

```
scripts/
├── check_env.sh                 # 环境检查脚本
├── run_coverage.sh              # 运行覆盖率测试
├── extract_coverage.sh          # 提取覆盖率数据
├── generate_summary.sh          # 生成汇总报告
└── batch_process.sh             # 批量处理脚本
```

### 3.3 示例脚本

```bash
#!/bin/bash
# scripts/check_env.sh - 环境检查脚本

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

---

## 四、更新优先级

### 4.1 高优先级 (立即更新)

| 更新项 | 原因 | 预计工作量 |
|--------|------|-----------|
| 环境自适应模式 | 解决环境依赖问题 | 2 小时 |
| 批量处理模式 | 提升批量处理效率 | 4 小时 |
| 快速参考章节 | 改善用户体验 | 1 小时 |

### 4.2 中优先级 (短期更新)

| 更新项 | 原因 | 预计工作量 |
|--------|------|-----------|
| 进度跟踪功能 | 提升批量处理体验 | 3 小时 |
| 模板库创建 | 标准化测试用例 | 4 小时 |
| 脚本工具集 | 自动化常用操作 | 6 小时 |

### 4.3 低优先级 (长期规划)

| 更新项 | 原因 | 预计工作量 |
|--------|------|-----------|
| 智能测试生成 | 提升自动化水平 | 20 小时 |
| 可视化报告 | 改善报告呈现 | 10 小时 |
| 增量分析 | 支持覆盖率对比 | 8 小时 |

---

## 五、实施建议

### 5.1 分阶段实施

**第一阶段 (本周)**
1. 添加环境自适应模式
2. 创建快速参考章节
3. 添加进度跟踪基础功能

**第二阶段 (本月)**
1. 实现批量处理模式
2. 创建模板库
3. 开发脚本工具集

**第三阶段 (下季度)**
1. 智能测试生成
2. 可视化报告
3. 增量分析功能

### 5.2 兼容性考虑

- 保持向后兼容
- 新功能作为可选特性
- 提供迁移指南

### 5.3 测试计划

- 单元测试: 测试新增功能
- 集成测试: 测试批量处理流程
- 用户验收测试: 邀请用户试用新功能

---

## 六、总结

本次 24 个算子的批量覆盖率增强为 SKILL 的优化提供了宝贵的经验。主要改进方向包括：

1. **环境自适应**: 解决环境依赖问题
2. **批量处理**: 提升大规模处理效率
3. **标准化**: 模板化测试用例设计
4. **可视化**: 改善进度和结果呈现
5. **自动化**: 减少手工操作

建议按优先级逐步实施这些改进，持续提升 SKILL 的易用性和效率。

---

**文档维护**: UT Coverage Enhancer Skill Team
**最后更新**: 2026-02-02
