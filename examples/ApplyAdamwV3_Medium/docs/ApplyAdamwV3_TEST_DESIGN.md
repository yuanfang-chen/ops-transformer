# ApplyAdamwV3 测试设计文档

## 1. 概述

### 1.1 算子功能
ApplyAdamwV3 实现 AdamW 优化器功能，用于神经网络训练中的权重更新。

### 1.2 测试范围
- 功能测试：验证算子的基本计算逻辑
- 精度测试：验证不同数据类型的计算精度
- 边界测试：验证边界条件下的正确性
- 异常测试：验证参数校验和错误处理

## 2. 输入参数分析

### 2.1 参数列表

| 参数名 | 类型 | Shape | 数据类型 | 必选/可选 | 说明 |
|--------|------|-------|----------|----------|------|
| varRef | Tensor | [1-8维] | FP16/BF16/FP32 | 必选 | 权重，输入输出 |
| mRef | Tensor | 与varRef相同 | 与varRef相同 | 必选 | 一阶动量，输入输出 |
| vRef | Tensor | 与varRef相同 | 与varRef相同 | 必选 | 二阶动量，输入输出 |
| beta1Power | Tensor | [1] | 与varRef相同 | 必选 | β1^(t-1) |
| beta2Power | Tensor | [1] | 与varRef相同 | 必选 | β2^(t-1) |
| lr | Tensor | [1] | 与varRef相同 | 必选 | 学习率 |
| weightDecay | Tensor | [1] | 与varRef相同 | 必选 | 权重衰减系数 |
| beta1 | Tensor | [1] | 与varRef相同 | 必选 | β1 |
| beta2 | Tensor | [1] | 与varRef相同 | 必选 | β2 |
| eps | Tensor | [1] | 与varRef相同 | 必选 | ε |
| grad | Tensor | 与varRef相同 | 与varRef相同 | 必选 | 梯度 |
| maxGradNorm | Tensor | 与varRef相同 | 与varRef相同 | 可选 | AMSGrad用 |
| amsgrad | bool | - | - | 可选 | AMSGrad开关 |
| maximize | bool | - | - | 可选 | 梯度取反 |

### 2.2 参数依赖关系

| 依赖维度 | 依赖关系 |
|---------|---------|
| 数据类型 | 所有Tensor参数的数据类型必须一致 |
| Shape | m、v、grad的shape必须与varRef一致 |
| Shape | 标量参数(beta1Power等)的shape必须为[1] |
| 条件必选 | amsgrad=true时，maxGradNorm必选 |

## 3. 测试级别规划

### 3.1 L0 级别（门槛用例，ST）

**目标**：核心功能直通，覆盖所有参数

| 用例编号 | 测试目的 | 数据类型 | Shape | amsgrad | maximize |
|---------|---------|---------|-------|---------|----------|
| OP-ApplyAdamwV3-L0-FUNC-001 | 基础功能验证 | FP32 | [1024] | false | false |
| OP-ApplyAdamwV3-L0-FUNC-002 | FP16基础功能 | FP16 | [1024] | false | false |
| OP-ApplyAdamwV3-L0-FUNC-003 | BF16基础功能 | BF16 | [1024] | false | false |
| OP-ApplyAdamwV3-L0-FUNC-004 | AMSGrad模式 | FP32 | [1024] | true | false |
| OP-ApplyAdamwV3-L0-FUNC-005 | maximize模式 | FP32 | [1024] | false | true |
| OP-ApplyAdamwV3-L0-FUNC-006 | 多维Shape | FP32 | [64, 64] | false | false |
| OP-ApplyAdamwV3-L0-FUNC-007 | 3维Shape | FP32 | [16, 16, 16] | false | false |
| OP-ApplyAdamwV3-L0-FUNC-008 | 全参数组合 | FP32 | [512] | true | true |

### 3.2 L1 级别（功能/精度，ST）

**目标**：BC组合测试，正常+典型边界

#### 3.2.1 Shape 覆盖

| Shape类型 | 示例 |
|----------|------|
| 1维 | [1], [1024], [4096], [16384] |
| 2维 | [64, 64], [128, 128], [256, 256] |
| 3维 | [16, 16, 16], [32, 32, 32] |
| 4维 | [8, 8, 8, 8], [2, 3, 4, 5] |
| 8维 | [2, 2, 2, 2, 2, 2, 2, 2] |
| 边界 | [1], [2], 大Shape |

#### 3.2.2 数据类型覆盖

| 数据类型 | 用例数 |
|---------|-------|
| FP32 | 全覆盖 |
| FP16 | 全覆盖 |
| BF16 | 全覆盖 |

#### 3.2.3 属性组合覆盖

| amsgrad | maximize | 覆盖场景 |
|---------|----------|---------|
| false | false | 标准AdamW |
| false | true | 梯度上升 |
| true | false | AMSGrad |
| true | true | AMSGrad+梯度上升 |

#### 3.2.4 精度测试场景

| 数据类型 | diff_thd | pct_thd |
|----------|----------|---------|
| FP16 | 0.001 | 0.001 |
| BF16 | 0.004 | 0.004 |
| FP32 | 0.0001 | 0.0001 |

### 3.3 L2 级别（异常用例，UT）

**目标**：异常测试场景

| 用例编号 | 测试目的 | 预期结果 |
|---------|---------|---------|
| OP-ApplyAdamwV3-L2-EXC-001 | 空指针varRef | ACLNN_ERR_PARAM_NULLPTR |
| OP-ApplyAdamwV3-L2-EXC-002 | 空指针mRef | ACLNN_ERR_PARAM_NULLPTR |
| OP-ApplyAdamwV3-L2-EXC-003 | 空指针grad | ACLNN_ERR_PARAM_NULLPTR |
| OP-ApplyAdamwV3-L2-EXC-004 | amsgrad=true但maxGradNorm为空 | ACLNN_ERR_PARAM_NULLPTR |
| OP-ApplyAdamwV3-L2-EXC-005 | 不支持的数据类型INT8 | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-006 | m与varRef数据类型不一致 | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-007 | v与varRef数据类型不一致 | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-008 | grad与varRef shape不一致 | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-009 | beta1Power shape不为[1] | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-010 | lr shape不为[1] | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-011 | maxGradNorm与varRef dtype不一致 | ACLNN_ERR_PARAM_INVALID |
| OP-ApplyAdamwV3-L2-EXC-012 | maxGradNorm与varRef shape不一致 | ACLNN_ERR_PARAM_INVALID |

## 4. Phase 规划

### 4.1 Phase 1：基础功能开发与验证

**轨道A（算子代码 + UT）**：
- 实现基础Kernel（单TilingKey，FP32，无AMSGrad）
- 实现Tiling逻辑
- 实现aclnn接口
- 开发L2级别UT用例（异常场景）

**轨道B（ST用例）**：
- 开发L0基础用例（FP32，基础shape）
- 实现golden数据生成和精度比对

**汇合验证**：
- UT通过 + ST基础用例通过

### 4.2 Phase 2：功能完善开发与验证

**轨道A（算子代码 + UT）**：
- 添加FP16、BF16支持
- 添加AMSGrad模板
- 添加maximize支持
- 扩展UT覆盖

**轨道B（ST用例）**：
- 开发L1完整用例
- 覆盖所有数据类型
- 覆盖边界条件

**汇合验证**：
- UT覆盖率达标 + ST精度测试全部通过

### 4.3 Phase 3：性能优化开发与验证（可选）

**轨道A（算子代码 + UT）**：
- 性能优化（如需要）
- UT回归验证

**轨道B（ST用例）**：
- 性能测试用例
- 精度回归测试

## 5. 测试代码结构

### 5.1 ST 测试代码

```
tests/st/aclnn/
├── CMakeLists.txt
├── test_aclnn_apply_adamw_v3.cpp      # 主测试文件
└── golden_generator.py                  # Golden数据生成
```

### 5.2 UT 测试代码

```
tests/ut/op_host/
├── CMakeLists.txt
└── test_aclnn_apply_adamw_v3.cpp      # UT测试文件
```

## 6. 验收标准

### 6.1 功能验收
- 所有L0用例通过
- 所有L1用例通过

### 6.2 精度验收
- FP16: 双千分之一
- FP32: 双万分之一
- BF16: 双千分之一

### 6.3 UT验收
- 所有L2异常用例通过
- Host代码覆盖率达标

## 7. 测试数据准备

### 7.1 标准测试数据

| 参数 | 典型值 |
|------|-------|
| beta1Power | 0.9^t |
| beta2Power | 0.999^t |
| lr | 0.001 |
| weightDecay | 0.01 |
| beta1 | 0.9 |
| beta2 | 0.999 |
| eps | 1e-8 |

### 7.2 Golden 生成逻辑

参考 PyTorch AdamW 优化器实现生成 golden 数据：
```python
# 伪代码
def adamw_golden(var, m, v, grad, beta1, beta2, lr, weight_decay, eps, maximize):
    if maximize:
        grad = -grad
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    var = var - lr * m / (sqrt(v) + eps) - lr * weight_decay * var
    return var, m, v
```

## 8. 交付物

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 测试设计文档 | `docs/ApplyAdamwV3_TEST_DESIGN.md` | 本文档 |
| ST测试代码 | `tests/st/aclnn/` | ST测试实现 |
| UT测试代码 | `tests/ut/op_host/` | UT测试实现 |
| 测试用例表 | `cannbot_generated/ApplyAdamwV3-testcase-*.csv` | 测试用例列表 |
