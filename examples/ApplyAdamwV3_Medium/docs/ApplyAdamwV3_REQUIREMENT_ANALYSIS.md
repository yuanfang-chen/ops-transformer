# ApplyAdamwV3 需求文档

## 基本信息
| 项目 | 内容 |
|------|------|
| **算子名称** | ApplyAdamwV3 |
| **算子功能** | 实现AdamW优化器功能 |
| **仓规划** | ops-math/math/apply_adamw_v3 |
| **参考算子** | ops-nn/optim/apply_adam_w |

## 数学公式

### 梯度处理（maximize参数）
$$
g_t=\begin{cases}-g_t & \text{ if } maximize = true\\ g_t & \text{ if } maximize=false\end{cases}
$$

### 动量更新
$$
m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t}
$$

### 二阶矩更新
$$
v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
$$

### Beta幂次更新
$$
\beta_{1}^{t}=\beta_{1}^{t-1}\times\beta_{1}
$$
$$
\beta_{2}^{t}=\beta_{2}^{t-1}\times\beta_{2}
$$

### AMSGrad模式（可选）
$$
v_t=\begin{cases}\max(maxGradNorm, v_t) & \text{ if } amsgrad = true\\ v_t & \text{ if } amsgrad = false\end{cases}
$$

### 偏差修正
$$
\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}
$$
$$
\hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}
$$

### 权重更新
$$
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
$$

## 输入输出规格

### 输入参数
| 参数名 | 类型 | Shape | 数据类型 | 说明 |
|--------|------|-------|----------|------|
| varRef | aclTensor* | [1-8维] | FP16/BF16/FP32 | 待更新的权重，公式中的θ，**输入同时也是输出** |
| mRef | aclTensor* | 与varRef相同 | 与varRef相同 | 一阶动量估计，公式中的m，**输入同时也是输出** |
| vRef | aclTensor* | 与varRef相同 | 与varRef相同 | 二阶动量估计，公式中的v，**输入同时也是输出** |
| beta1Power | aclTensor* | [1] | 与varRef相同 | β1的(t-1)次幂 |
| beta2Power | aclTensor* | [1] | 与varRef相同 | β2的(t-1)次幂 |
| lr | aclTensor* | [1] | 与varRef相同 | 学习率η |
| weightDecay | aclTensor* | [1] | 与varRef相同 | 权重衰减系数λ |
| beta1 | aclTensor* | [1] | 与varRef相同 | 一阶矩估计的指数衰减率 |
| beta2 | aclTensor* | [1] | 与varRef相同 | 二阶矩估计的指数衰减率 |
| eps | aclTensor* | [1] | 与varRef相同 | 防止除零的小常数ε |
| grad | aclTensor* | 与varRef相同 | 与varRef相同 | 梯度数据，公式中的g_t |
| maxGradNormOptional | aclTensor* | 与varRef相同 | 与varRef相同 | 可选，保存v的最大值，amsgrad=true时必选 |

### 输出参数
| 参数名 | 类型 | Shape | 数据类型 | 说明 |
|--------|------|-------|----------|------|
| varRef | aclTensor* | 与输入相同 | 与输入相同 | 更新后的权重 |
| mRef | aclTensor* | 与输入相同 | 与输入相同 | 更新后的一阶动量 |
| vRef | aclTensor* | 与输入相同 | 与输入相同 | 更新后的二阶动量 |

### 属性参数
| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| amsgrad | bool | false | 是否使用AMSGrad变体 |
| maximize | bool | false | 是否对梯度取反（梯度上升） |

## 数据类型支持
- [x] FP16 (float16)
- [x] BF16 (bfloat16)
- [x] FP32 (float32)

## 精度要求
| 数据类型 | 精度标准 |
|----------|----------|
| FP16 | 双千分之一（2/1000） |
| FP32 | 双万分之一（2/10000） |
| BF16 | 双千分之一（2/1000） |

## 目标环境
| 项目 | 内容 |
|------|------|
| **服务器类型** | Ascend910B |
| **AI Core 数量** | 24 (AIV) |
| **架构代际** | arch35 (A3/A2系列) |

## 约束条件
1. **Shape约束**：
   - varRef支持1-8维度
   - m、v、grad的shape必须与varRef完全一致
   - beta1Power、beta2Power、lr、weightDecay、beta1、beta2、eps的shape必须为[1]

2. **Dtype约束**：
   - 所有输入张量的数据类型必须一致
   - 输出数据类型与输入相同

3. **amsgrad约束**：
   - 当amsgrad=true时，maxGradNormOptional参数必选
   - maxGradNormOptional的shape和dtype必须与varRef一致

4. **内存格式**：
   - 支持非连续Tensor
   - 数据格式支持ND

## 接口说明

### 两段式接口
```cpp
aclnnStatus aclnnApplyAdamwV3GetWorkspaceSize(
    aclTensor* varRef,
    aclTensor* mRef,
    aclTensor* vRef,
    const aclTensor* beta1Power,
    const aclTensor* beta2Power,
    const aclTensor* lr,
    const aclTensor* weightDecay,
    const aclTensor* beta1,
    const aclTensor* beta2,
    const aclTensor* eps,
    const aclTensor* grad,
    const aclTensor* maxGradNormOptional,
    bool amsgrad,
    bool maximize,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
);

aclnnStatus aclnnApplyAdamwV3(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream
);
```

## 错误码

| 错误码 | 说明 |
|--------|------|
| ACLNN_ERR_PARAM_NULLPTR | 传入参数为空指针；或amsgrad=true时maxGradNormOptional为空 |
| ACLNN_ERR_PARAM_INVALID | 数据类型不在支持范围；数据类型不一致；shape不一致；标量参数shape不为1 |

## 确定性计算
- aclnnApplyAdamwV3默认确定性实现

## 性能预期
- 不指定性能指标，不进行性能测试

## 与参考算子的差异说明
- 基于ops-nn/optim/apply_adam_w实现，功能完全相同
- 目标芯片仅支持ascend910b（参考算子支持ascend910_95）
- 算子仓位置从ops-nn迁移到ops-math

## 参考资料
- 参考算子：ops-nn/optim/apply_adam_w/
- 文档：ops-nn/optim/apply_adam_w/docs/aclnnApplyAdamw.md
