# MhcPost

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

### 算子功能

MhcPost是Manifold-Constraint Hyper-Connection (mHC)架构中的核心算子之一，用于实现Post Mapping和残差连接。该算子基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_l$进行Res Mapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。

### 计算公式

$$x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}$$

其中：
- $(H_{l}^{res})^{T} \times x_l$：表示对输入$x_l$使用转置后的$h_{res}$矩阵（双随机矩阵）进行矩阵乘法变换，实现Res Mapping
- $h_{l}^{out} \otimes H_{t}^{post}$：表示将Atten/MLP层的输出$h_{out}$与$h_{post}$变换矩阵进行逐元素相乘，实现Post Mapping
- 最后将两个结果相加，得到下一层的输入$x_{l+1}$

## 参数说明

### 输入参数

| 参数名 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:-------|:-----|:---------|:---------|:------------|:------------:|
| x | 待计算的张量，表示网络中mHC层的输入数据 | FLOAT16、BFLOAT16 | ND | [B,S,N,D]、[T,N,D] | √ |
| h_res | mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵 | FLOAT32 | ND | [B,S,N,N]、[T,N,N] | √ |
| h_out | Atten/MLP层的输出 | FLOAT16、BFLOAT16 | ND | [B,S,D]、[T,D] | √ |
| h_post | mHC的h_post变换矩阵 | FLOAT32 | ND | [B,S,N]、[T,N] | √ |

### 输出参数

| 参数名 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:-------|:-----|:---------|:---------|:------------|:------------:|
| out | 网络中mHC层的输出数据，作为下一层的输入。数据类型与x相同 | FLOAT16、BFLOAT16 | ND | [B,S,N,D]、[T,N,D] | √ |

## 约束说明

### 维度约束

1. 输入x的维度必须是3维[T,N,D]或4维[B,S,N,D]
2. 输入h_res的维度必须与x相同，为3维[T,N,N]或4维[B,S,N,N]
3. 输入h_out的维度必须是2维[T,D]或3维[B,S,D]
4. 输入h_post的维度必须与h_out相同，为2维[T,N]或3维[B,S,N]
5. 输出out的维度与输入x相同

### 数据类型约束

1. x和h_out的数据类型必须一致，支持FLOAT16或BFLOAT16
2. h_res和h_post的数据类型必须是FLOAT32
3. out的数据类型与x相同

### Shape匹配约束

1. x和h_res的维度数必须相等，且除最后两维外的所有维度必须相同
2. h_out和h_post的维度数必须相等，且所有维度必须相同
3. h_res的最后两维必须相等（N × N的方阵）
4. x的最后一维(D)必须与h_out的最后一维(D)相同
5. h_post的最后一维(N)必须与h_res的最后一维(N)相同
6. h_res和h_out除最后两维外的维度必须匹配