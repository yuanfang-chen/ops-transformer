# fused_k_rms_norm_rope_store_kv_cache_mx_quant 算子

## 一、算子调用使用指南

### 1.1 编译与安装

> **注意**: 以下示例中的 `${ASCEND_HOME}` 需替换为实际的 CANN 安装路径，如 `/usr/local/Ascend` 或 `/home/xxx/Ascend`。

```bash
# 0. 设置基础 CANN 环境变量（替换 ${ASCEND_HOME} 为实际路径）
source ${ASCEND_HOME}/ascend-toolkit/set_env.sh

# 1. 编译算子包
bash build.sh --ops=fused_k_rms_norm_rope_store_kv_cache_mx_quant --soc=ascend950 --pkg

# 2. 安装算子包（替换 ${ASCEND_HOME} 为实际路径）
./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=${ASCEND_HOME}

# 3. 设置自定义算子包环境变量
source ${ASCEND_HOME}/vendors/custom_transformer/bin/set_env.bash

# 4. 编译 Torch 扩展
cd torch_extension/ && python3 setup.py bdist_wheel

# 5. 安装 Torch 扩展
cd dist && pip3 install --force-reinstall npu_ops_transformer-1.0.0-py3-none-any.whl
```

### 1.2 Torch 调用示例

> **注意**: 直接导入特定算子模块可避免加载其他无关算子，减少 import 时间。

```python
import torch
import torch_npu

# 直接导入特定算子，避免加载其他无关算子
from npu_ops_transformer.ops.fused_k_rms_norm_rope_store_kv_cache_mx_quant import (
    npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant
)

# 定义参数
T = 2048       # 序列长度
Nq = 20        # Query 头数
Nk = 2         # Key 头数
Nv = 2         # Value 头数
D = 128        # 头维度
Bs = 512       # PagedAttention block size
Bn = (T + Bs - 1) // Bs  # block 数量
N = Nq + Nk + Nv

# 创建输入张量
qkv = torch.randn(T, N, D, dtype=torch.bfloat16, device="npu")
cos = torch.randn(T, 1, D, dtype=torch.bfloat16, device="npu")
sin = torch.randn(T, 1, D, dtype=torch.bfloat16, device="npu")
gamma = torch.randn(D, dtype=torch.float32, device="npu").abs() + 0.1

# slot mapping
kv_slot_mapping = torch.randperm(T, dtype=torch.int64, device="npu")
v_scale_slot_mapping = torch.randperm(T // 32 // 2, dtype=torch.int64, device="npu")

# KV Cache (FP8)
k_cache = torch.zeros(Bn, Nk, Bs, D, dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)
k_scale_cache = torch.zeros(Bn, Nk, Bs, D // 32 // 2, 2, dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)
v_cache = torch.zeros(Bn, Nv, Bs, D, dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)
v_scale_cache = torch.zeros(Bn, Nv, Bs // 32 // 2, D, 2, dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)

# 调用融合算子
q, q_scale, k_cache_out, k_scale_cache_out, v_cache_out, v_scale_cache_out = \
    npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant(
        qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
        k_cache, k_scale_cache, v_cache, v_scale_cache, epsilon=1e-5
    )
```

---

## 二、Torch 接口参数说明

### 2.1 函数签名

```python
torch.ops.npu_ops_transformer.npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant(
    qkv, cos, sin, gamma,
    kv_slot_mapping, v_scale_slot_mapping,
    k_cache, k_scale_cache, v_cache, v_scale_cache,
    *, epsilon=1e-5
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
```

### 2.2 输入参数

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `qkv` | `[T, N, D]` | BF16 | QKV 拼接输入，其中 `N = Nq + Nk + Nv` |
| `cos` | `[T, 1, D]` | BF16 | RoPE 余弦编码 |
| `sin` | `[T, 1, D]` | BF16 | RoPE 正弦编码 |
| `gamma` | `[D]` | FP32 | RMSNorm 缩放权重 |
| `kv_slot_mapping` | `[T]` | INT64 | KV Cache slot 映射索引 |
| `v_scale_slot_mapping` | `[T//32//2]` | INT64 | V Scale Cache slot 映射索引 |
| `k_cache` | `[Bn, Nk, Bs, D]` | FP8_E4M3FN | Key Cache (inplace) |
| `k_scale_cache` | `[Bn, Nk, Bs, D//32//2, 2]` | FP8_E8M0FNU | Key Scale Cache (inplace) |
| `v_cache` | `[Bn, Nv, Bs, D]` | FP8_E4M3FN | Value Cache (inplace) |
| `v_scale_cache` | `[Bn, Nv, Bs//32//2, D, 2]` | FP8_E8M0FNU | Value Scale Cache (inplace) |

### 2.3 属性参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `epsilon` | float | 1e-5 | RMSNorm 防止除零的极小值 |

### 2.4 输出参数

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `q` | `[T, Nq, D]` | FP8_E4M3FN | 量化后的 Query 输出 |
| `q_scale` | `[T, Nq, D//32//2, 2]` | FP8_E8M0FNU | Query 量化 Scale |
| `k_cache` | `[Bn, Nk, Bs, D]` | FP8_E4M3FN | 更新后的 Key Cache |
| `k_scale_cache` | `[Bn, Nk, Bs, D//32//2, 2]` | FP8_E8M0FNU | 更新后的 Key Scale Cache |
| `v_cache` | `[Bn, Nv, Bs, D]` | FP8_E4M3FN | 更新后的 Value Cache |
| `v_scale_cache` | `[Bn, Nv, Bs//32//2, D, 2]` | FP8_E8M0FNU | 更新后的 Value Scale Cache |

### 2.5 符号说明

- `T`: 总序列长度
- `Nq`: Query 头数
- `Nk`: Key 头数
- `Nv`: Value 头数
- `D`: 头维度 (head_dim)
- `Bs`: PagedAttention block size
- `Bn`: block 数量，`Bn = ceil(T / Bs)`
- 量化 block size: 32 (MX Quantization)

### 2.6 算子限制

#### 数据类型限制

| 张量 | 支持的数据类型 |
|------|----------------|
| `qkv` | BF16 |
| `cos`, `sin` | BF16 |
| `gamma` | FP32 |
| `kv_slot_mapping`, `v_scale_slot_mapping` | INT64 |
| `k_cache`, `v_cache` | FP8_E4M3FN |
| `k_scale_cache`, `v_scale_cache` | FP8_E8M0FNU |
| `q` (输出) | FP8_E4M3FN |
| `q_scale` (输出) | FP8_E8M0FNU |

#### 维度取值限制

| 参数 | 限制条件 | 说明 |
|------|----------|------|
| `T` (序列长度) | 必须是 **64 的整数倍** | V 轴向量化对齐要求 |
| `D` (head_dim) | 仅支持 **128** | 当前版本固定值 |
| `Bs` (block_size) | 仅支持 **512** | 当前版本固定值 |

#### 头数配置限制

| 参数 | 支持的配置 |
|------|------------|
| `(Nq, Nk, Nv)` | 仅支持 **{(20, 2, 2), (80, 8, 8)}** 两种配置 |

#### 典型的网络配置

| 配置 | T | Nq | Nk | Nv | D | Bs | Bn |
|------|---|----|----|----|---|-----|-----|
| 配置1 | 2048 | 20 | 2 | 2 | 128 | 512 | 4 |
| 配置2 | 2048 | 80 | 8 | 8 | 128 | 512 | 4 |
| 配置3 | 10240 | 20 | 2 | 2 | 128 | 512 | 20 |
| 配置4 | 10240 | 80 | 8 | 8 | 128 | 512 | 20 |

> **注意**: T (序列长度) 可以是 64 的任意整数倍，上表仅列出典型值。

---

## 三、算子设计

### 3.1 功能概述

`fused_k_rms_norm_rope_store_kv_cache_mx_quant` 是一个高性能融合算子，将以下操作合并为单次 Kernel 执行：

1. **K RMSNorm**: 对 Key 进行 RMS 归一化
2. **RoPE**: 对 Q 和 K 应用旋转位置编码
3. **MX Quantization**: 将 BF16 数据量化为 FP8 (E4M3FN)，Scale 为 FP8 (E8M0FNU)
4. **KV Cache Scatter**: 将量化后的 K/V 数据按 slot_mapping 写入 PagedAttention Cache

### 3.2 计算流程

算子采用三阶段并行设计，充分利用 NPU 多核能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入: qkv [T, N, D]                       │
│                    cos [T, 1, D], sin [T, 1, D]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Phase 1     │  │     Phase 2     │  │     Phase 3     │
│   (Q 处理)       │  │   (K 处理)       │  │   (V 处理)       │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ 1. 提取 Q       │  │ 1. 提取 K       │  │ 1. 提取 V       │
│ 2. RoPE        │  │ 2. RMSNorm     │  │ 2. MXQuant     │
│ 3. MXQuant     │  │ 3. RoPE        │  │    (axis=0)    │
│    (axis=1)    │  │ 4. MXQuant     │  │ 3. Scatter to  │
│ 4. 输出 Q/Q_Scale│  │    (axis=1)    │  │    V_Cache     │
│                │  │ 5. Scatter to  │  │                │
│                │  │    K_Cache     │  │                │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  q [T,Nq,D]     │  │ k_cache [Bn,...]│  │ v_cache [Bn,...]│
│  q_scale [...]  │  │ k_scale_cache   │  │ v_scale_cache   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 3.3 关键子算子

#### 3.3.1 RMSNorm
对 K 进行 RMS 归一化，公式如下：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2 + \epsilon}} \cdot \gamma$$

#### 3.3.2 RoPE (Rotary Position Embedding)
旋转位置编码，将位置信息融入 Q 和 K：

$$\text{out}[0:D/2] = \text{in}[0:D/2] \cdot \cos[0:D/2] - \text{in}[D/2:D] \cdot \sin[0:D/2]$$
$$\text{out}[D/2:D] = \text{in}[D/2:D] \cdot \cos[D/2:D] + \text{in}[0:D/2] \cdot \sin[D/2:D]$$

#### 3.3.3 MX Quantization (OCP 格式)
基于 block 的 MX 量化，block size = 32：

- **Q/K 量化 (axis=1)**: 沿 D 维度分 block 量化
  - 输出 scale 形状: `[T, Nq/Nk, D//32//2, 2]`

- **V 量化 (axis=0)**: 沿 T 维度分 block 量化
  - 输出 scale 形状: `[T//32//2, Nv, D, 2]`

### 3.4 性能优化

1. **Regbase (SIMT) 优化**: 利用 arch35 架构的向量计算单元，实现高效的 BF16/FP32 转换和量化操作
2. **三阶段并行**: Q/K/V 处理分配到不同核组，最大化并行度
3. **Double Buffer**: 采用双缓冲机制，隐藏数据搬运延迟
4. **Block 对齐**: 输出 Scale 进行 32B 对齐，优化 Scatter 写入效率

### 3.5 支持的 SoC

- ascend950 (arch35)
