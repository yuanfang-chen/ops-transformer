# MLA Prolog V3 算子测试框架

## 文件结构

```text
pytest/
├── test.py                          # pytest 泛化测试用例运行主程序
├── testcases.py                     # 泛化测试用例入参配置
├── check_valid_param.py             # 入参检查及精度对比
├── mla_prolog_v3_cpu_ref.py         # CPU 侧算子逻辑实现获取 Golden，NPU 算子直调
├── aclnnMlaPrologV3RefNew.py        # 旧框架 CPU 参考实现（保留原样，不修改）
├── pytest.ini                       # 创建 ci 和 fuzz 的测试标记
```

## 概述

pytest 框架作为一个轻量化精度对比的测试框架，提供了简单化的验证方式及流程：

- 测试用例只需要执行一个 test.py 脚本，简化执行流程，并能够对测试用例进行泛化
- 支持自定义生成的 custom 包适配，可通过直接调用 custom 包验证，无需交付件
- 脚本内各功能分离，易于特性验证及自定义修改
- Golden 生成逻辑完善，为先构建输入，后进行运算
- 环境配置较简单，无依赖冲突，仅需 python 环境及 torch & torch_npu 安装

## 功能说明

基于 pytest 测试框架，实现 MLA Prolog V3 算子的功能验证：

- **CPU 侧**：复现算子功能用以生成 Golden 数据
- **NPU 侧**：通过 torch_npu 进行算子直调获取实际数据
- **精度对比**：进行 CPU 与 NPU 结果的精度对比验证算子功能

### 主要特性

- **实现逻辑**：在 CPU 端复现算子逻辑生成 Golden 脚本，与调用算子的结果进行精度对比
- **用例泛化**：通过 pytest 进行测试用例的泛化，可以通过输入不同的参数在运行时自动交叉组合
- **量化模式支持**：支持无量化、INT8 权重量化、MXFP8 量化、逐块 KV 量化等多种模式
- **Fuzz 测试**：支持随机化参数的 Fuzz 测试，通过环境变量控制
- **参数校验**：自动校验参数约束，跳过非法参数组合

### 参数限制

- **batch_size** (B): <= 65536
- **head_num** (N): {1, 2, 4, 8, 16, 32, 64, 128}
- **He**: {1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192}
- **数据格式**: BF16 / INT8 / FP8_E4M3FN
- **cache_mode**: PA_BSND / PA_NZ / PA_BLK_BSND / PA_BLK_NZ / BSND / TND
- **weight_quant_mode**: 0（无量化）/ 1（UqQr量化）/ 2（全量化）/ 3（MXFP8全量化）
- **kv_cache_quant_mode**: 0（无量化）/ 1（per-tensor）/ 2（per-channel）/ 3（per-tile）
- **query_quant_mode**: 0（无量化）/ 1（per-token-head），仅 weight_quant_mode=2/3 时有效
- **block_size**: [16, 1024]，必须为 16 的倍数

## 固定维度

以下维度在所有测试用例中保持不变：

| 参数 | 值 | 含义 |
|------|-----|------|
| Hcq | 1536 | 压缩后的 Query 隐藏维度 |
| Hckv | 512 | 压缩后的 KV 隐藏维度 |
| D | 128 | Head 维度 |
| Dr | 64 | RoPE 维度 |
| N2 (kv_head_num) | 1 | KV Head 数量 |

## 环境配置

### 前置要求

1. torch_npu 安装包下载路径：[torch_npu 安装教程](https://gitcode.com/Ascend/pytorch)
2. CANN 包环境配置可参考：[环境部署](../../../../docs/zh/context/quick_install.md)

## 运行方式

在 pytest 文件夹路径下执行：

### CI 测试（基础无量化场景）

```bash
pytest test.py -v -m ci
```

### Fuzz 测试

```bash
MLA_PROLOG_V3_ENABLE_FUZZ=1 pytest test.py::test_mla_prolog_v3_fuzz -v
```

### 自定义 Fuzz 参数

```bash
MLA_PROLOG_V3_ENABLE_FUZZ=1 MLA_PROLOG_V3_FUZZ_CASES=50 MLA_PROLOG_V3_FUZZ_SEED=42 pytest test.py::test_mla_prolog_v3_fuzz -v
```

### 运行所有测试

```bash
pytest test.py -v
```
