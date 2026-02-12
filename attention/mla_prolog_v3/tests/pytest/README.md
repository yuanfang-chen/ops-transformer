# MlaPrologV3算子测试框架

## 文件结构

pytest/
- test.py                      # pytest泛化测试用例运行主程序
- testcases.py                    # 泛化测试用例入参配置
- check_valid_param.py            # 入参检查及精度对比
- prologv3_no_quant_pa_bsnd.py       # CPU侧算子逻辑实现获取golden，npu算子直调
- prologv3_generalized.py         # 全量化/全cache_mode泛化CPU参考实现与NPU调用
- pytest.ini                      # 创建ci、graph、fuzz测试标记

## 功能说明

基于pytest测试框架，实现MlaPrologV3算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据， 通过torchair入图暂不支持
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围

✅**已实现**：MlaPrologV3泛化场景（含无量化/部分量化/全量化）及多cache_mode验证  
⚠️**运行时限制**：mxfp8量化依赖Ascend 950 + float8运行时支持，不满足时会自动跳过

### 参数限制

- **数据格式**：BF16
- **B**：Batch表示输入样本批量大小，取值范围为1~65536。
- **S**：Seq-Length表示输入样本序列长度，取值范围为1~16。
- **He**：Head-Size表示隐藏层的大小，取值为7168。
- **Hcq**：q低秩矩阵维度，取值为1536。
- **N**：Head-Num表示多头数，取值范围为8、16、32、64、128。
- **Hckv**：kv低秩矩阵维度，取值为512。
- **D**：qk不含位置编码维度，取值为128。
- **Dr**：qk位置编码维度，取值为64。
- **Nkv**：kv的head数，取值为1。
- **BlockNum**：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度。
- **BlockSize**：PagedAttention场景下的块大小，取值范围为16、128。
- **T**：BS合轴后的大小，取值范围：1~1048576。
- **cache_mode**: PA_BSND / PA_NZ / PA_BLK_BSND / PA_BLK_NZ / BSND / TND，其用户不特意指定时可传入默认值"PA_BSND"。
- **weight_quant_mode**: 0表示非量化，1表示weight_uq_qr量化，2表示weight_dq、weight_uq_qr、weight_dkv_kr量化，3表示mxfp8量化，默认值为0。
- **kv_quant_mode**: 0表示非量化，1表示per-tensor量化，2表示per-channel量化，3表示per-tile量化，默认值为0。
- **query_quant_mode**: 0表示非量化，1表示per-token-head量化，默认值为0。
- **ckvkr_repo_mode**: 0表示kv_cache和kr_cache分别存储，1表示合并存储，默认值为0。
- **quant_scale_repo_mode**: 0表示量化scale和数据分别存储，1表示合并存储，默认值为0。
- **tile_size**: per-tile量化时每个tile的大小，默认值为128。
- **qc_qr_scale**: Query的尺度矫正系数，默认值为1.0。
- **kc_scale**: Key的尺度矫正系数，默认值为1.0。

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

### 运行方式：运行泛化测试用例
单算子直调+图模式
```bash
python3 -m pytest -rA -s test.py
```

单算子直调
```bash
python3 -m pytest -rA -s test.py -v -m ci
```

图模式
```bash
python3 -m pytest -rA -s test.py -v -m graph
```

随机模糊测试（默认关闭）
```bash
MLA_PROLOG_V3_ENABLE_FUZZ=1 python3 -m pytest -rA -s test.py -v -m fuzz
```

可选环境变量：
- `MLA_PROLOG_V3_FUZZ_CASES`：随机用例数，默认 `20`
- `MLA_PROLOG_V3_FUZZ_SEED`：随机种子，默认 `3`
