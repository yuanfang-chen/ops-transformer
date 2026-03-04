# MlaPrologV3算子测试框架

## 文件结构

pytest/
- `test.py`                         # pytest泛化测试入口（CI + Fuzz）
- `testcases.py`                    # 测试参数空间配置
- `check_valid_param.py`            # 入参与结果校验
- `prologv3_generalized.py`         # CPU参考实现与NPU调用封装
- `prologv3_no_quant_pa_bsnd.py`    # 历史参考脚本（非泛化主路径）
- `pytest.ini`                      # pytest marker定义

## 功能说明

基于pytest测试框架，实现MlaPrologV3算子的功能验证：
- CPU侧：复现算子逻辑，生成参考结果
- NPU侧：通过 `torch_npu.npu_mla_prolog_v3` 直调算子
- 对比方式：同时校验
- `outputs`：`query`、`query_rope`、`dequant_scale_q_nope`、`query_norm`、`dequant_scale_q_norm`
- `inplace`：`kv_cache`、`kr_cache` 原地更新结果

### 当前实现范围

✅**已实现**：MlaPrologV3泛化场景（含无量化/部分量化/全量化）及多cache_mode验证  
⚠️**运行时限制**：mxfp8量化依赖Ascend 950 + float8运行时支持，不满足时会自动跳过

### 参数限制

- **数据格式**：BF16
- **B**：Batch表示输入样本批量大小，取值范围为1~65536。
- **S**：Seq-Length表示输入样本序列长度，取值范围为1~16。
- **S2说明**：当前pytest实现中 `S2` 跟随 `S1`，即 `S2 = S1`。`kv_seq` 已从参数化中移除。
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

### 精度规则

- `int8` 输出：允许逐元素绝对误差 `<= 1`
- 其他整型/布尔输出：严格相等
- 浮点输出：按dtype使用对应 `rtol/atol`

### 连续/非连续误差检查模式

- 默认模式：连续严格检查（不允许不连续误差点）
- 开启非连续误差容忍：允许少量离散点误差，阈值由比例和数量共同控制
- 测试开始时会打印当前模式和开启方式

开关与阈值环境变量：
- `MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR`：`0`/`1`，默认 `0`
- `MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_RATIO`：允许误差点占比上限，默认 `0.001`
- `MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_COUNT`：允许误差点个数上限，默认 `0`（表示仅按比例约束）

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

### 运行方式：运行泛化测试用例
默认运行
```bash
python3 -m pytest -rA -s test.py
```

CI用例
```bash
python3 -m pytest -rA -s test.py -v -m ci
```

图模式（若有对应用例）
```bash
python3 -m pytest -rA -s test.py -v -m graph
```

随机模糊测试（默认关闭）
```bash
MLA_PROLOG_V3_ENABLE_FUZZ=1 python3 -m pytest -rA -s test.py -v -m fuzz
```

开启非连续误差容忍模式（示例）
```bash
MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR=1 \
MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_RATIO=0.001 \
MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_COUNT=0 \
python3 -m pytest -rA -s test.py -v -m ci
```

使用默认连续严格检查（示例）
```bash
MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR=0 python3 -m pytest -rA -s test.py -v -m ci
```

可选环境变量：
- `MLA_PROLOG_V3_FUZZ_CASES`：随机用例数，默认 `20`
- `MLA_PROLOG_V3_FUZZ_SEED`：随机种子，默认 `3`
- `MLA_PROLOG_V3_CPU_INFO_LOG`：CPU参考实现INFO日志开关，默认 `0`（关闭），`1` 为开启
- `MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR`：比较检查模式开关，默认 `0`（连续严格检查）
- `MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_RATIO`：非连续误差比例阈值，默认 `0.001`
- `MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_COUNT`：非连续误差数量阈值，默认 `0`（仅比例生效）
