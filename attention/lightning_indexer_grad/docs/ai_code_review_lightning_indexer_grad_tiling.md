# 代码检视报告
**项目名称**：NPU算子检视报告
**检视模块**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp
**检视人**：CANNBot Code Reviewer
**检视日期**：2026-03-13


## 🔍 检视概览
| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 33 个 |
| 严重级（HIGH）问题 | 16 个 |
| 中等级（MEDIUM）问题 | 17 个 |
| 轻微级（LOW）问题 | 0 个 |
| 误报数量 | 0 个 |

**核心结论**：该代码存在较多安全风险，主要集中在指针解引用未判空、除法运算未检查除零、外部输入未校验等方面。建议优先修复HIGH级别问题，特别是空指针解引用风险，这些风险可能导致程序崩溃。

---

## ❌ 问题详情及修改建议

### 问题ID：NUM-001 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - BSND分支
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.3 | 除法运算前未检查除数是否为0 | +40% | 40% |
| 2 | 外部数据来源 | - | headNumQ和headNumK来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 除法运算在第86行执行，但参数校验在第113行（除法之后） | +30% | 95% |
| 4 | 校验不充分 | - | 第113行只检查headNumK!=LIMIT_HEADNUMK，未检查是否为0 | +25.0% | 120% |

**结论**：自信值 **120%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.3 确保除法和余数运算不会导致除以零的错误
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:86
**问题类型**：除零未保护
**问题描述**：headNumK来自外部输入，在（第86行）除法运算前未检查是否为0，可能导致除零错误，违反红线规范"除法/求余必须做除零保护"要求。

#### 修改建议
**修改前代码**：
```cpp
groupNum = headNumQ / headNumK;
```
**修改后代码**：
```cpp
OP_CHECK_IF(headNumK == 0,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "headNumK is zero."),
    return ge::GRAPH_FAILED);
groupNum = headNumQ / headNumK;
```
**修改说明**：在除法运算前添加headNumK除零校验，报错并返回错误码，符合红线规范第2.3条要求，彻底避免除零崩溃风险。

---

### 问题ID：NUM-002 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - TND分支
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.3 | 除法运算前未检查除数是否为0 | +40% | 40% |
| 2 | 外部数据来源 | - | headNumQ和headNumK来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 除法运算在第102行执行，但参数校验在第113行（除法之后） | +30% | 95% |
| 4 | 校验不充分 | - | 第113行只检查headNumK!=LIMIT_HEADNUMK，未检查是否为0 | +25% | 120% |

**结论**：自信值 **120%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.3 确保除法和余数运算不会导致除以零的错误
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:102
**问题类型**：除零未保护
**问题描述**：headNumK来自外部输入，在（第102行）除法运算前未检查是否为0，可能导致除零错误，违反红线规范"除法/求余必须做除零保护"要求。

#### 修改建议
**修改前代码**：
```cpp
groupNum = headNumQ / headNumK;
```
**修改后代码**：
```cpp
OP_CHECK_IF(headNumK == 0,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "headNumK is zero."),
    return ge::GRAPH_FAILED);
groupNum = headNumQ / headNumK;
```
**修改说明**：在除法运算前添加headNumK除零校验，报错并返回错误码，符合红线规范第2.3条要求，彻底避免除零崩溃风险。

---

### 问题ID：MEM-001 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取query输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | opParamInfo.query.desc来自context_->GetInputDesc(QUERY_INDEX) | +25% | 65% |
| 3 | 上下文防御缺失 | - | 第50行获取desc后，第52行直接解引用，未检查是否为nullptr | +30% | 95% |
| 4 | 类似模式 | - | 第54、58、61、64行都存在相同的模式 | +20% | 115% |

**结论**：自信值 **115%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:52
**问题类型**：空指针未保护
**问题描述**：GetInputDesc()返回的指针可能为nullptr，直接解引用会导致程序崩溃，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
opParamInfo.query.shape = context_->GetInputShape(QUERY_INDEX);
queryDataType = opParamInfo.query.desc->GetDataType();
```
**修改后代码**：
```cpp
opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
OP_CHECK_IF(opParamInfo.query.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(QUERY_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.query.shape = context_->GetInputShape(QUERY_INDEX);
OP_CHECK_IF(opParamInfo.query.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(QUERY_INDEX) failed."),
    return ge::GRAPH_FAILED);
queryDataType = opParamInfo.query.desc->GetDataType();
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求，彻底避免空指针解引用崩溃风险。

---

### 问题ID：MEM-002 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取key输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | 来自context_->GetInputDesc(KEY_INDEX)和GetInputShape(KEY_INDEX) | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:54
**问题类型**：空指针未保护
**问题描述**：GetInputDesc()和GetInputShape()返回的指针可能为nullptr，未检查直接使用，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
OP_CHECK_IF(opParamInfo.key.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(KEY_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);
OP_CHECK_IF(opParamInfo.key.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(KEY_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-003 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取dy输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | 来自context_->GetInputDesc(DY_INDEX)和GetInputShape(DY_INDEX) | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:58
**问题类型**：空指针未保护
**问题描述**：GetInputDesc()和GetInputShape()返回的指针可能为nullptr，未检查直接使用，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.dy.desc = context_->GetInputDesc(DY_INDEX);
opParamInfo.dy.shape = context_->GetInputShape(DY_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.dy.desc = context_->GetInputDesc(DY_INDEX);
OP_CHECK_IF(opParamInfo.dy.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(DY_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.dy.shape = context_->GetInputShape(DY_INDEX);
OP_CHECK_IF(opParamInfo.dy.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(DY_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-004 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取sparseIndices输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | 来自context_->GetInputDesc()和GetInputShape() | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:61
**问题类型**：空指针未保护
**问题描述**：GetInputDesc()和GetInputShape()返回的指针可能为nullptr，未检查直接使用，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INDEX);
opParamInfo.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INDEX);
OP_CHECK_IF(opParamInfo.sparseIndices.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(SPARSE_INDICES_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INDEX);
OP_CHECK_IF(opParamInfo.sparseIndices.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(SPARSE_INDICES_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-005 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取weights输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | 来自context_->GetInputDesc()和GetInputShape() | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:64
**问题类型**：空指针未保护
**问题描述**：GetInputDesc()和GetInputShape()返回的指针可能为nullptr，未检查直接使用，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp

opParamInfo.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
opParamInfo.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
OP_CHECK_IF(opParamInfo.weights.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(WEIGTHS_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
OP_CHECK_IF(opParamInfo.weights.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(WEIGTHS_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-006 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取属性
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | attrs来自context_->GetAttrs() | +25% | 65% |
| 3 | 上下文防御缺失 | - | 第68行获取attrs后，第69行直接解引用，未检查是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling:69
**问题类型**：空指针未保护
**问题描述**：GetAttrs()返回的指针可能为nullptr，直接解引用会导致程序崩溃，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
auto attrs = context_->GetAttrs();
opParamInfo.headNum = *attrs->GetInt(ATTR_HEADNUM_INDEX);
```
**修改后代码**：
```cpp
auto attrs = context_->GetAttrs();
OP_CHECK_IF(attrs == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetAttrs() failed."),
    return ge::GRAPH_FAILED);

auto headNumPtr = attrs->GetInt(ATTR_HEADNUM_INDEX);
OP_CHECK_IF(headNumPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_HEADNUM_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.headNum = *headNumPtr;
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-007 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取属性值
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | GetInt()和GetStr()返回的指针可能为nullptr | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:69-72
**问题类型**：空指针未保护
**问题描述**：GetInt()和GetStr()返回的指针可能为nullptr，解引用会导致程序崩溃，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.headNum = *attrs->GetInt(ATTR_HEADNUM_INDEX);
opParamInfo.layout = attrs->GetStr(ATTR_LAYOUT_INDEX);
opParamInfo.sparseMode = *attrs->GetInt(ATTR_SPARSEMODE_INDEX);
opParamInfo.preTokens = *attrs->GetInt(ATTR_PRETOKENS_INDEX);
opParamInfo.nextTokens = *attrs->GetInt(ATTR_NEXTTOKENS_INDEX);
```
**修改后代码**：
```cpp
auto headNumPtr = attrs->GetInt(ATTR_HEADNUM_INDEX);
OP_CHECK_IF(headNumPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_HEADNUM_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.headNum = *headNumPtr;

auto layoutPtr = attrs->GetStr(ATTR_LAYOUT_INDEX);
OP_CHECK_IF(layoutPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetStr(ATTR_LAYOUT_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.layout = *layoutPtr;

auto sparseModePtr = attrs->GetInt(ATTR_SPARSEMODE_INDEX);
OP_CHECK_IF(sparseModePtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_SPARSEMODE_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.sparseMode = *sparseModePtr;

auto preTokensPtr = attrs->GetInt(ATTR_PRETOKENS_INDEX);
OP_CHECK_IF(preTokensPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_PRETOKENS_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.preTokens = *preTokensPtr;

auto nextTokensPtr = attrs->GetInt(ATTR_NEXTTOKENS_INDEX);
OP_CHECK_IF(nextTokensPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_NEXTTOKENS_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.nextTokens = *nextTokensPtr;
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-008 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取dy shape维度
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 上下文防御缺失 | - | opParamInfo.dy.shape在第58行获取，未检查是否为nullptr | +30% | 70% |
| 3 | 后续依赖 | - | dyShapeDim在第87行和第103行被使用 | +25% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:76
**问题类型**：空指针未保护
**问题描述**：opParamInfo.dy.shape可能为nullptr，直接解引用会导致程序崩溃，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
uint32_t dyShapeDim = opParamInfo.dy.shape->GetStorageShape().GetDimNum();
```
**修改后代码**：
```cpp
OP_CHECK_IF(opParamInfo.dy.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dy.shape is nullptr."),
    return ge::GRAPH_FAILED);
uint32_t dyShapeDim = opParamInfo.dy.shape->GetStorageShape().GetDimNum();
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-009 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - TND分支获取optional输入
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 外部数据来源 | - | GetOptionalInputTensor()和GetOptionalInputDesc()返回的指针可能为nullptr | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:91-94
**问题类型**：空指针未保护
**问题描述**：GetOptionalInputTensor()和GetOptionalInputDesc()返回的指针可能为nullptr，未检查直接使用，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
opParamInfo.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
opParamInfo.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
opParamInfo.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.tensor == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
OP_CHECK_IF(opParamInfo.actualSeqLengthsK.tensor == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
OP_CHECK_IF(opParamInfo.actualSeqLengthsK.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-010 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - TND分支获取batch
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 指针操作使用前必须判空 | +40% | 40% |
| 2 | 上下文防御缺失 | - | opParamInfo.actualSeqLengthsQ.tensor在第91行获取，未检查是否为nullptr | +30% | 70% |
| 3 | 解引用操作 | - | 直接调用GetShapeSize()，如果tensor为nullptr会崩溃 | +25% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:96
**问题类型**：空指针未保护
**问题描述**：opParamInfo.actualSeqLengthsQ.tensor可能为nullptr，直接解引用会导致程序崩溃，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
batch = static_cast<uint32_t>(opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize());
```
**修改后代码**：
```cpp
OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.tensor == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "actualSeqLengthsQ.tensor is nullptr."),
    return ge::GRAPH_FAILED);
batch = static_cast<uint32_t>(opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize());
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：MEM-011 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取dy shape维度用于数组索引
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.6 | 外部数据作为数组索引时必须确保在数组大小范围内 | +40% | 40% |
| 2 | 外部数据来源 | - | dyShapeDim来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未检查dyShapeDim是否大于0 | +30% | 95% |
| 4 | 减法操作 | - | dyShapeDim - 1可能导致下溢（如果dyShapeDim为0） | +25% | 120% |

**结论**：自信值 **120%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.6 外部数据作为数组索引时必须确保在数组大小范围内
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:87,103
**问题类型**：数组越界
**问题描述**：dyShapeDim - 1作为数组索引，未检查dyShapeDim是否大于0，可能导致数组越界，违反红线规范"外部数据作为数组索引时必须确保在数组大小范围内"要求。

#### 修改建议
**修改前代码**：
```cpp
topK = static_cast<uint32_t>(opParamInfo.dy.shape->GetStorageShape().GetDim(dyShapeDim - 1));
```
**修改后代码**：
```cpp
OP_CHECK_IF(dyShapeDim == 0,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dyShapeDim is zero."),
    return ge::GRAPH_FAILED);
topK = static_cast<uint32_t>(opParamInfo.dy.shape->GetStorageShape().GetDim(dyShapeDim - 1));
```
**修改说明**：在数组索引前检查dyShapeDim是否大于0，报错并返回错误码，符合红线规范第2.6条要求。

---

### 问题ID：INPUT-001 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取属性值
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.11 | 外部输入数据需要做合法性校验 | +40% | 40% |
| 2 | 外部数据来源 | - | headNum、layout、sparseMode、preTokens、nextTokens都来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未对这些参数进行范围校验 | +30% | 95% |
| 4 | 后续使用 | - | 这些参数在后续计算中使用（如第86行groupNum = headNumQ / headNumK） | +25% | 120% |

**结论**：自信值 **120%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.11 外部输入数据需要做合法性校验
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:69-72
**问题类型**：外部输入未校验
**问题描述**：外部输入的属性参数未进行合法性校验，可能导致后续计算错误，违反红线规范"外部输入数据需要做合法性校验"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.headNum = *attrs->GetInt(ATTR_HEADNUM_INDEX);
opParamInfo.layout = attrs->GetStr(ATTR_LAYOUT_INDEX);
opParamInfo.sparseMode = *attrs->GetInt(ATTR_SPARSEMODE_INDEX);
opParamInfo.preTokens = *attrs->GetInt(ATTR_PRETOKENS_INDEX);
opParamInfo.nextTokens = *attrs->GetInt(ATTR_NEXTTOKENS_INDEX);
```
**修改后代码**：
```cpp
auto headNumPtr = attrs->GetInt(ATTR_HEADNUM_INDEX);
OP_CHECK_IF(headNumPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_HEADNUM_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.headNum = *headNumPtr;
OP_CHECK_IF(opParamInfo.headNum == 0 || opParamInfo.headNum > MAX_HEADNUM,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "headNum is out of range."),
    return ge::GRAPH_FAILED);

auto layoutPtr = attrs->GetStr(ATTR_LAYOUT_INDEX);
OP_CHECK_IF(layoutPtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetStr(ATTR_LAYOUT_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.layout = *layoutPtr;
OP_CHECK_IF(opParamInfo.layout != "BSND" && opParamInfo.layout != "TND",
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "layout must be BSND or TND."),
    return ge::GRAPH_FAILED);

auto sparseModePtr = attrs->GetInt(ATTR_SPARSEMODE_INDEX);
OP_CHECK_IF(sparseModePtr == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInt(ATTR_SPARSEMODE_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.sparseMode = *sparseModePtr;
OP_CHECK_IF(opParamInfo.sparseMode > MAX_SPARSEMODE,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "sparseMode is out of range."),
    return ge::GRAPH_FAILED);
```
**修改说明**：添加外部输入参数的合法性校验，包括空指针检查和范围校验，符合红线规范第2.11条要求。

---

### 问题ID：INPUT-002 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - BSND分支获取shape维度
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.11 | 外部输入数据需要做合法性校验 | +40% | 40% |
| 2 | 外部数据来源 | - | batch、seqlenQ、headNumQ、headDim都来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 未对这些参数进行范围校验 | +30% | 95% |
| 4 | 后续使用 | - | 这些参数在后续计算中使用（如第88行dkSize = batch * seqlenK * headNumK * headDim） | +25% | 120% |

**结论**：自信值 **120%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.11 外部输入数据需要做合法性校验
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:80-84
**问题类型**：外部输入未校验
**问题描述**：外部输入的shape参数未进行合法性校验，可能导致后续计算错误，违反红线规范"外部输入数据需要做合法性校验"要求。

#### 修改建议
**修改前代码**：
```cpp
batch = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
seqlenQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
headNumQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
headDim = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_FOUR));
```
**修改后代码**：
```cpp
batch = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
OP_CHECK_IF(batch == 0 || batch > MAX_BATCH,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "batch is out of range."),
    return ge::GRAPH_FAILED);

seqlenQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
OP_CHECK_IF(seqlenQ == 0 || seqlenQ > MAX_SEQLEN,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "seqlenQ is out of range."),
    return ge::GRAPH_FAILED);

headNumQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
OP_CHECK_IF(headNumQ == 0 || headNumQ > MAX_HEADNUM,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "headNumQ is out of range."),
    return ge::GRAPH_FAILED);

headDim = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_FOUR));
OP_CHECK_IF(headDim == 0 || headDim > MAX_HEADDIM,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "headDim is out of range."),
    return ge::GRAPH_FAILED);
```
**修改说明**：添加外部输入shape参数的合法性校验，包括范围校验，符合红线规范第2.11条要求。

---

### 问题ID：INPUT-003 | 严重级别：HIGH（严重）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 参数校验不完整
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.11 | 外部入参参与循环、递归条件的运算，必须严格校验边界和终止条件 | +20% | 20% |
| 2 | 校验不完整 | - | 只校验了headDim、groupNum、headNumK，未校验其他参数 | +25% | 45% |
| 3 | 上下文防御缺失 | - | batch、seqlenQ、seqlenK、topK等参数未进行范围校验 | +30% | 75% |
| 4 | 后续使用 | - | 这些参数在后续计算中使用 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.11 外部输入数据需要做合法性校验
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:113
**问题类型**：外部输入校验不完整
**问题描述**：只校验了部分参数，未校验其他外部输入参数的合法性，违反红线规范"外部输入数据需要做合法性校验"要求。

#### 修改建议
**修改前代码**：
```cpp
OP_CHECK_IF((headDim != MAX_HEADIM) || (groupNum != MAX_GROUPNUM) || (headNumK != LIMIT_HEADNUMK),
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    "only support headDim is %lu, groupNum is %lu, headNumK is %lu, but current headDim is %lu, groupNum is %lu, headNumK is %lu",
        MAX_HEADIM, MAX_GROUPNUM, LIMIT_HEADNUMK, headDim, groupNum, headNumK),
    return ge::GRAPH_FAILED);
```
**修改后代码**：
```cpp
OP_CHECK_IF((headDim != MAX_HEADIM) || (groupNum != MAX_GROUPNUM) || (headNumK != LIMIT_HEADNUMK),
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    "only support headDim is %lu, groupNum is %lu, headNumK is %lu, but current headDim is %lu, groupNum is %lu, headNumK is %lu",
        MAX_HEADIM, MAX_GROUPNUM, LIMIT_HEADNUMK, headDim, groupNum, headNumK),
    return ge::GRAPH_FAILED);

OP_CHECK_IF(batch == 0 || batch > MAX_BATCH,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "batch is out of range."),
    return ge::GRAPH_FAILED);

OP_CHECK_IF(seqlenQ == 0 || seqlenQ > MAX_SEQLEN,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "seqlenQ is out of range."),
    return ge::GRAPH_FAILED);

OP_CHECK_IF(seqlenK == 0 || seqlenK > MAX_SEQLEN,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "seqlenK is out of range."),
    return ge::GRAPH_FAILED);

OP_CHECK_IF(topK == 0 || topK > MAX_TOPK,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "topK is out of range."),
    return ge::GRAPH_FAILED);
```
**修改说明**：添加更多外部输入参数的合法性校验，符合红线规范第2.11条要求。

---

### 问题ID：NUM-003 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - BSND分支计算dkSize
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +40% | 40% |
| 2 | 外部数据来源 | - | batch、seqlenK、headNumK、headDim都来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 整个函数中没有任何溢出检查 | +30% | 95% |
| 4 | 用于内存分配 | - | dkSize后续用于工作空间大小计算（第136行），溢出会导致内存分配错误 | +25% | 120% |

**结论**：自信值 **120%**** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:88
**问题类型**：整数溢出
**问题描述**：多个外部输入的uint32_t类型变量相乘，未进行溢出检查，可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
dkSize = batch * seqlenK * headNumK * headDim;
```
**修改后代码**：
```cpp
if (batch != 0 && seqlenK != 0 && headNumK != 0 && headDim != 0) {
    if (batch > UINT32_MAX / (seqlenK * headNumK * headDim)) {
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dkSize multiplication overflow."),
        return ge::GRAPH_FAILED;
    }
}
dkSize = batch * seqlenK * headNumK * headDim;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-004 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - TND分支计算dkSize
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +40% | 40% |
| 2 | 外部数据来源 | - | seqlenK、headNumK、headDim都来自外部输入 | +25% | 65% |
| 3 | 上下文防御缺失 | - | 整个函数中没有任何溢出检查 | +30% | 95% |
| 4 | 用于内存分配 | - | dkSize后续用于工作空间大小计算（第136行），溢出会导致内存分配错误 | +25% | 120% |

**结论**：自信值 **120%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:104
**问题类型**：整数溢出
**问题描述**：多个外部输入的uint32_t类型变量相乘，未进行溢出检查，可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
dkSize = seqlenK * headNumK * headDim;
```
**修改后代码**：
```cpp
if (seqlenK != 0 && headNumK != 0 && headDim != 0) {
    if (seqlenK > UINT32_MAX / (headNumK * headDim)) {
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dkSize multiplication overflow."),
        return ge::GRAPH_FAILED;
    }
}
dkSize = seqlenK * headNumK * headDim;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-005 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 设置usedCoreNum
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | blockDim来自CalcTschBlockDim()，参数来自GetCoreNumAiv()和GetCoreNumAic() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:128
**问题类型**：整数溢出
**问题描述**：blockDim * 2可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
tilingData_->set_usedCoreNum(blockDim * 2);
```
**修改后代码**：
```cpp
if (blockDim > UINT32_MAX / 2) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "blockDim * 2 overflow."),
    return ge::GRAPH_FAILED;
}
tilingData_->set_usedCoreNum(blockDim * 2);
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-006 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算dk workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | dkSize来自外部输入，已存在溢出风险 | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |
| 4 | 用于内存分配 | - | workspaceOffset用于工作空间大小计算 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:136
**问题类型**：整数溢出
**问题描述**：dkSize * sizeof(float)可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + dkSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
if (dkSize > UINT64_MAX / sizeof(float)) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dkSize * sizeof(float) overflow."),
    return ge::GRAPH_FAILED;
}
workspaceOffset = (workspaceOffset + dkSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-007 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算dkCore workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | aicNum、dkCoreSize都来自外部输入 | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:139
**问题类型**：整数溢出
**问题描述**：aicNum * dkCoreSize * sizeof(float)可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + aicNum * dkCoreSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
if (aicNum > UINT64_MAX / (dkCoreSize * sizeof(float))) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "aicNum * dkCoreSize * sizeof(float) overflow."),
    return ge::GRAPH_FAILED;
}
workspaceOffset = (workspaceOffset + aicNum * dkCoreSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-008 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算keyGather workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | aicNum来自GetCoreNumAic() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:143
**问题类型**：整数溢出
**问题描述**：a
icNum * keyGatherWorkspaceSize可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + aicNum * keyGatherWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
if (aicNum > UINT64_MAX / keyGatherWorkspaceSize) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "aicNum * keyGatherWorkspaceSize overflow."),
    return ge::GRAPH_FAILED;
}
workspaceOffset = (workspaceOffset + aicNum * keyGatherWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-009 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算reluIn workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | aicNum来自GetCoreNumAic() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:147
**问题类型**：整数溢出
**问题描述**：aicNum * reluInWorkspaceSize可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + aicNum * reluInWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
if (aicNum > UINT64_MAX / reluInWorkspaceSize) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "aicNum * reluInWorkspaceSize overflow."),
    return ge::GRAPH_FAILED;
}
workspaceOffset = (workspaceOffset + aicNum * reluInWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-010 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算reluGrad workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | aicNum来自GetCoreNumAic() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:151
**问题类型**：整数溢出
**问题描述**：aicNum * reluGradWorkspaceSize可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + aicNum * reluGradWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
if (aicNum > UINT64_MAX / reluGradWorkspaceSize) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "aicNum * reluGradWorkspaceSize overflow."),
    return ge::GRAPH_FAILED;
}
workspaceOffset = (workspaceOffset + aicNum * reluGradWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：NUM-011 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算scatterAdd workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.1 | 乘法运算前需要确保不会导致溢出 | +20% | 20% |
| 2 | 外部数据来源 | - | aicNum来自GetCoreNumAic() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 没有溢出检查 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.1 确保有符号整数运算不溢出
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:155
**问题类型**：整数溢出
**问题描述**：aicNum * scatterAddWorkspaceSize可能导致整数溢出，违反红线规范"确保有符号整数运算不溢出"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + aicNum * scatterAddWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
if (aicNum > UINT64_MAX / scatterAddWorkspaceSize) {
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "aicNum * scatterAddWorkspaceSize overflow."),
    return ge::GRAPH_FAILED;
}
workspaceOffset = (workspaceOffset + aicNum * scatterAddWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：在乘法运算前添加溢出检查，报错并返回错误码，符合红线规范第2.1条要求。

---

### 问题ID：MEM-012 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取workspace sizes
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.8 | 指针操作使用前必须判空 | +20% | 20% |
| 2 | 外部数据来源 | - | workSpaces来自context_->GetWorkspaceSizes(1) | +25% | 45% |
| 3 | 上下文防御缺失 | - | 第158行workSpaces[0] = ...直接解引用，未检查是否为nullptr | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:47
**问题类型**：空指针未保护
**问题描述**：GetWorkspaceSizes()返回的指针可能为nullptr，第158行直接解引用会导致程序崩溃，违反红线规范"指针操作使用前必须要判空"要求。

#### 修改建议
**修改前代码**：
```cpp
size_t *workSpaces = context_->GetWorkspaceSizes(1);
```
**修改后代码**：
```size_t *workSpaces = context_->GetWorkspaceSizes(1);
OP_CHECK_IF(workSpaces == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetWorkspaceSizes() failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在解引用指针前添加空指针检查，报错并返回错误码，符合红线规范第2.8条要求。

---

### 问题ID：RES-001 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取workspace sizes
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | workSpaces来自context_->GetWorkspaceSizes(1) | +25% | 45% |
| 3 | 上下文防御缺失 | - | 第158行workSpaces[0] = ...直接使用，未检查是否为nullptr | +30% | 75% |
| 4 | 潜在崩溃 | - | 如果GetWorkspaceSizes()返回nullptr，第158行会导致程序崩溃 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:47
**问题类型**：资源申请未检查
**问题描述**：GetWorkspaceSizes()可能返回nullptr，未检查返回值就直接使用，可能导致程序崩溃，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
size_t *workSpaces = context_->GetWorkspaceSizes(1);
```
**修改后代码**：
```cpp
size_t *workSpaces = context_->GetWorkspaceSizes(1);
OP_CHECK_IF(workSpaces == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetWorkspaceSizes() failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-002 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取query输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | desc来自context_->GetInputDesc(QUERY_INDEX) | +25% | 45% |
| 3 | 上下文防御缺失 | - | 第52行直接解引用desc->GetDataType()，未检查是否为nullptr | +30% | 75% |
| 4 | 潜在崩溃 | - | 如果GetInputDesc()返回nullptr，第52行会导致程序崩溃 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:50
**问题类型**：资源申请未检查
**问题描述**：GetInputDesc()可能返回nullptr，未检查返回值就直接使用，可能导致程序崩溃，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
OP_CHECK_IF(opParamInfo.query.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(QUERY_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-003 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取key输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | desc和shape来自context_->GetInputDesc()和GetInputShape() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:54
**问题类型**：资源申请未检查
**问题描述**：GetInputDesc()和GetInputShape()可能返回nullptr，未检查返回值，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
OP_CHECK_IF(opParamInfo.key.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(KEY_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);
OP_CHECK_IF(opParamInfo.key.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(KEY_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-004 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取dy输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | desc和shape来自context_->GetInputDesc()和GetInputShape() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:58
**问题类型**：资源申请未检查
**问题描述**：GetInputDesc()和GetInputShape()可能返回nullptr，未检查返回值，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.dy.desc = context_->GetInputDesc(DY_INDEX);
opParamInfo.dy.shape = context_->GetInputShape(DY_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.dy.desc = context_->GetInputDesc(DY_INDEX);
OP_CHECK_IF(opParamInfo.dy.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(DY_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.dy.shape = context_->GetInputShape(DY_INDEX);
OP_CHECK_IF(opParamInfo.dy.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(DY_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-005 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取sparseIndices输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | desc和shape来自context_->GetInputDesc()和GetInputShape() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:61
**问题类型**：资源申请未检查
**问题描述**：GetInputDesc()和GetInputShape()可能返回nullptr，未检查返回值，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INDEX);
opParamInfo.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INDEX);
OP_CHECK_IF(opParamInfo.sparseIndices.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(SPARSE_INDICES_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INDEX);
OP_CHECK_IF(opParamInfo.sparseIndices.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(SPARSE_INDICES_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-006 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取weights输入描述
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | desc和shape来自context_->GetInputDesc()和GetInputShape() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 未检查返回值是否为nullptr | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:64
**问题类型**：资源申请未检查
**问题描述**：GetInputDesc()和GetInputShape()可能返回nullptr，未检查返回值，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
opParamInfo.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
OP_CHECK_IF(opParamInfo.weights.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputDesc(WEIGTHS_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
OP_CHECK_IF(opParamInfo.weights.shape == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetInputShape(WEIGTHS_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-007 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 获取属性
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | attrs来自context_->GetAttrs() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 第69-72行直接解引用attrs->GetInt()和attrs->GetStr()，未检查是否为nullptr | +30% | 75% |
| 4 | 潜在崩溃 | - | 如果GetAttrs()返回nullptr，第69-72行会导致程序崩溃 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:68
**问题类型**：资源申请未检查
**问题描述**：GetAttrs()可能返回nullptr，未检查返回值就直接使用，可能导致程序崩溃，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
auto attrs = context_->GetAttrs();
```
**修改后代码**：
```cpp
auto attrs = context_->GetAttrs();
OP_CHECK_IF(attrs == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetAttrs() failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：RES-008 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - TND分支获取optional输入
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.9 | 资源申请后必须判断是否成功 | +20% | 20% |
| 2 | 外部数据来源 | - | tensor和desc来自GetOptionalInputTensor()和GetOptionalInputDesc() | +25% | 45% |
| 3 | 上下文防御缺失 | - | 第96行直接解引用tensor->GetShapeSize()，未检查是否为nullptr | +30% | 75% |
| 4 | 潜在崩溃 | - | 如果GetOptionalInputTensor()返回nullptr，第96行会导致程序崩溃 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.9 资源申请后必须判断是否成功
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:91-94
**问题类型**：资源申请未检查
**问题描述**：GetOptionalInputTensor()和GetOptionalInputDesc()可能返回nullptr，未检查返回值，违反红线规范"资源申请后必须判断是否成功"要求。

#### 修改建议
**修改前代码**：
```cpp
opParamInfo.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
opParamInfo.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
```
**修改后代码**：
```cpp
opParamInfo.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.tensor == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX) failed."),
    return ge::GRAPH_FAILED);
opParamInfo.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.desc == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX) failed."),
    return ge::GRAPH_FAILED);
```
**修改说明**：在资源申请后添加返回值检查，报错并返回错误码，符合红线规范第2.9条要求。

---

### 问题ID：INPUT-004 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算dk workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.10 | 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 | +20% | 20% |
| 2 | 外部数据来源 | - | dkSize来自外部输入（batch * seqlenK * headNumK * headDim） | +25% | 45% |
| 3 | 上下文防御缺失 | - | 未对dkSize进行范围校验 | +30% | 75% |
| 4 | 用于内存分配 | - | dkSize用于工作空间大小计算，如果过大可能导致内存分配失败 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.10 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:136
**问题类型**：外部输入未校验
**问题描述**：dkSize来自外部输入，未进行范围校验就用于工作空间大小计算，可能导致内存分配失败，违反红线规范"外部输入作为内存操作（长度时，需要校验其合法性"要求。

#### 修改建议
**修改前代码**：
```cpp
tilingData_->set_dkWorkSpaceOffset(workspaceOffset);
workspaceOffset = (workspaceOffset + dkSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
OP_CHECK_IF(dkSize > MAX_DKSIZE,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dkSize is too large."),
    return ge::GRAPH_FAILED);
tilingData_->set_dkWorkSpaceOffset(workspaceOffset);
workspaceOffset = (workspaceOffset + dkSize * sizeof(float) + GM GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：添加dkSize的范围校验，报错并返回错误码，符合红线规范第2.10条要求。

---

### 问题ID：INPUT-005 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：DoTiling()函数 - 计算dkCore workspace offset
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.10 | 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 | +20% | 20% |
| 2 | 外部数据来源 | - | aicNum、dkCoreSize来自外部输入 | +25% | 45% |
| 3 | 上下文防御缺失 | - | 未对aicNum、dkCoreSize进行范围校验 | +30% | 75% |
| 4 | 用于内存分配 | - | 用于工作空间大小计算 | +25% | 100% |

**结论**：自信值 **100%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.10 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性
**代码路径**：/mnt/workspace/gitCode/ChenYiran/ops-transformer/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp:139
**问题类型**：外部输入未校验
**问题描述**：aicNum、dkCoreSize来自外部输入，未进行范围校验就用于工作空间大小计算，违反红线规范"外部输入作为内存操作相关函数的复制长度时，需要校验其合法性"要求。

#### 修改建议
**修改前代码**：
```cpp
workspaceOffset = (workspaceOffset + aicNum * dkCoreSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改后代码**：
```cpp
OP_CHECK_IF(aicNum > MAX_AICNUM,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "aicNum is too large."),
    return ge::GRAPH_FAILED);
OP_CHECK_IF(dkCoreSize > MAX_DKCORESIZE,
    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "dkCoreSize is too large."),
    return ge::GRAPH_FAILED);
workspaceOffset = (workspaceOffset + aicNum * dkCoreSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
```
**修改说明**：添加aicNum和dkCoreSize的范围校验，报错并返回错误码，符合红线规范第2.10条要求。

---

## 报告生成时间
2026-03-13 10:00:00
## 报告状态
已完成检视，待修复验证
