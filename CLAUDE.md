# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

---

## 一、项目概述

ops-transformer 是 CANN (Compute Architecture for Neural Networks) 的进阶算子库，提供 transformer 类大模型计算能力。该项目使用 AscendC 编程模型开发 AI Core 算子，支持两种开发模式：

- **标准算子开发**：完整的 op_host/op_kernel 结构，适用于生产级算子
- **快速内核启动开发**：最小化模板，用于快速原型设计和社区贡献

**算子分类**：attention、MoE、FFN、posembedding、mc2（多芯片通信）、gmm（分组矩阵乘法）

**支持硬件**：Atlas A2/A3、Ascend 950PR/DT、Atlas 推理、移动平台（Kirin 系列）

---

## 二、快速上手

### 2.1 环境准备

```bash
# 安装依赖
bash install_deps.sh

# 确保环境变量已设置
echo $ASCEND_HOME_PATH  # 应指向 CANN 安装目录
```

### 2.2 编译和运行

```bash
# 构建单个算子
bash build.sh --pkg --soc=ascend910b --ops=add_example -j16

# 构建所有算子
bash build.sh --pkg --soc=ascend910b -j16

# 安装算子包
./build_out/cann-ops-transformer-*.run

# 配置环境变量
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

# 运行示例
bash build.sh --run_example add_example eager cust --vendor_name=custom
```

### 2.3 常用命令速查

| 任务 | 命令 |
|------|------|
| 创建新算子目录 | `bash build.sh --genop=attention/my_new_op` |
| 构建算子 | `bash build.sh --pkg --soc=ascend910b --ops=op_name` |
| 运行单元测试 | `bash build.sh -u` |
| 只构建不运行测试 | `bash build.sh -u --noexec` |
| 清理构建产物 | `bash build.sh --make_clean` |

---

## 三、代码架构

### 3.1 算子目录结构

```
${op_class}/                          # 算子类别
└── ${op_name}/                       # 算子名称（下划线命名）
    ├── op_host/                      # Host 端实现
    │   ├── ${op_name}_def.cpp       # 算子定义（输入/输出/数据类型/格式）
    │   ├── ${op_name}_tiling.cpp    # Tiling 实现（数据分块策略）
    │   └── CMakeLists.txt
    ├── op_kernel/                    # Device 端 Kernel 实现
    │   ├── ${op_name}.cpp           # Kernel 入口点（__global__ __aicore__ void）
    │   ├── ${op_name}.h             # Kernel 类定义（Init/Process/CopyIn/Compute/CopyOut）
    │   ├── ${op_name}_tiling_data.h # Tiling 数据结构
    │   ├── ${op_name}_tiling_key.h  # Tiling 键（策略选择）
    │   └── arch${ARCH}/             # 架构特定实现
    ├── examples/                     # 使用示例
    │   ├── test_aclnn_${op_name}.cpp  # Eager 模式示例
    │   └── test_geir_${op_name}.cpp   # 图模式示例
    ├── tests/ut/                     # 单元测试
    ├── CMakeLists.txt               # 算子 CMake 配置
    └── README.md                    # 算子文档
```

### 3.2 核心组件说明

#### Host 端（op_host/）
- **`*_def.cpp`**：使用 OP_ADD 宏定义算子接口
- **`*_tiling.cpp`**：实现数据 tiling 策略，将张量划分为块以进行并行计算
- **`*_infershape.cpp`**（可选）：动态形状推断逻辑

#### Device 端 Kernel（op_kernel/）
- **`*.cpp`**：入口点，使用 `__global__ __aicore__ void` 函数和宏（REGISTER_TILING_DEFAULT、GET_TILING_DATA_WITH_STRUCT）
- **`*.h`**：Kernel 类模板，包含核心方法：
  - `Init()`：初始化 GlobalTensor 和缓冲区
  - `Process()`：主处理循环
  - `CopyIn()`：数据从 GM 到 UB
  - `Compute()`：核心计算逻辑
  - `CopyOut()`：结果从 UB 到 GM
- **数据流管理**：使用 TPipe、TQue、GlobalTensor、LocalTensor
- **双缓冲**：使用 BUFFER_NUM=2 实现流水线并行

#### Tiling 系统
- **`*_tiling_data.h`**：存储 tiling 配置的结构体（blockLength、tileNum、tileLength）
- **`*_tiling_key.h`**：定义不同 tiling 策略的枚举
- **性能关键**：根据 UB（统一缓冲区）大小和可用的 AI Core 分区数据

### 3.3 项目目录结构

| 目录 | 说明 |
|------|------|
| `attention/` | Attention 算子（flash_attention_score、lightning_indexer、attention_update 等） |
| `moe/` | 混合专家算子（路由、专家计算、token 分配/取消排列） |
| `ffn/` | 前馈网络算子（SwiGLU 变体、GEMM 操作） |
| `posembedding/` | 位置编码算子（RoPE、KV 缓存、RMS 归一化与 RoPE） |
| `common/` | 共享工具、头文件和基础设施（tiling_base、kernel、framework、op_graph） |
| `mc2/` | 多芯片通信算子（alltoall、attention_to_ffn、ffn_to_attention、moe 操作） |
| `gmm/` | 分组矩阵乘法算子 |
| `examples/` | 示例算子（add_example、fast_kernel_launch_example） |
| `experimental/` | 社区贡献的算子（最小交付件） |
| `tests/` | 集成测试和单元测试 |
| `cmake/` | 构建系统配置 |
| `scripts/` | 实用脚本（opgen、构建工具） |

---

## 四、开发指南

### 4.1 标准算子开发流程

#### 步骤 1：创建算子目录
```bash
bash build.sh --genop=${category}/${op_name}
```

#### 步骤 2：实现核心文件

**必需文件：**
- `op_host/${op_name}_def.cpp`：算子定义
- `op_host/${op_name}_tiling.cpp`：Tiling 策略
- `op_kernel/${op_name}.cpp`：Kernel 入口点
- `op_kernel/${op_name}.h`：Kernel 类实现
- `op_kernel/${op_name}_tiling_data.h`：Tiling 数据结构
- `op_kernel/${op_name}_tiling_key.h`：Tiling 键定义
- `README.md`：算子文档

**可选文件：**
- `op_host/${op_name}_infershape.cpp`：动态形状推断
- `op_kernel/arch${ARCH}/`：架构特定实现

#### 步骤 3：构建和测试
```bash
# 编译
bash build.sh --pkg --soc=ascend910b --ops=${op_name}

# 安装
./build_out/cann-ops-transformer-*.run

# 配置环境
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

# 运行示例
bash build.sh --run_example ${op_name} eager cust
```

### 4.2 快速内核启动开发（实验性/社区贡献）

适用场景：快速原型设计、社区贡献、PyTorch 集成

```bash
cd examples/fast_kernel_launch_example
pip install -r requirements.txt
pip install --no-build-isolation -e .  # 开发模式
```

**最小交付件：**
- Kernel 实现（`*.cpp`）
- CMakeLists.txt
- README.md

**特点：**
- ✅ PyTorch 集成，支持 torch_npu
- ✅ 使用 `<<<>>>` 内核启动语法
- ✅ 快速构建周期
- ✅ 自动微分支持

参考 `examples/fast_kernel_launch_example/README.md` 了解完整使用示例（如 grouped_matmul）。

### 4.3 社区贡献指南

贡献提交到 `experimental/${category}/` 目录：

**交付件：**
- Kernel 实现（`*.cpp`、`*.h`）
- Python 测试（`test_${op_name}.py`）
- `README.md`
- `CMakeLists.txt`

**参考模板：** `examples/fast_kernel_launch_example/`

### 4.4 关键开发注意事项

| 主题 | 说明 |
|------|------|
| **Tiling 策略** | 性能关键 - 根据 UB 大小和可用 AI Core 分区数据 |
| **双缓冲** | 使用 BUFFER_NUM=2 的 TQue 实现流水线并行 |
| **模板参数** | 使用 TilingKey 支持多种数据类型和计算策略 |
| **架构变体** | 将架构特定代码放在 `arch${ARCH}/` 子目录 |
| **CANN 集成** | 从 `*_def.cpp` 自动生成 aclnn API |
| **内存管理** | GM（全局内存）、UB（统一缓冲区）、LM（本地内存）层次结构影响性能 |
| **流水线优化** | 通过复制和计算阶段的重叠隐藏数据传输延迟 |

---

## 五、执行模式和硬件

### 5.1 执行模式

| 模式 | 说明 | 示例文件 |
|------|------|----------|
| **Eager 模式** | 通过 aclnn 接口直接调用 ACL API | `test_aclnn_*.cpp` |
| **图模式** | 算子集成到计算图中 | `test_geir_*.cpp` |
| **ONNX 插件** | 某些算子提供 ONNX 框架集成 | `framework/` 子目录 |

### 5.2 执行硬件

| 硬件单元 | 说明 | 示例算子 |
|----------|------|----------|
| **AI Core** | 高性能向量/cube 操作 | 大多数算子 |
| **AI CPU** | 控制逻辑 | attention_worker_scheduler |

### 5.3 支持的硬件平台

| NPU 型号 | 产品系列 | 架构目录 |
|----------|----------|----------|
| ascend310p | Atlas 推理 | arch20 |
| ascend910b | Atlas A2（默认） | arch32 |
| ascend910_93 | Atlas A3 | arch32 |
| ascend950 | Ascend 950PR/DT | arch35 |
| kirinx90, kirin9030 | 移动平台 | arch32 |
| mc62cm12a | 移动平台 | arch38 |

---

## 六、测试和调试

### 6.1 调试方法

| 方法 | 适用场景 | 命令/示例 |
|------|----------|------------|
| **PRINTF()** | 打印标量值 | `AscendC::PRINTF("value: %llu\n", value)` |
| **DumpTensor()** | 打印张量内容 | `DumpTensor(tensor, 0, 128)` |
| **msprof** | 性能分析（Atlas A2/A3） | `msprof --application="./test_aclnn_xxx"` |
| **cannsim** | Ascend 950PR 仿真 | `cannsim record ./test_executable -s Ascend950 --gen-report` |
| **msDebug** | 单步调试（死锁、内存违规） | 使用 msDebug 工具 |

### 6.2 测试类型

#### 单元测试（UT）
```bash
# 构建并运行所有单元测试
bash build.sh -u

# 只构建不运行
bash build.sh -u --noexec

# 特定测试类型
bash build.sh --ophost_test      # Host 端测试
bash build.sh --opapi_test       # API 测试
bash build.sh --opgraph_test     # 图模式测试
```

**测试位置：** `tests/ut/`，按组件组织（op_host、op_api、op_kernel、op_graph）

#### 集成测试
**测试位置：** `examples/`，包括 eager 模式（`test_aclnn_*.cpp`）和图模式（`test_geir_*.cpp`）

#### CI 测试
- 编译检查
- 静态检查
- UT 测试
- 冒烟测试

**触发方式：** 在 PR 上评论 `/compile`

### 6.3 性能分析

#### Atlas A2/A3
```bash
# 采集板载性能指标
msprof op ./test_aclnn_xxx

# 流水线仿真
msprof op simulator --output=$PWD/pipeline --kernel-name "OpName" ./test_executable
```

#### Ascend 950PR
```bash
# 生成仿真报告
cannsim record ./test_executable -s Ascend950 --gen-report
```

---

## 七、构建系统详解

### 7.1 构建流程

```
build.sh → 解析参数 → 设置 CMake 选项
    ↓
CMake → op_add_subdirectory() 发现算子
    ↓
对每个算子：
  ├── 编译 op_host
  ├── 生成 aclnn 包装器
  └── 编译 op_kernel 二进制文件
    ↓
打包 → .run 可安装包
    ↓
使用 bisheng 编译器（可用 ccache 包装）
```

### 7.2 构建参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--soc` | 目标 NPU 型号 | `--soc=ascend910b` |
| `--ops` | 特定算子（逗号分隔） | `--ops=add_example,flash_attention_score` |
| `--vendor_name` | 自定义包名 | `--vendor_name=my_custom` |
| `--experimental` | 构建实验性算子 | `--experimental` |
| `--jit` | 构建 JIT 包（无内核二进制） | `--jit` |
| `-j[n]` | 编译线程数 | `-j16` |
| `-O[n]` | 编译优化级别 | `-O3` |

---

## 八、重要说明

### 8.1 环境要求

- **环境变量**：确保 `ASCEND_HOME_PATH` 设置为 CANN 安装目录
- **依赖项**：CANN toolkit、Python 3.7+、GCC 7.3+、CMake 3.16+
- **版本兼容性**：将 CANN 版本与 release 仓库中的项目标签匹配

### 8.2 自定义算子安装

```bash
# 安装自定义算子到 CANN vendors 目录
./build_out/cann-ops-transformer-*.run

# 配置环境变量
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

### 8.3 特殊功能

| 功能 | 说明 |
|------|------|
| **ONNX 插件支持** | FlashAttention、MultiHeadAttention、MoEComputeExpertTokens 等算子提供 ONNX 框架插件 |
| **Docker 环境** | 提供预配置的 Docker 镜像，预装 CANN 和依赖项 |
| **快速内核启动** | 最小化模板用于快速原型设计，无需完整 op_host 实现 |

---

## 九、常见问题

### Q1: 如何选择架构目录？
根据目标硬件选择：
- ascend310p → arch20
- ascend910b/910_93/kirinx90/kirin9030 → arch32
- ascend950 → arch35
- mc62cm12a → arch38

### Q2: 什么时候需要动态形状推断？
当算子输出形状依赖于输入数据（而非仅形状）时，需要实现 `*_infershape.cpp`。

### Q3: 如何优化 tiling 性能？
1. 合理设置 blockLength（每个 AI Core 处理的数据量）
2. 优化 tileNum 和 tileLength 以充分利用 UB
3. 使用双缓冲（BUFFER_NUM=2）隐藏数据传输延迟
4. 根据数据类型选择合适的 TilingKey

### Q4: 社区贡献和标准开发的区别？

| 特性 | 社区贡献（experimental） | 标准开发 |
|------|----------------------|----------|
| 交付件 | 最小（Kernel + 测试） | 完整（op_host + op_kernel） |
| 位置 | `experimental/${category}/` | 对应类别目录 |
| 适用场景 | 快速原型、实验性功能 | 生产级算子 |
| 文档要求 | README.md | README.md + 详细文档 |

---

## 十、参考资料

- **官方文档**：[CANN 官方文档](https://hiascend.com/document)
- **算子列表**：`docs/zh/op_list.md`
- **开发指南**：`docs/zh/develop/aicore_develop_guide.md`
- **调试调优**：`docs/zh/debug/op_debug_prof.md`
- **示例代码**：`examples/add_example/`、`examples/fast_kernel_launch_example/`
- **贡献指南**：`CONTRIBUTING.md`

---

## 附录：完整命令速查表

### 构建相关
```bash
bash build.sh --pkg --soc=ascend910b --ops=add_example -j16           # 构建单个算子
bash build.sh --pkg --soc=ascend910b -j16                            # 构建所有算子
bash build.sh --genop=attention/my_new_op                              # 创建新算子目录
bash build.sh --make_clean                                           # 清理构建产物
```

### 测试相关
```bash
bash build.sh -u                                                     # 构建并运行所有单元测试
bash build.sh -u --noexec                                            # 只构建不运行
bash build.sh --ophost_test                                          # Host 端测试
bash build.sh --opapi_test                                           # API 测试
bash build.sh --opgraph_test                                         # 图模式测试
bash build.sh --run_example add_example eager cust --vendor_name=custom  # 运行示例
```

### 性能分析
```bash
msprof op ./test_aclnn_xxx                                           # 板载性能分析（A2/A3）
msprof op simulator --output=$PWD/pipeline --kernel-name "OpName" ./test_executable  # 流水线仿真
cannsim record ./test_executable -s Ascend950 --gen-report            # Ascend 950PR 仿真
```

## 三、算子设计文档
算子的设计文档章节如下：
1.需求分析
1.1 需求背景
1.2 功能分析
1.3 对标实现分析
2 特性实现方案
2.1 算子整体计算流程图
2.2 aclnn设计
2.3 Infershape设计
2.4 Tiling设计
2.5 片上内存资源设计
2.6 kernel方案
3 其他算子交付件
4 测试设计
4.1 测试标杆
4.2 测试用例设计
4.2.1 白盒用例
5 遗留问题