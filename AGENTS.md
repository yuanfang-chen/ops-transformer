# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## 项目概述

ops-transformer 是华为昇腾 CANN 算子库中的 Transformer 类大模型进阶算子库，包含 attention、moe、ffn、posembedding、mc2 等算子类别。

## 构建命令

```bash
# 编译指定算子包（必须指定 --soc 和 --ops）
bash ops-transformer-dev/build.sh --pkg --soc=ascend910b --ops=<算子名>

# 编译并运行单元测试
bash ops-transformer-dev/build.sh -u --ops=<算子名>

# 仅编译 UT 不执行
bash ops-transformer-dev/build.sh -u --noexec --ops=<算子名>

# 运行算子样例
bash ops-transformer-dev/build.sh --run_example <算子名> eager cust --vendor_name=custom

# 编译特定库
bash ops-transformer-dev/build.sh --ophost   # ophost_transformer.so
bash ops-transformer-dev/build.sh --opapi    # opapi_transformer.so
bash ops-transformer-dev/build.sh --opgraph  # graph_plugin_transformer.so

# 创建新算子目录
bash ops-transformer-dev/build.sh --genop=<算子类别>/<算子名>
```

## 代码风格

- **C++ 标准**: C++17（`-std=gnu++1z`）
- **格式化**: 使用 `.clang-format`，4 空格缩进，120 列宽
- **指针对齐**: 右对齐（`int *ptr`）
- **函数大括号**: 新行
- **控制语句大括号**: 同行

## 算子目录结构（非显而易见）

每个算子必须遵循以下结构，文件名必须与算子名一致：
```
<算子类别>/<算子名>/
├── op_host/
│   ├── <算子名>_def.cpp        # 算子定义（OP_ADD 宏注册）
│   ├── <算子名>_tiling.cpp     # Tiling 实现（IMPL_OP_OPTILING 宏注册）
│   └── CMakeLists.txt
├── op_kernel/
│   ├── <算子名>.cpp            # Kernel 入口
│   ├── <算子名>.h              # Kernel 实现
│   ├── <算子名>_tiling_data.h  # TilingData 结构
│   └── <算子名>_tiling_key.h   # TilingKey 定义
├── examples/
│   └── test_aclnn_<算子名>.cpp # eager 模式样例
├── tests/ut/                   # 单元测试
└── CMakeLists.txt
```

## 关键约定

- **算子命名**: 使用 snake_case（如 `flash_attention_score`）
- **Kernel 函数**: 必须使用 `__aicore__` 修饰符
- **Tiling 注册**: 使用 `IMPL_OP_OPTILING(OpName).Tiling(TilingFunc)` 宏
- **算子定义注册**: 使用 `OP_ADD(OpName)` 宏
- **SoC 版本**: 支持 ascend310p、ascend910b、ascend910_93、ascend910_95 等
- **测试配置**: 在 `tests/test_config.yaml` 中配置 CI 触发规则

## 依赖

- 需要安装 CANN Toolkit
- CANN Toolkit已经安装在ops-transformer-dev文件夹同级目录下，执行所有build.sh命令前只需要`source Ascend/ascend-toolkit/set_env.sh`即可
