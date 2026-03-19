# AscendC Flash Attention 2 — Ascend 910A

自定义 AscendC kernel 实现 Flash Attention 2 算法，运行于昇腾 910A (c100) 架构。

## 架构

```
ascendc_fa2/
├── fa2_tiling.h        # Tiling 数据结构（Host ↔ Device 共享）
├── fa2_kernel.cpp      # AscendC kernel：在线 softmax + 分块注意力
├── test_fa2_host.cpp   # C++ 测试程序：ACL 运行时启动 kernel
├── CMakeLists.txt      # 构建系统（ascendc_library）
└── README.md
```

## 算法

实现了完整的 Flash Attention 2 在线 softmax 算法：

1. **分块计算** — Q 按 blockM 分块，K/V 按 blockN 分块
2. **QK^T 点积** — 逐元素计算（后续可优化为 Cube Matmul）
3. **在线 Softmax** — 增量更新 row max 和 row sum
   - 使用 AscendC `Exp`/`Adds` 向量指令加速
   - 校正因子 `exp(oldMax - newMax)` 修正历史累积
4. **PV 累积** — 加权 Value 求和
5. **归一化输出** — 使用 AscendC `Muls` 向量指令

支持：
- ✅ Causal masking
- ✅ 任意序列长度（自动分块 + 尾块处理）
- ✅ fp16 输入/输出，fp32 内部计算
- ✅ 多 batch / 多 head 并行（每个 head 占一个 AI Core）

## 构建

```bash
source /usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux/bin/setenv.bash
cd ascendc_fa2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 运行测试

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# 基本测试：device=4, seq_len=128, causal=1
./test_fa2_host 4 128 1

# 批量测试
for S in 32 64 128 256; do
  for C in 0 1; do
    ./test_fa2_host 4 $S $C
  done
done
```

## 测试结果

| seq_len | causal | max_diff | 状态 |
|---------|--------|----------|------|
| 32      | ✗      | 0.000244 | PASS |
| 32      | ✓      | 0.000488 | PASS |
| 64      | ✗      | 0.000244 | PASS |
| 64      | ✓      | 0.000488 | PASS |
| 128     | ✗      | 0.000122 | PASS |
| 128     | ✓      | 0.000488 | PASS |
| 256     | ✗      | 0.000122 | PASS |
| 256     | ✓      | 0.000488 | PASS |

所有误差均在 fp16 精度范围内（< 0.001）。

## 后续优化方向

1. **Cube Matmul 加速** — 将 QK^T 和 PV 点积替换为 `Matmul` 高阶 API
2. **DataCopy 批量搬运** — 替换逐元素 GM 读取为 DataCopy 块搬运
3. **多核 M 维度并行** — 将 Q 的 blockM 分配到多个 AI Core
4. **流水线优化** — 利用 MTE2/V/MTE3 流水线隐藏延迟
5. **支持 GQA** — Group-Query Attention（KV heads < Q heads）
