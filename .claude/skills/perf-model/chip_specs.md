# Ascend NPU Chip Specifications

## Ascend 950

| Parameter | Value |
|-----------|-------|
| AI Core (Cube) count | 32 |
| AI Vector count | 64 |
| Frequency | 1.65 GHz |

### Compute Throughput (per core, per cycle)

| Unit | BF16 | INT8 | FP8 |
|------|------|------|-----|
| MMAD (MACs/cycle) | 4096 | 4096 | 8192 |
| MMAD (FLOPS/cycle, ×2) | 8192 | 8192 | 16384 |

Chip total: BF16 = 432.5 TFLOPS, FP8 = 865.1 TFLOPS

### Bandwidth

**Chip-total bandwidth caps**:

| Bus | Chip Total |
|-----|-----------|
| HBM | 1.6 TB/s (1600 GB/s) |
| L2 | 5.2 TB/s (5200 GB/s) |

**Per-core bandwidth (bytes/cycle)**:

| Bus | B/cycle | GB/s @1.65GHz | Contention |
|-----|---------|---------------|-----------|
| HBM→L1 | 33.3 | 54.9 (theoretical) | Yes: min(theoretical, chip_total/active_cores) |
| L2→Core | 108.0 | 178.2 (theoretical) | Yes: min(theoretical, chip_total/active_cores) |
| L1→L0A | 256.0 | 422.4 | No (per-core local bus) |
| L1→L0B | 256.0 | 422.4 | No |
| L0C→FixPipe | 256.0 | 422.4 | No |
| UB→L1 | 245.0 | 404.2 | No |
| UB→Reg | 512.0 | 844.8 | No |

**Effective per-core bandwidth at 32 cores**:
- HBM: min(54.9, 1600/32) = **50.0 GB/s**
- L2: min(178.2, 5200/32) = **162.5 GB/s**

### Cache Sizes (per core)

| Cache | Size |
|-------|------|
| UB (Unified Buffer) | 256 KB |
| L1 | 1024 KB |
| L0A (input A) | 256 KB |
| L0B (input B) | 256 KB |
| L0C (accumulator) | 512 KB |

### Contention Rule

When `active_cores × per_core_bw > chip_total_bw`:
```
effective_per_core_bw = chip_total_bw / active_cores
```

### Mixed HBM/L2 Bandwidth

- First access to data: HBM bandwidth (cold path)
- Reused data (e.g., shared A matrix in split-N): L2 bandwidth (warm path)
- Cross-matmul reuse: if matmul B uses the same input as matmul A, it hits L2

---

## Ascend 910B

| Parameter | Value |
|-----------|-------|
| AI Core (Cube) count | 24 |
| AI Vector count | 48 |
| BF16 compute | 320.0 TFLOPS |
| INT8 compute | 640.0 TOPS |
| FP8 compute | 640.0 TFLOPS |
| HBM bandwidth | 1.2 TB/s |
| L2 bandwidth | 6.4 TB/s |
| Vector BF16 | 10.0 TOPS |

### Cache Sizes (per core)

| Cache | Size |
|-------|------|
| UB | 192 KB |
| L1 | 512 KB |
| L0A | 64 KB |
| L0B | 64 KB |
| L0C | 128 KB |

---

## Data Type Sizes

| Type | Bytes | Used In |
|------|-------|---------|
| BF16 / FP16 | 2 | Default activation/weight |
| INT8 | 1 | Quantized weight |
| FP8 (E4M3) | 1 | MXFP8 activation/weight |
| FP32 / float | 4 | INT8/FP8 matmul output accumulator |
| INT32 | 4 | INT8 matmul output accumulator |

### Quantization Impact on Matmul

| Quant Mode | dtype_a (activation) | dtype_b (weight) | dtype_c (output) |
|------------|---------------------|------------------|-----------------|
| BF16 (no quant) | 2B | 2B | 2B |
| INT8 partial | 2B | 1B | 4B |
| INT8 full | 1B | 1B | 4B |
| MXFP8 | 1B | 1B | 4B |
