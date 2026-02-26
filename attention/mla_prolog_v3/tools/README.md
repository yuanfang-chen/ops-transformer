# MlaPrologV3 Performance Model (Theoretical)

Standalone Python CLI to estimate MlaPrologV3 fused-kernel performance on Ascend 910B-style hardware.

What it models:
- Split-`N` kernel path only (host tiling currently forces `splitM=0`)
- Exact FLOPs for the 4 matmuls (`MatmulCq`, `MatmulCkvKr`, `MatmulQcQr`, `MatmulQn`)
- Approximate vector-side FLOPs (RMSNorm / RoPE / dequant / dynamic-quant)
- HBM vs L2 traffic (cold first-step and warm steady-state)
- Roofline-style stage time estimates + coarse fused CV overlap via a DAG scheduler

What it does not model:
- Cycle-accurate L0/L1/L2 behavior
- Exact synchronization penalty cycles
- Empty-tensor fast paths

## Usage

Shape mode (recommended):

```bash
python3 attention/mla_prolog_v3/tools/mla_prolog_perf_model.py \
  --input-mode shape \
  --input-json attention/mla_prolog_v3/tools/examples/mla_prolog_910b_example.json \
  --report table
```

Base-params mode (use existing tiling/base params directly):

```bash
python3 attention/mla_prolog_v3/tools/mla_prolog_perf_model.py \
  --input-mode base-params \
  --input-json attention/mla_prolog_v3/tools/examples/mla_prolog_base_params_example.json \
  --report both
```

Override performance constants:

```bash
python3 attention/mla_prolog_v3/tools/mla_prolog_perf_model.py \
  --input-mode shape \
  --input-json attention/mla_prolog_v3/tools/examples/mla_prolog_910b_example.json \
  --cube-peak-tflops 180 \
  --vector-peak-tflops 24 \
  --hbm-bw-gbps 1400 \
  --l2-bw-gbps 4000 \
  --cube-eff 0.7 \
  --vector-eff 0.6 \
  --report json \
  --json-out /tmp/mla_prolog_perf_report.json
```

## CLI Options (main)

- `--input-mode {shape,base-params}`
- `--input-json <path>`
- `--quant-mode <0..9>`
- `--cache-mode {PA_BSND,PA_NZ,PA_BLK_BSND,PA_BLK_NZ,BSND,TND}`
- `--query-norm-flag {0,1}`
- `--l2-mode {cold,warm,both}` (default: `both`)
- `--report {table,json,both}` (default: `both`)
- `--json-out <path>`

Performance knobs:
- `--cube-peak-tflops`
- `--vector-peak-tflops`
- `--hbm-bw-gbps`
- `--l2-bw-gbps`
- `--cube-eff`
- `--vector-eff`
- `--hbm-eff`
- `--l2-eff`
- `--l2-reserve-bytes`
- `--l2-reserve-ratio`

## JSON Input Schemas

### Shape mode (`ProblemSpec`)
Required keys:
- `T`, `He`, `Hcq`, `N`, `D`, `Dr`, `Hckv`, `Nkv`, `Dtile`

Optional keys:
- `B`, `S`, `cache_mode`, `quant_mode`, `query_norm_flag`
- `weight_quant_mode`, `kv_cache_quant_mode`, `query_quant_mode`
- `tile_size`, `qc_qr_scale`, `kc_scale`
- `hardware` (override defaults)
- `coefficients` (vector FLOP coefficients)

### Base-params mode (`BaseParamsSpec`)
- Either a root object with base-param fields or `{ "base_params": { ... } }`
- Needs the tiling/base fields used by the model (see example JSON)
- Put `quant_mode`, `cache_mode`, `query_norm_flag` at top level (or rely on defaults)

## Notes on HBM/L2 Modeling

- Matmul FLOPs are exact.
- HBM/L2 traffic is source-level and heuristic, not a hardware cache simulation.
- Workspace producer-consumer reuse is treated as L2.
- Warm mode retains persistent tensors in L2 using a deterministic budget/pinning policy (weights first, then small static params/scales).
- The script reports both a `cold_then_warm` kernel estimate (first step cold, rest warm) and a fully warm steady-state estimate.
