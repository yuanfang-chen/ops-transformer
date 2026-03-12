# MlaPrologV3 ST Coverage Report

- Hardware profile: AIC=24, AIV=48, CV=1:2
- Candidate count: 17277
- Selected positive cases: 10
- Reachable tag count: 68
- Uncovered reachable tags: 0

## Factor Summary
- `weight_quant_mode / kv_quant_mode / query_quant_mode`: QUANT_MODE routing and dynamic-quant/dequant paths
- `cache_mode`: ND/PA scatter path family and per-tile legality
- `bs_fused_flag`: tokenX 2D fused path; gates actualSeqMode for PA_BLK cache modes
- `ckvkr_repo_mode / quant_scale_repo_mode / tile_size`: per-tile storage path and dtile constraints
- `batch_size / q_seq`: token count, multi-step outer loop, vector split tail, and step batch tail
- `q_head_num`: mm3/mm4 split and tail behavior, dequant-opt gating
- `He`: mm1/mm2 stepK (3/4), kL1 loops, and baseK path
- `block_size`: PA block behavior and legality
- `kv_head_num`: fixed positive at 1; invalid values used for negative runtime cases
- `query_norm_flag`: query_norm output path
- `qc_qr_scale / kc_scale`: scale-enable branches for query/ckv normalization
- `smooth_scales_cq_flag`: optional smooth scales for dynamic quant
- `hardware profile (aic_num/aiv_num)`: cube/vector partition and tail reachability

## Tree Map
```text
MlaPrologV3 Condition Coverage
├─ Quant Scenario
│  ├─ qm0 (reachable)
│  ├─ qm1 (reachable)
│  ├─ qm2 (reachable)
│  ├─ qm3 (reachable)
│  ├─ qm4 (reachable)
│  ├─ qm5 (reachable)
│  ├─ qm6 (reachable)
│  ├─ qm7 (reachable)
│  ├─ qm8 (reachable)
│  ├─ qm9 (reachable)
├─ Cache Path
│  ├─ BSND (reachable)
│  ├─ PA_BLK_BSND (reachable)
│  ├─ PA_BLK_NZ (reachable)
│  ├─ PA_BSND (reachable)
│  ├─ PA_NZ (reachable)
│  ├─ TND (reachable)
├─ Tiling Flags
│  ├─ enableDequantOpt = {0,1} (reachable, reachable)
│  ├─ enableGroupComputeOpt = {0} [unreachable=1; current V3 contract fixes Nkv=1]
│  ├─ bsFusedFlag = {0,1} (reachable, reachable)
│  ├─ actualSeqMode = {DISABLED, EN_Q_LEN} (reachable, reachable)
│  ├─ splitMFlag = 0 (fixed by tiling)
│  ├─ emptyTensorMode = NON_EMPTY (positive set), EMPTY_* via negative/runtime-fail set
│  └─ cvMode = 1:2
├─ Cube Core Tails
│  ├─ mm1 tail yes/no (unreachable, reachable)
│  ├─ mm2 tail yes/no (unreachable, reachable)
│  ├─ mm3 tail yes/no (reachable, reachable)
│  └─ mm4 tail yes/no (reachable, reachable)
├─ L1 Loops
│  ├─ nL1 tail mm1 yes/no (reachable, reachable)
│  ├─ nL1 tail mm2 yes/no (reachable, unreachable)
│  ├─ nL1 tail mm3 yes/no (reachable, reachable)
│  ├─ kL1 loops mm12 single/multi (reachable, reachable)
│  └─ stepK mode {3,4} (reachable, reachable)
├─ L0 Loops
│  └─ baseK inner loops stepK {3,4} (reachable, reachable)
├─ Vector Core
│  ├─ multi-step outer loop yes/no (reachable, reachable)
│  ├─ step batch tail yes/no (reachable, reachable)
│  ├─ vector token split tail yes/no (reachable, reachable)
│  ├─ paged scatter spill(rows>1) [unreachable; vectorRow is fixed to 1]
│  └─ active/inactive vector lanes (reachable, reachable)
└─ Postprocess
   ├─ needQnDynamicQuant yes/no (reachable, reachable)
   ├─ isPertile yes/no (reachable, reachable)
   ├─ qc_qr_scale_enable yes/no (reachable, reachable)
   ├─ kc_scale_enable yes/no (reachable, reachable)
   ├─ smooth_scales_cq yes/no (reachable, reachable)
   └─ query_norm_flag yes/no (reachable, reachable)
```

## Known Unreachable Conditions
- `splitMFlag=1`: host tiling hardcodes the split-N template.
- `enableGroupComputeOpt=1`: current V3 contract fixes `Nkv=1`, while this path requires `Nkv=8`.
- `PA_BLK spill(rows>1)`: kernel `vectorRow_` is fixed to `1`, so the spill branch is not taken.
- `cvMode=1:1`: not part of the current default hardware profile (`AIC=24`, `AIV=48`).

## Selected Cases
### coverage_case_000

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 7680,
  "batch_size": 9,
  "block_size": 16,
  "bs_fused_flag": 1,
  "cache_mode": "PA_BLK_BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 1,
  "q_head_num": 128,
  "q_seq": 16,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 1,
  "query_quant_mode": 1,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 1,
  "tile_size": 128,
  "weight_quant_mode": 3
}
```

- Covered tags:
  - `cache_mode:PA_BLK_BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:1`
  - `cube:mm4_tail:1`
  - `l0:mm12_basek_stepk:3`
  - `l1:mm12_kl1_loops:multi`
  - `l1:mm12_stepk:3`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:0`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:multi`
  - `l1:mm3_nl1_tail:0`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:1`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:1`
  - `post:smooth_scales_cq:1`
  - `quant_mode:qm8`
  - `tiling:actual_seq_mode:en_q_len`
  - `tiling:bs_fused_flag:1`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:1`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:0`
  - `vector:multi_step_loop:1`
  - `vector:step_batch_tail:1`
  - `vector:token_split_tail:1`

### coverage_case_001

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "PA_BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 1,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 3,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 1,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 2
}
```

- Covered tags:
  - `cache_mode:PA_BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:single`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:1`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm6`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_002

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "PA_BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.1,
  "kv_head_num": 1,
  "kv_quant_mode": 0,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.1,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 0
}
```

- Covered tags:
  - `cache_mode:PA_BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:multi`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:1`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:1`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm0`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_003

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "PA_BLK_NZ",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 0,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 1
}
```

- Covered tags:
  - `cache_mode:PA_BLK_NZ`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:multi`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm1`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_004

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "PA_NZ",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 2,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 1
}
```

- Covered tags:
  - `cache_mode:PA_NZ`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:multi`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm2`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_005

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 1,
  "cache_mode": "TND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 1,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 3,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 1,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 1
}
```

- Covered tags:
  - `cache_mode:TND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:multi`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:1`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm5`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:1`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_006

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 0,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 2
}
```

- Covered tags:
  - `cache_mode:BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:single`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm3`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_007

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 1,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 0,
  "query_quant_mode": 1,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 2
}
```

- Covered tags:
  - `cache_mode:BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:single`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:1`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:single`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:1`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm4`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:0`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_008

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 0,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 0,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 0,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 3
}
```

- Covered tags:
  - `cache_mode:BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:single`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:0`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:multi`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:0`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm7`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:1`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

### coverage_case_009

```json
{
  "Hckv": 512,
  "Hcq": 1536,
  "He": 1024,
  "batch_size": 1,
  "block_size": 16,
  "bs_fused_flag": 0,
  "cache_mode": "BSND",
  "ckv_epsilon": 0.0005,
  "ckvkr_repo_mode": 1,
  "cq_epsilon": 0.0005,
  "dtype": "torch.bfloat16",
  "head_dim": 128,
  "input_layout": "BSH",
  "kc_scale": 1.0,
  "kv_head_num": 1,
  "kv_quant_mode": 3,
  "q_head_num": 1,
  "q_seq": 1,
  "qc_qr_scale": 1.0,
  "quant_scale_repo_mode": 1,
  "query_norm_flag": 0,
  "query_quant_mode": 0,
  "rope_head_dim": 64,
  "smooth_scales_cq_flag": 0,
  "tile_size": 128,
  "weight_quant_mode": 3
}
```

- Covered tags:
  - `cache_mode:BSND`
  - `cube:mm1_tail:0`
  - `cube:mm2_tail:0`
  - `cube:mm3_tail:0`
  - `cube:mm4_tail:0`
  - `l0:mm12_basek_stepk:4`
  - `l1:mm12_kl1_loops:single`
  - `l1:mm12_stepk:4`
  - `l1:mm1_nl1_loops:single`
  - `l1:mm1_nl1_tail:0`
  - `l1:mm2_nl1_loops:single`
  - `l1:mm2_nl1_tail:1`
  - `l1:mm3_kl1_loops:multi`
  - `l1:mm3_nl1_loops:multi`
  - `l1:mm3_nl1_tail:1`
  - `post:is_pertile:1`
  - `post:kc_scale_enable:0`
  - `post:need_qn_dynamic_quant:0`
  - `post:qc_qr_scale_enable:0`
  - `post:query_norm_flag:0`
  - `post:smooth_scales_cq:0`
  - `quant_mode:qm9`
  - `tiling:actual_seq_mode:disabled`
  - `tiling:bs_fused_flag:0`
  - `tiling:cv_mode:1:2`
  - `tiling:empty_tensor_mode:non_empty`
  - `tiling:enable_dequant_opt:1`
  - `tiling:enable_group_compute_opt:0`
  - `tiling:split_m_mode:0`
  - `vector:inactive_lanes:1`
  - `vector:multi_step_loop:0`
  - `vector:step_batch_tail:0`
  - `vector:token_split_tail:0`

## Case-Condition Matrix
| condition tag | coverage_case_000 | coverage_case_001 | coverage_case_002 | coverage_case_003 | coverage_case_004 | coverage_case_005 | coverage_case_006 | coverage_case_007 | coverage_case_008 | coverage_case_009 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cache_mode:BSND` |  |  |  |  |  |  | Y | Y | Y | Y |
| `cache_mode:PA_BLK_BSND` | Y |  |  |  |  |  |  |  |  |  |
| `cache_mode:PA_BLK_NZ` |  |  |  | Y |  |  |  |  |  |  |
| `cache_mode:PA_BSND` |  | Y | Y |  |  |  |  |  |  |  |
| `cache_mode:PA_NZ` |  |  |  |  | Y |  |  |  |  |  |
| `cache_mode:TND` |  |  |  |  |  | Y |  |  |  |  |
| `cube:mm1_tail:0` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `cube:mm2_tail:0` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `cube:mm3_tail:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `cube:mm3_tail:1` | Y |  |  |  |  |  |  |  |  |  |
| `cube:mm4_tail:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `cube:mm4_tail:1` | Y |  |  |  |  |  |  |  |  |  |
| `l0:mm12_basek_stepk:3` | Y |  |  |  |  |  |  |  |  |  |
| `l0:mm12_basek_stepk:4` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `l1:mm12_kl1_loops:multi` | Y |  | Y | Y | Y | Y |  |  |  |  |
| `l1:mm12_kl1_loops:single` |  | Y |  |  |  |  | Y | Y | Y | Y |
| `l1:mm12_stepk:3` | Y |  |  |  |  |  |  |  |  |  |
| `l1:mm12_stepk:4` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `l1:mm1_nl1_loops:single` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `l1:mm1_nl1_tail:0` | Y |  |  |  |  |  |  |  | Y | Y |
| `l1:mm1_nl1_tail:1` |  | Y | Y | Y | Y | Y | Y | Y |  |  |
| `l1:mm2_nl1_loops:single` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `l1:mm2_nl1_tail:1` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `l1:mm3_kl1_loops:multi` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `l1:mm3_nl1_loops:multi` | Y |  |  |  |  |  |  |  | Y | Y |
| `l1:mm3_nl1_loops:single` |  | Y | Y | Y | Y | Y | Y | Y |  |  |
| `l1:mm3_nl1_tail:0` | Y |  |  |  |  |  |  |  |  |  |
| `l1:mm3_nl1_tail:1` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `post:is_pertile:0` | Y |  | Y | Y | Y |  | Y | Y | Y |  |
| `post:is_pertile:1` |  | Y |  |  |  | Y |  |  |  | Y |
| `post:kc_scale_enable:0` | Y | Y |  | Y | Y | Y | Y | Y | Y | Y |
| `post:kc_scale_enable:1` |  |  | Y |  |  |  |  |  |  |  |
| `post:need_qn_dynamic_quant:0` |  | Y | Y | Y | Y | Y | Y |  | Y | Y |
| `post:need_qn_dynamic_quant:1` | Y |  |  |  |  |  |  | Y |  |  |
| `post:qc_qr_scale_enable:0` | Y | Y |  | Y | Y | Y | Y | Y | Y | Y |
| `post:qc_qr_scale_enable:1` |  |  | Y |  |  |  |  |  |  |  |
| `post:query_norm_flag:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `post:query_norm_flag:1` | Y |  |  |  |  |  |  |  |  |  |
| `post:smooth_scales_cq:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `post:smooth_scales_cq:1` | Y |  |  |  |  |  |  |  |  |  |
| `quant_mode:qm0` |  |  | Y |  |  |  |  |  |  |  |
| `quant_mode:qm1` |  |  |  | Y |  |  |  |  |  |  |
| `quant_mode:qm2` |  |  |  |  | Y |  |  |  |  |  |
| `quant_mode:qm3` |  |  |  |  |  |  | Y |  |  |  |
| `quant_mode:qm4` |  |  |  |  |  |  |  | Y |  |  |
| `quant_mode:qm5` |  |  |  |  |  | Y |  |  |  |  |
| `quant_mode:qm6` |  | Y |  |  |  |  |  |  |  |  |
| `quant_mode:qm7` |  |  |  |  |  |  |  |  | Y |  |
| `quant_mode:qm8` | Y |  |  |  |  |  |  |  |  |  |
| `quant_mode:qm9` |  |  |  |  |  |  |  |  |  | Y |
| `tiling:actual_seq_mode:disabled` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `tiling:actual_seq_mode:en_q_len` | Y |  |  |  |  |  |  |  |  |  |
| `tiling:bs_fused_flag:0` |  | Y | Y | Y | Y |  | Y | Y | Y | Y |
| `tiling:bs_fused_flag:1` | Y |  |  |  |  | Y |  |  |  |  |
| `tiling:cv_mode:1:2` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `tiling:empty_tensor_mode:non_empty` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `tiling:enable_dequant_opt:0` |  | Y | Y | Y | Y | Y | Y | Y |  |  |
| `tiling:enable_dequant_opt:1` | Y |  |  |  |  |  |  |  | Y | Y |
| `tiling:enable_group_compute_opt:0` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `tiling:split_m_mode:0` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `vector:inactive_lanes:0` | Y |  |  |  |  |  |  |  |  |  |
| `vector:inactive_lanes:1` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `vector:multi_step_loop:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `vector:multi_step_loop:1` | Y |  |  |  |  |  |  |  |  |  |
| `vector:step_batch_tail:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `vector:step_batch_tail:1` | Y |  |  |  |  |  |  |  |  |  |
| `vector:token_split_tail:0` |  | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `vector:token_split_tail:1` | Y |  |  |  |  |  |  |  |  |  |

## Uncovered Reachable Tags
- None

## Reachable Tags
- `cache_mode:BSND`
- `cache_mode:PA_BLK_BSND`
- `cache_mode:PA_BLK_NZ`
- `cache_mode:PA_BSND`
- `cache_mode:PA_NZ`
- `cache_mode:TND`
- `cube:mm1_tail:0`
- `cube:mm2_tail:0`
- `cube:mm3_tail:0`
- `cube:mm3_tail:1`
- `cube:mm4_tail:0`
- `cube:mm4_tail:1`
- `l0:mm12_basek_stepk:3`
- `l0:mm12_basek_stepk:4`
- `l1:mm12_kl1_loops:multi`
- `l1:mm12_kl1_loops:single`
- `l1:mm12_stepk:3`
- `l1:mm12_stepk:4`
- `l1:mm1_nl1_loops:single`
- `l1:mm1_nl1_tail:0`
- `l1:mm1_nl1_tail:1`
- `l1:mm2_nl1_loops:single`
- `l1:mm2_nl1_tail:1`
- `l1:mm3_kl1_loops:multi`
- `l1:mm3_nl1_loops:multi`
- `l1:mm3_nl1_loops:single`
- `l1:mm3_nl1_tail:0`
- `l1:mm3_nl1_tail:1`
- `post:is_pertile:0`
- `post:is_pertile:1`
- `post:kc_scale_enable:0`
- `post:kc_scale_enable:1`
- `post:need_qn_dynamic_quant:0`
- `post:need_qn_dynamic_quant:1`
- `post:qc_qr_scale_enable:0`
- `post:qc_qr_scale_enable:1`
- `post:query_norm_flag:0`
- `post:query_norm_flag:1`
- `post:smooth_scales_cq:0`
- `post:smooth_scales_cq:1`
- `quant_mode:qm0`
- `quant_mode:qm1`
- `quant_mode:qm2`
- `quant_mode:qm3`
- `quant_mode:qm4`
- `quant_mode:qm5`
- `quant_mode:qm6`
- `quant_mode:qm7`
- `quant_mode:qm8`
- `quant_mode:qm9`
- `tiling:actual_seq_mode:disabled`
- `tiling:actual_seq_mode:en_q_len`
- `tiling:bs_fused_flag:0`
- `tiling:bs_fused_flag:1`
- `tiling:cv_mode:1:2`
- `tiling:empty_tensor_mode:non_empty`
- `tiling:enable_dequant_opt:0`
- `tiling:enable_dequant_opt:1`
- `tiling:enable_group_compute_opt:0`
- `tiling:split_m_mode:0`
- `vector:inactive_lanes:0`
- `vector:inactive_lanes:1`
- `vector:multi_step_loop:0`
- `vector:multi_step_loop:1`
- `vector:step_batch_tail:0`
- `vector:step_batch_tail:1`
- `vector:token_split_tail:0`
- `vector:token_split_tail:1`
