# FIAS Kernel Execution: Original vs Pybind Flow

## 1. Original Flow (CANN Framework)

```
Python / User Code
  │
  ▼
aclnnFusedInferAttentionScoreV5GetWorkspaceSize()     ← op_api/aclnn_fused_infer_attention_score_v5.cpp
  │  Converts aclTensor/aclTensorList → internal representations
  │  Creates "fake" tensors for null optional parameters
  │
  ▼
FusedInferAttentionScoreTilingImpl::DoOpTiling()       ← op_host/arch35/fused_infer_attention_score_impl.cpp
  │  Uses gert::TilingContext (context_) to get shapes/dtypes
  │  Calls SetPlatMemoryInfo() → platform core count, UB/L1/L0 sizes
  │  Calls SplitPolicy() → how to split B/N/S across cores
  │  Calls ComputeTilingData() → fills tiling struct fields
  │  Calls GenTilingKey() → encodes 12 template params into uint64_t
  │  Calls SetBlockDim() → AIC core count
  │  Calls GetWorkspace() → workspace size
  │  Serializes tiling data into aclOpExecutor
  │
  ▼
aclnnFusedInferAttentionScoreV5(workspace, executor, stream)
  │  Framework deserializes executor → launches kernel on device
  │
  ▼
__global__ fused_infer_attention_score<12 template params>(...)   ← op_kernel/fused_infer_attention_score_apt.cpp
  │  Receives __gm__ uint8_t* tiling (pointer to GM buffer)
  │  if quantMode >= 15 → prompt_flash_attention_FIAS_regbase<...>()
  │  else → incre_flash_attention_FIAS_regbase<...>()
  │
  ▼
incre_flash_attention_FIAS_regbase<...>(... __gm__ uint8_t* tiling)  ← incre_flash_attention_entry_regbase.h
  │  GET_TILING_DATA_WITH_STRUCT() copies tiling from GM → local variable
  │  #if ORIG_DTYPE_* selects the correct dtype path
  │  INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI():
  │    - Creates CubeBlockType / VecBlockType based on g_coreType
  │    - Instantiates kernel operator class
  │    - op.Init(query, key, value, ..., tilingData, &pipe)
  │    - op.Process()
```

### Key characteristics of the original flow:
- **Tensor info** obtained via `context_->GetInputShape()`, `context_->GetInputDataType()` etc.
- **Tiling data** serialized into an opaque GM buffer, kernel deserializes via macro
- **Template dispatch** handled by ASCENDC_TPL system (auto-generates specializations from tiling key bits)
- **Kernel launch** managed by CANN framework (async via executor)

---

## 2. New Pybind Flow

```
Python: ascendc_ops.fias_decode(query, key, value, ...)
  │
  ▼
fias_decode() in fias_torch.asc                        ← examples/fias/fias_torch.asc
  │
  │  ┌─ Step 1: Validate tensors on NPU
  │  │   checkTensorOnNPU(query, "query", false)
  │  │   checkTensorOnNPU(key, "key", false)  ...
  │  │
  │  ├─ Step 2: Extract tensor info into custom descriptors (NO context_)
  │  │   AddTensorDesc(query, ..., inTensors)    → {shape=[B,N,S1,D], dtype=BF16, isNull=false}
  │  │   AddTensorDesc(key, ..., inTensors)      → {shape=[B,Nkv,S2,D], dtype=BF16, isNull=false}
  │  │   AddTensorDesc(atten_mask, ..., inTensors) → {shape=[], dtype=MAX, isNull=true} or filled
  │  │
  │  ├─ Step 3: Pack scalar attributes
  │  │   FiasAttrTiling attrs = {numHeads, numKVHeads, scaleValue, ...}
  │  │
  │  ├─ Step 4: Allocate output tensor
  │  │   at::Tensor attentionOut = at::empty(outShape, bf16, npu)
  │  │
  │  ├─ Step 5: Compute tiling (custom class, no context_)
  │  │   FiasTilingData tilingData = {};       ← zero-initialized
  │  │   FiasTiling opTiling(&tilingData);
  │  │   opTiling.DoTiling(inTensors, attrs, outTensors,
  │  │                     tilingKey, workspaceSize, blockDim)
  │  │
  │  ├─ Step 6: Allocate workspace via aclrtMalloc
  │  │
  │  ├─ Step 7: Extract raw GM pointers from torch tensors
  │  │   q_ptr = query.data_ptr()
  │  │   k_ptr = key.data_ptr()  ...
  │  │
  │  ├─ Step 8: Dispatch to pre-compiled named kernel
  │  │   switch(tilingKey) {
  │  │     case FIAS_BF16_BNSD_C3_NOMASK_NOPA_NOFD:
  │  │       FiasKernel_BF16_BNSD_C3_NoMask_NoPA_NoFD<<<blockDim, nullptr, stream>>>(
  │  │           q_ptr, k_ptr, v_ptr, ..., tilingData);   ← tiling passed BY VALUE
  │  │   }
  │  │
  │  └─ Step 9: Free workspace, return attentionOut
  │
  ▼
__global__ FiasKernel_BF16_BNSD_C3_NoMask_NoPA_NoFD(   ← fias_torch.asc (DEFINE_FIAS_KERNEL macro)
    ..., FiasTilingData tilingData)                       ← received by value in input param buffer
  │
  │  Calls incre_flash_attention_FIAS_regbase<
  │      0,     // InOutLayoutType = BNSD
  │      3,     // Config = D128
  │      9,     // PseMode = NONE
  │      31,    // QuantMode = NoQuant
  │      false, // HasAttenMask
  │      false, // HasRope
  │      false, // IsPa
  │      false, // IsFd
  │      false, // EmptyTensor
  │      0,     // PFAMask
  │      0,     // PFAMatMulType
  │      false  // EnableKVPrefix
  │  >(..., (__gm__ uint8_t*)&tilingData)
  │         ↑ cast by-value param back to GM pointer
  │
  ▼
incre_flash_attention_FIAS_regbase<...>()                ← same IFA kernel as original
  │  GET_TILING_DATA_WITH_STRUCT() copies from GM → local
  │  (works because by-value params live in DDR input param buffer = GM space)
  │  op.Init() → op.Process()
```

---

## 3. What Changed and Why

### 3.1 Removed: CANN Framework Dependency (`context_`)

| Original | Pybind |
|----------|--------|
| `context_->GetInputShape(0)` | `inTensors[0].tShape` (from `tensor.sizes().vec()`) |
| `context_->GetInputDataType(0)` | `inTensors[0].tDataType` (from `scalarToFiasDataType` map) |
| `context_->GetOptionalInputShape(3)` | `inTensors[3].tIsNull` check |
| `context_->GetAttrs()` | `FiasAttrTiling attrs` struct |

**Why:** `context_` is a CANN framework abstraction (`gert::TilingContext`) that requires the full operator registration pipeline. By replacing it with `FiasTensorDescTiling` descriptors extracted directly from `torch::Tensor` via pybind, we eliminate the framework dependency entirely.

**Files involved:**
- `fias_tiling.h`: `FiasTensorDescTiling` struct + `FiasTiling` class
- `fias_torch.h`: `AddTensorDesc()` template that converts `torch::Tensor` → descriptor

### 3.2 Changed: Tiling Data Passing (GM pointer → by-value)

| Original | Pybind |
|----------|--------|
| Framework serializes `FusedInferAttentionScoreTilingData` into GM buffer | Host fills `FiasTilingData` struct directly |
| Kernel receives `__gm__ uint8_t* tiling` | Kernel receives `FiasTilingData tilingData` by value |
| Macro `GET_TILING_DATA_WITH_STRUCT` DMA-copies from GM → local | By-value param already in input param buffer (DDR) |

**Why:** By-value passing eliminates the serialize/deserialize overhead. The CCE runtime places by-value kernel parameters in a DDR input parameter buffer (controlled by `--cce-aicore-input-parameter-size=4096` in the original CCE build, or equivalently by the ASC compiler). Since DDR is global memory, `(__gm__ uint8_t*)&tilingData` yields a valid GM pointer that the existing `GET_TILING_DATA_WITH_STRUCT` macro can read from.

**Size constraint:** `FlashAttentionScoreSimplifiedTilingData` is ~1000 bytes:
- `InputParamsRegbase`: ~300B (dims, scale, flags)
- `MultiCoreParamsRegbase`: ~620B (coreNum, bnStartIdx[48], sparseStartIdx[48])
- `DropmaskParamsRegbase`: ~40B
- `InitOutputParams`: ~24B

This fits within the default 4096-byte parameter limit.

**Files involved:**
- `op_kernel/fias_tilingdata.h`: `using FiasTilingData = optiling::FlashAttentionScoreSimplifiedTilingData`
- `fias_torch.asc`: `DEFINE_FIAS_KERNEL` macro, last parameter is `FiasTilingData tilingData`

### 3.3 Changed: Template Dispatch (ASCENDC_TPL system → named kernel functions)

| Original | Pybind |
|----------|--------|
| `ASCENDC_TPL_ARGS_DECL` declares 12 template params | Same 12 params, but baked into function names |
| `ASCENDC_TPL_SEL` auto-generates instantiations per dtype combo | Manual `DEFINE_FIAS_KERNEL` macro for each variant |
| Single `__global__` function, framework selects specialization via tiling key | Multiple `__global__` functions, host `switch(tilingKey)` selects |

**Why:** The ASCENDC_TPL system requires the CANN compilation pipeline to generate and register template specializations. With pybind, we manually instantiate the exact combinations we need as named kernel functions, and dispatch at the host level using a simple switch statement.

**Phase 1 template parameters (fixed for all 8 variants):**
- `InOutLayoutType = 0` (BNSD)
- `PseMode = 9` (NONE)
- `QuantMode = 31` (NoQuant)
- `HasRope = false`
- `EmptyTensor = false`
- `PFAMask = 0`
- `PFAMatMulType = 0`
- `EnableKVPrefix = false`

**Phase 1 template parameters (varying across 8 variants):**
- `Config`: 1 (D64) or 3 (D128)
- `HasAttenMask`: true or false
- `IsPa`: true or false
- `IsFd`: false (all decode, no flash-decode)

**Files involved:**
- `fias_torch.asc`: `DEFINE_FIAS_KERNEL` macro + 8 instantiations + `switch(tilingKey)` dispatch
- `fias_tiling.h`: `ComputeTilingKey()` returns `FiasTilingKey` enum

### 3.4 Changed: Kernel Launch (framework → direct `<<<>>>`)

| Original | Pybind |
|----------|--------|
| `aclnnFusedInferAttentionScoreV5(workspace, size, executor, stream)` | `FiasKernel_XXX<<<blockDim, nullptr, stream>>>(...)` |
| Framework manages stream, sync, error handling | Manual stream from `c10_npu::getCurrentNPUStream()` |
| Workspace via framework allocation | Manual `aclrtMalloc` / `aclrtFree` |

**Why:** Direct kernel launch removes framework overhead (op registration, graph compilation, executor creation). The kernel is launched immediately on the current NPU stream.

**Files involved:**
- `fias_torch.asc`: `FIAS_KERNEL_LAUNCH` macro + workspace management

### 3.5 Changed: Build System (CANN op build → ASC + pybind11)

| Original | Pybind |
|----------|--------|
| CANN operator build system (CMakeLists in op_host/op_kernel) | `find_package(ASC)` + `pybind11_add_module` |
| `.cpp` compiled by CCE with `--cce-soc-version` flags | `.asc` compiled by ASC with `--npu-arch=dav-2201` |
| Op registered via CANN framework macros | Module registered via `PYBIND11_MODULE(ascendc_ops, m)` |
| Installed as part of CANN custom op package | Installed via `pip install -e .` |

**Files involved:**
- `CMakeLists.txt`: ASC compiler detection, pybind11 integration, linking against `torch_npu`
- `setup.py`: CMakeExtension + CMakeBuild for pip packaging

### 3.6 Unchanged: Kernel Implementation

The actual IFA kernel code (`incre_flash_attention_FIAS_regbase` and everything it calls) is **completely unchanged**. The pybind wrapper:
1. Includes the same headers: `incre_flash_attention_entry_regbase.h`
2. Calls the same function: `incre_flash_attention_FIAS_regbase<...>()`
3. Passes the same tiling struct: `FlashAttentionScoreSimplifiedTilingData`
4. Uses the same 12 template parameters

The only adaptation is the last argument: `(__gm__ uint8_t*)&tilingData` instead of the framework's serialized buffer pointer.

---

## 4. File Map

```
examples/fias/
├── CMakeLists.txt              # ASC + pybind11 build
├── setup.py                    # pip install packaging
├── fias_torch.asc              # Entry: pybind11 module + 8 named kernels + dispatch
├── fias_torch.h                # Utilities: checkTensorOnNPU, AddTensorDesc, get_first_tensor_address
├── fias_tiling.h               # Custom tiling: FiasTiling class, FiasTensorDescTiling, FiasAttrTiling
├── op_kernel/
│   └── fias_tilingdata.h       # Type alias: FiasTilingData = FlashAttentionScoreSimplifiedTilingData
└── tests/
    └── test_fias.py            # Python tests
```

Kernel headers reused from the original implementation (included, not modified):
- `attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h`
- `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h`
- `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h`
