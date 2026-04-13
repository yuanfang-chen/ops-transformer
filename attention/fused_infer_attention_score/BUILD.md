## How `_apt.cpp` Gets Compiled

The `_apt.cpp` file is **not referenced directly in any CMakeLists.txt**. Instead, the CANN build system discovers it through the **operator definition** in `fused_infer_attention_score_def.cpp`.

### The Mechanism

The op definition registers **two hardware configs**, each pointing to a different kernel source file via `opFile.value`:

```cpp
// Line 568 — for ascend910b / ascend910_93 (Arch32)
.ExtendCfgInfo("opFile.value", "fused_infer_attention_score")
this->AICore().AddConfig("ascend910b", aicore_config);
this->AICore().AddConfig("ascend910_93", aicore_config);

// Line 2090 — for ascend950 (Arch35)
.ExtendCfgInfo("opFile.value", "fused_infer_attention_score_apt")
this->AICore().AddConfig("ascend950", aicore_config_95);
```

### The Build Flow

```
fused_infer_attention_score_def.cpp
  │
  │  OP_ADD(FusedInferAttentionScore, ...)
  │  registers two AICore configs with different opFile.value
  │
  ▼
CANN build system (cmake/scripts/util/opdesc_parser.py + opbuild.cmake)
  │
  │  Reads opFile.value per SoC config
  │  Looks for matching .cpp in op_kernel/
  │
  ├── ascend910b/910_93 → op_kernel/fused_infer_attention_score.cpp
  │
  └── ascend950          → op_kernel/fused_infer_attention_score_apt.cpp
```

The `op_host/CMakeLists.txt` only declares the **build dependency** chain:

```cmake
# _apt variant depends on IFA + PFA + common (because it #includes them)
set(fused_infer_attention_score_apt_depends
    attention/incre_flash_attention
    attention/prompt_flash_attention
    attention/common)

# plain variant has the same dependencies
set(fused_infer_attention_score_depends
    attention/incre_flash_attention
    attention/prompt_flash_attention
    attention/common)
```

### Summary

There's no explicit `add_executable` or `target_sources` for the kernel `.cpp`. The CANN framework uses a **convention-based build**: the `opFile.value` string maps to `op_kernel/{value}.cpp`, and the build system compiles the right file for the target SoC. That's why you see two parallel kernel files — one per hardware generation — selected entirely by the operator definition metadata.