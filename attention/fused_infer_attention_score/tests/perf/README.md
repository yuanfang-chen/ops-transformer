# FIAS NPU kernel-time benchmark

A small standalone benchmark that times `aclnnFusedInferAttentionScoreV5`
on a fixed decode-shaped input. Intended for quick before/after comparisons
when refactoring the FIAS dispatch table or kernel code on `ascend950`.

Why this and not pytest:

- Pytest needs `torch_npu`, which currently fails to import on the a5
  container (`libhccl.so` not on `LD_LIBRARY_PATH`). The C++ harness here
  links directly against `libascendcl` / `libnnopbase` / `libcust_opapi`
  and avoids the python stack entirely.
- Pytest is correctness-oriented; per-test wall-clock noise is high
  (Python overhead, decorator stacks, host-side timers aren't stream-aware).
  `aclrtEvent` based timing in C++ gives sub-microsecond resolution and
  matches the `experimental/mhc/mhc_pre/test/perf_test.cpp` precedent.

## Files

- `fias_perf.cpp` — entry point. Allocates Q/K/V/O for a single decode
  shape, runs `aclnnFusedInferAttentionScoreV5GetWorkspaceSize` once,
  then 5 warmup + 100 timed iterations of `aclnnFusedInferAttentionScoreV5`.
  Reports median / min / p95 / max in microseconds.
- `run.sh` — sources the installed run package's `set_env.bash`, compiles
  `fias_perf.cpp` via `g++` with the same include / link flags
  `build.sh --run_example` uses (see `build.sh:589-596`), and executes the
  binary.

## Default shape

Decode-style: batch=1, seqlen_q=1, seqlen_kv=4096, num_q_heads=32,
num_kv_heads=8 (GQA 4:1), head_dim=128, fp16, layout BNSD, no attention
mask, no quantization, with PSE (per-head fp32 scalar, pseType=2 — same
as the in-tree FIAS V5 example).

To change shape or iteration count, edit the `Config` struct at the top
of `fias_perf.cpp`.

## Usage

```
# 1. From the repo root, build the FIAS run package on a real ascend950
#    host. (See ../../README.md or repo CLAUDE.md for full build steps.)
bash build.sh --pkg --ops=fused_infer_attention_score --soc=ascend950

# 2. Install the run package INTO the OPP tree, not into a top-level
#    custom location. Two reasons:
#    - --install-path=$(realpath $ASCEND_HOME_PATH/../) puts files at
#      /usr/local/Ascend/vendors/custom_transformer/, which leaves any
#      pre-existing ${ASCEND_OPP_PATH}/vendors/custom_transformer/ install
#      intact and creates a dual-install conflict that op_master may
#      resolve in confusing ways.
#    - Installing directly to ASCEND_OPP_PATH gives op_master one
#      canonical vendor and matches the path the in-tree example expects.
#    If a previous install of a DIFFERENT ops set is already there
#    (e.g. apply_rotary_pos_emb), wipe it first:
#      rm -rf $ASCEND_OPP_PATH/vendors/custom_transformer
#    Then:
./build_out/cann-ops-transformer-*.run --quiet --install-path="${ASCEND_OPP_PATH}"

# 3. Run the benchmark.
cd attention/fused_infer_attention_score/tests/perf
bash run.sh
```

Expected output on success is one line:

```
fias_perf decode  B=1 Sq=2 Skv=2 Nq=2 Nkv=2 D=16 BNSD fp16  median=NN.NN us  min=...  p95=...  max=...  iters=100  warmup=5
```

If you see `ACL error 507015 at ... (aclrtSynchronizeStream)`, the
kernel launched but aborted asynchronously. This was the symptom of
a wiring bug in the type-erasure refactor where
`constInfo.hasAttenMaskRT` got the (always-true) template param
instead of a true runtime signal — fixed by reading
`tilingData->inputParamsRegbase.attenMaskS1Size > 0` at the 6 init
sites.

## Diffing across a refactor

```
# Before refactor:
bash run.sh | tee /tmp/fias_before.txt

# Refactor, rebuild the run package, reinstall, then:
bash run.sh | tee /tmp/fias_after.txt

diff /tmp/fias_before.txt /tmp/fias_after.txt
```

Acceptance bar for the planned compile-time-reduction refactors
(template-table shrinks, runtime guard additions): **median should drop
or stay the same**. A regression in median for a refactor that's only
supposed to reduce instantiation count is a flag to investigate.

## Out of scope (deliberate)

- Multiple shapes / sweep — start with one decode case. Add prefill
  (seqlen_q > 1, causal mask) once decode is producing stable numbers.
- Comparison against torch_npu reference — measuring NPU kernel time, not
  framework overhead.
- ascend910b — the arch35-only refactors don't touch the arch32 dispatch
  path. Building this on a2 is a separate exercise.
- CI gate integration — currently invoked ad-hoc via `remote-exec.yml` or
  manually. Promote to a `bench` gate in `pre_commit_check.py` once the
  numbers are reliable.
