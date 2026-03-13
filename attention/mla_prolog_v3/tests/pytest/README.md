# MlaPrologV3 Pytest Guide

This folder contains the pytest-based validation project for `torch_npu.npu_mla_prolog_v3`.
The test flow builds a CPU reference implementation, runs the NPU operator on device `0`,
and compares both functional outputs and inplace cache updates.

The main entry points are:

- `test.py`: pytest runner for positive, fuzz, and negative runtime cases.
- `gen_st_cases.py`: coverage-driven testcase generator.
- `prologv3_generalized.py`: CPU reference path and NPU invocation wrapper.
- `check_valid_param.py`: parameter validation and tensor comparison policy.

For the detailed CPU dataflow, see [CPU_REFERENCE_DESIGN.md](CPU_REFERENCE_DESIGN.md).
For generator-specific CLI details, see [GENERATOR_USAGE.md](GENERATOR_USAGE.md).

## Quick Start

All commands below assume the current working directory is:

```bash
cd attention/mla_prolog_v3/tests/pytest
```

Recommended workflow:

1. Source the required CANN and `torch_npu` environment.
2. Regenerate cases if you changed factor coverage, legality rules, or hardware profile.
3. Run the stable suite (`ci` + `negative`).
4. Run fuzz only when you explicitly want extra randomized coverage.

Typical commands:

```bash
python3 gen_st_cases.py
python3 -m pytest -rA -s test.py -m "ci or negative"
MLA_PROLOG_V3_ENABLE_FUZZ=1 python3 -m pytest -rA -s test.py -m fuzz
```

## Folder Layout

| File | Purpose |
| --- | --- |
| `README.md` | Entry document for the pytest project. |
| `GENERATOR_USAGE.md` | Detailed usage guide for `gen_st_cases.py`. |
| `CPU_REFERENCE_DESIGN.md` | Detailed design note for the CPU reference implementation. |
| `gen_st_cases.py` | Generates `testcases.py` and `st_case_coverage_report.md` from a factor pool. |
| `testcases.py` | Auto-generated testcase module imported by `test.py`. Do not hand-edit. |
| `st_case_coverage_report.md` | Auto-generated coverage report for the currently selected positive set. |
| `test.py` | Collects and runs positive (`ci`), fuzz, and negative runtime tests. |
| `check_valid_param.py` | Validates legal inputs and compares CPU/NPU outputs with configurable error policy. |
| `prologv3_generalized.py` | Builds randomized tensors, executes the CPU reference, then calls the NPU op. |
| `pytest.ini` | Registers pytest markers used by this folder. |

## How the Pieces Fit Together

### 1. Case generation

`gen_st_cases.py` starts from a default factor pool, removes illegal combinations, tags each
remaining case with coverage labels, and selects a deterministic minimal positive set by set cover.

It writes two generated artifacts:

- `testcases.py`: Python module consumed by `test.py`.
- `st_case_coverage_report.md`: human-readable summary of selected cases and covered tags.

### 2. Test collection and execution

`test.py` imports three generated constants from `testcases.py`:

- `ENABLED_PARAMS`: positive coverage cases used by `@pytest.mark.ci`.
- `FUZZ_PARAM_SPACE`: random sampling pool used by `@pytest.mark.fuzz`.
- `NEGATIVE_RUNTIME_CASES`: invalid inputs that should fail at runtime.

Positive cases are expanded by `_build_param_combinations()`, deduplicated, and filtered again by
`validate_quant_cache_combo()` before execution.

### 3. Validation and compare

`check_valid_param.py` has two roles:

- `validate_config(...)` catches front-door parameter errors for positive and fuzz cases.
- `check_result(...)` compares CPU and NPU outputs, including inplace-updated cache tensors.

Negative runtime tests intentionally bypass `validate_config(...)` so the failure comes from the
actual runtime path instead of the pre-check layer.

### 4. CPU reference and NPU call

`prologv3_generalized.py` provides:

- `GeneralizedPrologV3`: CPU reference implementation.
- `test_prologv3_generalized(...)`: random input builder + CPU/NPU runner.
- `validate_quant_cache_combo(...)`: shared legality filter for quantization/cache combinations.

This function currently fixes the NPU device with `torch_npu.npu.set_device(0)`, so the test
environment must have a usable Ascend device `0`.

## Generated Artifacts

`testcases.py` is generated code. It exports:

- `TEST_PARAMS`: named positive cases.
- `FULL_COVERAGE_PARAMS`: ordered list of selected positive cases.
- `ENABLED_PARAMS`: currently aliased to `FULL_COVERAGE_PARAMS`.
- `FUZZ_PARAM_SPACE`: factor pool used by random fuzz sampling.
- `NEGATIVE_RUNTIME_CASES`: invalid cases plus expected error substrings.

`st_case_coverage_report.md` is the audit trail for the current generation. Use it to answer:

- how many candidates were enumerated,
- how many positive cases were selected,
- which coverage tags are reachable,
- which tags each selected case covers.

## Common Pytest Commands

Run the stable suite:

```bash
python3 -m pytest -rA -s test.py -m "ci or negative"
```

Run only the selected positive cases:

```bash
python3 -m pytest -rA -s test.py -m ci
```

Run only negative runtime failures:

```bash
python3 -m pytest -rA -s test.py -m negative
```

Run fuzz cases:

```bash
MLA_PROLOG_V3_ENABLE_FUZZ=1 python3 -m pytest -rA -s test.py -m fuzz
```

Run everything in the module:

```bash
python3 -m pytest -rA -s test.py
```

Notes:

- The fuzz test is collected by default but skips unless `MLA_PROLOG_V3_ENABLE_FUZZ=1`.
- `pytest.ini` also registers a `graph` marker, but `test.py` does not currently define any
  `@pytest.mark.graph` tests.
- Some quantized paths are skipped intentionally when the local runtime does not provide the
  required float8 or hif8 dtype support.

## Useful Environment Variables

| Variable | Default | Effect |
| --- | --- | --- |
| `MLA_PROLOG_V3_ENABLE_FUZZ` | `0` | Enables the fuzz test body when set to `1`. |
| `MLA_PROLOG_V3_FUZZ_CASES` | `20` | Number of random fuzz cases to generate. |
| `MLA_PROLOG_V3_FUZZ_SEED` | `3` | Seed for deterministic fuzz case sampling. |
| `MLA_PROLOG_V3_CPU_INFO_LOG` | `0` | Enables info logs in the CPU reference path. |
| `MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR` | `0` | Allows sparse mismatches during compare when set to `1`. |
| `MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_RATIO` | `0.001` | Maximum mismatch ratio in discontinuous mode. |
| `MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_COUNT` | `0` | Maximum mismatch count in discontinuous mode; `0` means ratio-only. |

Example:

```bash
MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR=1 \
MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_RATIO=0.001 \
MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_COUNT=0 \
python3 -m pytest -rA -s test.py -m ci
```

## Current Test Model Assumptions

The pytest project intentionally models a constrained runtime slice. Important current assumptions:

- `S2` follows `S1`, so the KV sequence length is modeled as `q_seq`.
- Positive generation fixes `kv_head_num=1`, `Hcq=1536`, `Hckv=512`, `head_dim=128`,
  `rope_head_dim=64`, `input_layout="BSH"`, and `dtype=torch.bfloat16`.
- `tile_size` is fixed to `128` in the generated positive and fuzz pools.
- Quant/cache legality is guarded both at generation time and again at runtime by
  `validate_quant_cache_combo(...)`.

If you need to extend the legal space, update the relevant implementation and then regenerate
`testcases.py` and `st_case_coverage_report.md`.

## Maintenance Workflow

When changing this test project, use this order:

1. Update generation logic, legality rules, or runtime/reference code.
2. Regenerate with `python3 gen_st_cases.py`.
3. Review `st_case_coverage_report.md` for unexpected tag loss or case growth.
4. Run `ci` and `negative`.
5. Run fuzz if the change affects case enumeration, validation, or compare logic.
