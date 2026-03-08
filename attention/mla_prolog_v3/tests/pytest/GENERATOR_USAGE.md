# `gen_st_cases.py` Usage Guide

`gen_st_cases.py` generates the checked-in testcase module for this pytest project.
It is the source of truth for:

- the selected positive coverage set,
- the fuzz factor pool,
- the negative runtime cases,
- the coverage report used to review reachability.

The script does not run pytest by itself. Its job is to produce deterministic inputs for `test.py`.

## What the Generator Produces

By default, the script writes:

- `testcases.py`
- `st_case_coverage_report.md`

Both paths are relative to the script directory unless you override them on the command line.

## CLI

```bash
python3 gen_st_cases.py [options]
```

Available options:

| Option | Meaning |
| --- | --- |
| `--factor-space-json PATH` | Override part of the default factor pool with a JSON file. |
| `--aic-num N` | Hardware AIC core count used when computing coverage tags. Default: `24`. |
| `--aiv-num N` | Hardware AIV core count used when computing coverage tags. Default: `48`. |
| `--output-testcases PATH` | Custom output path for generated `testcases.py`. |
| `--output-report PATH` | Custom output path for generated coverage report. |
| `--focus-factors a,b,c` | Generate a feature-relative case set that only varies the listed model factors. |

Supported `--focus-factors` names:

```text
batch_size,He,q_head_num,q_seq,block_size,cache_mode,bs_fused_flag,
weight_quant_mode,kv_quant_mode,smooth_scales_cq_flag,query_norm_flag
```

## Recommended Commands

Regenerate the default checked-in artifacts:

```bash
python3 gen_st_cases.py
```

Generate using a different hardware profile:

```bash
python3 gen_st_cases.py --aic-num 20 --aiv-num 40
```

Generate a smaller feature-relative set for cache/quant behavior:

```bash
python3 gen_st_cases.py \
  --focus-factors cache_mode,weight_quant_mode,kv_quant_mode
```

Write outputs somewhere else for review before replacing checked-in files:

```bash
python3 gen_st_cases.py \
  --output-testcases /tmp/testcases.py \
  --output-report /tmp/st_case_coverage_report.md
```

## Custom Factor Space JSON

`--factor-space-json` merges a JSON object into the script's `DEFAULT_FACTOR_SPACE`.
Only known factor names are applied; unknown keys are ignored.

Example:

```json
{
  "batch_size": [1, 4, 8],
  "He": [1024, 7168],
  "cache_mode": ["PA_BSND", "BSND"],
  "weight_quant_mode": [0, 2],
  "kv_quant_mode": [0, 3]
}
```

Run with:

```bash
python3 gen_st_cases.py --factor-space-json custom_factor_space.json
```

## Generation Pipeline

The generator follows this flow:

1. Load the default factor pool and merge any JSON overrides.
2. Build candidate positive cases from the model factor space.
3. Derive dependent fields such as `query_quant_mode`, `ckvkr_repo_mode`, and `quant_scale_repo_mode`.
4. Drop illegal cases with `validate_positive_case(...)`.
5. Assign coverage tags with `build_case_tags(...)`.
6. Select a deterministic minimal positive set with `deterministic_set_cover(...)`.
7. Merge the fuzz factor pool and append the negative runtime cases.
8. Render `testcases.py` and `st_case_coverage_report.md`.

Two generation modes exist:

- Default mode: explores all configured model factors.
- Feature-relative mode: keeps a baseline case and varies only the factors listed in `--focus-factors`.

## Generated `testcases.py` Contract

`test.py` expects the generated module to export the following names:

- `TEST_PARAMS`
- `FULL_COVERAGE_PARAMS`
- `ENABLED_PARAMS`
- `FUZZ_PARAM_SPACE`
- `NEGATIVE_RUNTIME_CASES`

If you change the generator output format, update `test.py` at the same time.

## When You Should Regenerate

Rerun the generator when any of the following changes:

- `DEFAULT_FACTOR_SPACE`
- coverage tagging logic in `build_case_tags(...)`
- legality checks in `validate_positive_case(...)`
- quant/cache compatibility rules used by the pytest project
- the intended hardware profile (`aic_num` / `aiv_num`)
- the negative runtime case list

After regeneration, review `st_case_coverage_report.md` before committing.

## Reading the Coverage Report

The coverage report is designed for maintenance reviews. The most useful sections are:

- header summary: candidate count, selected positive count, reachable tag count
- tree map: coverage categories and currently unreachable branches
- selected cases: exact parameter sets and the tags each one covers
- uncovered tags: should remain empty for the reachable universe

If the selected case count suddenly spikes or reachable tags disappear, inspect the factor pool,
the hardware profile, or the legality rules before updating the checked-in artifacts.
