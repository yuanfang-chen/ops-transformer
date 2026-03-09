from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _perf_model import ACTUAL_SEQ_EN_Q_LEN
from _perf_model import AnalysisResult
from _perf_model import QUERY_QUANT_PER_TOKEN_HEAD
from _perf_model import KV_QUANT_PER_TENSOR
from _perf_model import KV_QUANT_PER_TILE
from _perf_model import WEIGHT_QUANT_FULL
from _perf_model import WEIGHT_QUANT_MXFP8
from _perf_model import analyze_case
from _perf_model import analysis_rows_for_csv
from _perf_model import apply_case_updates
from _perf_model import best_candidates
from _perf_model import case_from_mapping
from _perf_model import case_to_mapping
from _perf_model import dump_json
from _perf_model import load_json
from _perf_model import load_case_config
from _perf_model import load_hardware_config
from _perf_model import parse_sweep_items
from _perf_model import render_analysis_text
from _perf_model import render_report
from _perf_model import sweep_cases
from _pytest_case_loader import DEFAULT_PYTEST_TESTCASES
from _pytest_case_loader import load_pytest_case_set


DEFAULT_HW = THIS_DIR / "ascend950_hw.json"
DEFAULT_CASE = THIS_DIR / "example_case.json"


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_input_cases(args: argparse.Namespace) -> List[Dict[str, object]]:
    using_pytest = any(
        value is not None
        for value in (args.pytest_testcases, args.pytest_case_set, args.pytest_case_name)
    )
    if args.case_config is not None and using_pytest:
        raise ValueError("--case-config cannot be combined with pytest case flags")
    if using_pytest:
        pytest_path = args.pytest_testcases or DEFAULT_PYTEST_TESTCASES
        case_set = args.pytest_case_set or "enabled"
        records = load_pytest_case_set(pytest_path, case_set=case_set, case_name=args.pytest_case_name)
        return [
            {
                "case_name": record.name,
                "case_source": f"pytest:{record.source_set}",
                "normalized_case": record.perf_mapping,
                "case": case_from_mapping(record.perf_mapping),
            }
            for record in records
        ]
    case_path = args.case_config or DEFAULT_CASE
    raw_mapping = load_json(case_path)
    case_name = str(raw_mapping.get("case_name", Path(case_path).stem))
    case = load_case_config(case_path)
    return [
        {
            "case_name": case_name,
            "case_source": f"json:{Path(case_path).name}",
            "normalized_case": case_to_mapping(case, case_name=case_name),
            "case": case,
        }
    ]


def _build_report_analyses(hw_path: Path, case_path: Path) -> tuple[Dict[str, AnalysisResult], Dict[str, object]]:
    hw = load_hardware_config(hw_path)
    base_case = apply_case_updates(
        load_case_config(case_path),
        {
            "B": 9,
            "S": 16,
            "T": 144,
            "He": 7168,
            "Hcq": 1536,
            "Hckv": 512,
            "N": 128,
            "Nkv": 1,
            "D": 128,
            "Dr": 64,
            "block_size": 128,
            "cache_mode": "BSND",
            "actual_seq_mode": "DISABLED",
            "weight_quant_mode": 0,
            "kv_quant_mode": 0,
            "query_quant_mode": 0,
            "ckvkr_repo_mode": 0,
            "quant_scale_repo_mode": 0,
            "tile_size": 128,
            "query_norm_flag": 0,
            "qc_qr_scale": 1.0,
            "kc_scale": 1.0,
            "smooth_scales_enabled": False,
        },
    )
    analyses = {
        "bf16_small": analyze_case(apply_case_updates(base_case, {"B": 1, "S": 1, "T": 1}), hw),
        "bf16_large": analyze_case(base_case, hw),
        "int8_tensor": analyze_case(
            apply_case_updates(
                base_case,
                {
                    "weight_quant_mode": WEIGHT_QUANT_FULL,
                    "kv_quant_mode": KV_QUANT_PER_TENSOR,
                    "query_quant_mode": QUERY_QUANT_PER_TOKEN_HEAD,
                },
            ),
            hw,
        ),
        "fp8_tensor": analyze_case(
            apply_case_updates(
                base_case,
                {
                    "weight_quant_mode": WEIGHT_QUANT_MXFP8,
                    "kv_quant_mode": KV_QUANT_PER_TENSOR,
                    "query_quant_mode": QUERY_QUANT_PER_TOKEN_HEAD,
                },
            ),
            hw,
        ),
        "pa_blk_stress": analyze_case(
            apply_case_updates(
                base_case,
                {
                    "cache_mode": "PA_BLK_BSND",
                    "actual_seq_mode": ACTUAL_SEQ_EN_Q_LEN,
                    "weight_quant_mode": WEIGHT_QUANT_FULL,
                    "kv_quant_mode": KV_QUANT_PER_TILE,
                    "query_quant_mode": 0,
                    "quant_scale_repo_mode": 1,
                },
            ),
            hw,
        ),
    }
    rendered_searches = {
        "bf16_large_host_legal": best_candidates(analyses["bf16_large"].case, hw, mode="host_legal", top_k=1),
        "bf16_large_exploratory": best_candidates(analyses["bf16_large"].case, hw, mode="exploratory", top_k=3),
        "int8_tensor_exploratory": best_candidates(analyses["int8_tensor"].case, hw, mode="exploratory", top_k=3),
    }
    report_text = render_report(hw, analyses, rendered_searches)
    return analyses, {
        "report_text": report_text,
        "searches": {key: [item.to_dict() for item in value] for key, value in rendered_searches.items()},
        "soc_name": hw.soc_name,
    }


def _print_analyses(results: Iterable[Tuple[Dict[str, object], AnalysisResult]]) -> None:
    first = True
    for entry, result in results:
        if not first:
            print()
            print("=" * 80)
            print()
        print(f"Case {entry['case_name']} [{entry['case_source']}]")
        print()
        print(render_analysis_text(result))
        first = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze theoretical duration of mla_prolog_v3 kernel stages.")
    parser.add_argument("--hw-config", type=Path, default=DEFAULT_HW)
    parser.add_argument("--case-config", type=Path)
    parser.add_argument("--pytest-testcases", type=Path)
    parser.add_argument("--pytest-case-set", choices=("enabled", "all"))
    parser.add_argument("--pytest-case-name")
    parser.add_argument("--emit-json", type=Path)
    parser.add_argument("--emit-csv", type=Path)
    parser.add_argument("--emit-report", type=Path)
    parser.add_argument("--emit-case-json", type=Path)
    parser.add_argument("--sweep", action="append", default=[], help="Repeatable key=v1,v2 sweep, e.g. --sweep T=1,144")
    args = parser.parse_args()

    hw = load_hardware_config(args.hw_config)
    try:
        selected_entries = _resolve_input_cases(args)
    except ValueError as exc:
        parser.error(str(exc))

    sweep = parse_sweep_items(args.sweep)
    if sweep and len(selected_entries) > 1:
        parser.error("--sweep requires a single selected input case")
    result_entries: List[Tuple[Dict[str, object], AnalysisResult]] = []
    for entry in selected_entries:
        base_case = entry["case"]
        cases = sweep_cases(base_case, sweep) if sweep else [base_case]
        for idx, case in enumerate(cases):
            current_entry = dict(entry)
            if sweep:
                current_entry["case_name"] = f"{entry['case_name']}#sweep_{idx:03d}"
                current_entry["normalized_case"] = case_to_mapping(case, case_name=str(current_entry["case_name"]))
            result_entries.append((current_entry, analyze_case(case, hw)))

    _print_analyses(result_entries)

    if args.emit_json:
        dump_json(
            args.emit_json,
            [
                {
                    "case_name": entry["case_name"],
                    "case_source": entry["case_source"],
                    "normalized_case": entry["normalized_case"],
                    "analysis": result.to_dict(),
                }
                for entry, result in result_entries
            ],
        )

    if args.emit_csv:
        rows: List[Dict[str, object]] = []
        for entry, result in result_entries:
            for row in analysis_rows_for_csv(result, steady=False) + analysis_rows_for_csv(result, steady=True):
                row["soc_name"] = hw.soc_name
                row["case_name"] = entry["case_name"]
                row["case_source"] = entry["case_source"]
                rows.append(row)
        _write_csv(args.emit_csv, rows)

    if args.emit_case_json:
        if len(selected_entries) != 1 or sweep:
            parser.error("--emit-case-json requires exactly one selected base case with no sweep")
        selected_entry = selected_entries[0]
        dump_json(
            args.emit_case_json,
            case_to_mapping(selected_entry["case"], case_name=str(selected_entry["case_name"])),
        )

    report_path = args.emit_report
    if report_path:
        analyses, report_payload = _build_report_analyses(args.hw_config, args.case_config or DEFAULT_CASE)
        report_path.write_text(str(report_payload["report_text"]), encoding="utf-8")
        print()
        print(f"Report written to {report_path}")
        print(f"Report analyses: {', '.join(sorted(analyses))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
