from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List


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
from _perf_model import dump_json
from _perf_model import load_case_config
from _perf_model import load_hardware_config
from _perf_model import parse_sweep_items
from _perf_model import render_analysis_text
from _perf_model import render_report
from _perf_model import sweep_cases


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


def _build_report_analyses(hw_path: Path, case_path: Path) -> tuple[Dict[str, AnalysisResult], Dict[str, object]]:
    hw = load_hardware_config(hw_path)
    base_case = load_case_config(case_path)
    analyses = {
        "bf16_small": analyze_case(apply_case_updates(base_case, {"B": 1, "S": 1, "T": 1}), hw),
        "bf16_large": analyze_case(apply_case_updates(base_case, {"B": 9, "S": 16, "T": 144}), hw),
        "int8_tensor": analyze_case(
            apply_case_updates(
                base_case,
                {
                    "B": 9,
                    "S": 16,
                    "T": 144,
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
                    "B": 9,
                    "S": 16,
                    "T": 144,
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
                    "B": 9,
                    "S": 16,
                    "T": 144,
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


def _print_analyses(results: Iterable[AnalysisResult]) -> None:
    first = True
    for result in results:
        if not first:
            print()
            print("=" * 80)
            print()
        print(render_analysis_text(result))
        first = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze theoretical duration of mla_prolog_v3 kernel stages.")
    parser.add_argument("--hw-config", type=Path, default=DEFAULT_HW)
    parser.add_argument("--case-config", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--emit-json", type=Path)
    parser.add_argument("--emit-csv", type=Path)
    parser.add_argument("--emit-report", type=Path)
    parser.add_argument("--sweep", action="append", default=[], help="Repeatable key=v1,v2 sweep, e.g. --sweep T=1,144")
    args = parser.parse_args()

    hw = load_hardware_config(args.hw_config)
    base_case = load_case_config(args.case_config)
    sweep = parse_sweep_items(args.sweep)
    cases = sweep_cases(base_case, sweep)
    results = [analyze_case(case, hw) for case in cases]

    _print_analyses(results)

    if args.emit_json:
        dump_json(args.emit_json, [result.to_dict() for result in results])

    if args.emit_csv:
        rows: List[Dict[str, object]] = []
        for result in results:
            for row in analysis_rows_for_csv(result, steady=False) + analysis_rows_for_csv(result, steady=True):
                row["soc_name"] = hw.soc_name
                rows.append(row)
        _write_csv(args.emit_csv, rows)

    report_path = args.emit_report
    if report_path:
        analyses, report_payload = _build_report_analyses(args.hw_config, args.case_config)
        report_path.write_text(str(report_payload["report_text"]), encoding="utf-8")
        print()
        print(f"Report written to {report_path}")
        print(f"Report analyses: {', '.join(sorted(analyses))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
