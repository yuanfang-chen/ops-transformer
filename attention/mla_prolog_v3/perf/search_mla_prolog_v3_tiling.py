from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _perf_model import SearchCandidate
from _perf_model import best_candidates
from _perf_model import current_host_tiling
from _perf_model import dump_json
from _perf_model import load_case_config
from _perf_model import load_hardware_config


DEFAULT_HW = THIS_DIR / "ascend950_hw.json"
DEFAULT_CASE = THIS_DIR / "example_case.json"


def _format_candidate(candidate: SearchCandidate, baseline_us: float) -> str:
    speedup = baseline_us / candidate.search_objective_us if candidate.search_objective_us > 0 else 1.0
    lines = [
        f"- objective={candidate.search_objective_us:.3f} us, speedup={speedup:.3f}x, critical={candidate.analysis.critical_path_type}",
        "  tiling: "
        f"stepBatchSize={candidate.tiling.step_batch_size}, vectorBlockNum={candidate.tiling.vector_block_num}, "
        f"stepNumHeadDequant={candidate.tiling.step_num_head_dequant}, "
        f"mm1/mm2/mm3/mm4={candidate.tiling.mm1_single_core_n}/{candidate.tiling.mm2_single_core_n}/"
        f"{candidate.tiling.mm3_single_core_n}/{candidate.tiling.mm4_single_core_batch}, "
        f"mm3.baseN={candidate.tiling.mm3_base_n}",
        "  buffer_fit: "
        f"UB={candidate.analysis.buffer_fit.ub_bytes_used}/{candidate.analysis.buffer_fit.ub_bytes_available}B, "
        f"L1A={candidate.analysis.buffer_fit.l1_a_bytes_required}/{candidate.analysis.buffer_fit.l1_a_bytes_available}B, "
        f"L1B={candidate.analysis.buffer_fit.l1_b_bytes_required}/{candidate.analysis.buffer_fit.l1_b_bytes_available}B",
    ]
    if candidate.requires_host_changes:
        lines.append(f"  requires changes: {', '.join(candidate.requires_host_changes)}")
    else:
        lines.append("  requires changes: none")
    return "\n".join(lines)


def _print_summary(mode: str, candidates: Sequence[SearchCandidate]) -> None:
    if not candidates:
        print("No legal tiling candidate found.")
        return
    baseline = candidates[0].analysis if mode == "host_legal" else None
    if baseline is None:
        raise RuntimeError("exploratory mode requires separate baseline print")
    print(f"Host-legal baseline steady-step: {baseline.steady_step_time_us:.3f} us")
    print(_format_candidate(candidates[0], baseline.steady_step_time_us))


def main() -> int:
    parser = argparse.ArgumentParser(description="Search best tiling strategy for mla_prolog_v3 theory model.")
    parser.add_argument("--hw-config", type=Path, default=DEFAULT_HW)
    parser.add_argument("--case-config", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--mode", choices=("host_legal", "exploratory"), default="host_legal")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--objective", choices=("steady_step", "total_kernel"), default="steady_step")
    parser.add_argument("--emit-json", type=Path)
    args = parser.parse_args()

    hw = load_hardware_config(args.hw_config)
    case = load_case_config(args.case_config)
    baseline_tiling = current_host_tiling(case, hw)
    baseline_list = best_candidates(case, hw, mode="host_legal", top_k=1, objective=args.objective)
    baseline = baseline_list[0]
    print(f"Baseline host tiling on {hw.soc_name}")
    print(
        f"stepBatchSize={baseline_tiling.step_batch_size}, vectorBlockNum={baseline_tiling.vector_block_num}, "
        f"stepNumHeadDequant={baseline_tiling.step_num_head_dequant}, "
        f"mm1/mm2/mm3/mm4={baseline_tiling.mm1_single_core_n}/{baseline_tiling.mm2_single_core_n}/"
        f"{baseline_tiling.mm3_single_core_n}/{baseline_tiling.mm4_single_core_batch}, mm3.baseN={baseline_tiling.mm3_base_n}"
    )
    print(f"Baseline objective={baseline.search_objective_us:.3f} us, critical={baseline.analysis.critical_path_type}")

    if args.mode == "host_legal":
        print()
        _print_summary(args.mode, baseline_list)
        payload = [candidate.to_dict() for candidate in baseline_list]
    else:
        candidates = best_candidates(case, hw, mode="exploratory", top_k=args.top_k, objective=args.objective)
        print()
        print(f"Top {len(candidates)} exploratory candidates")
        for candidate in candidates:
            print(_format_candidate(candidate, baseline.search_objective_us))
        if not candidates:
            print("No better legal candidate found.")
        payload = [candidate.to_dict() for candidate in candidates]

    if args.emit_json:
        dump_json(
            args.emit_json,
            {
                "soc_name": hw.soc_name,
                "mode": args.mode,
                "objective": args.objective,
                "baseline": baseline.to_dict(),
                "candidates": payload,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
