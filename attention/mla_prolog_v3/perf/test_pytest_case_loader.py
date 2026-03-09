from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _perf_model import ACTUAL_SEQ_EN_Q_LEN
from _perf_model import case_from_mapping
from _perf_model import case_to_mapping
from _perf_model import load_case_config
from _pytest_case_loader import DEFAULT_PYTEST_TESTCASES
from _pytest_case_loader import load_pytest_case_catalog
from _pytest_case_loader import load_pytest_case_set


ANALYZE_SCRIPT = THIS_DIR / "analyze_mla_prolog_v3_theory.py"
SEARCH_SCRIPT = THIS_DIR / "search_mla_prolog_v3_tiling.py"
EXAMPLE_CASE = THIS_DIR / "example_case.json"
HW_CONFIG = THIS_DIR / "ascend950_hw.json"


class PytestCaseLoaderTest(unittest.TestCase):
    def test_loader_resolves_test_params_and_enabled_alias(self) -> None:
        catalog = load_pytest_case_catalog(DEFAULT_PYTEST_TESTCASES)
        enabled = load_pytest_case_set(DEFAULT_PYTEST_TESTCASES, case_set="enabled")
        all_cases = load_pytest_case_set(DEFAULT_PYTEST_TESTCASES, case_set="all")
        self.assertIn("coverage_case_001", catalog)
        self.assertEqual(len(enabled), 10)
        self.assertEqual(len(all_cases), len(catalog))
        self.assertEqual(enabled[1].name, "coverage_case_001")

    def test_named_case_normalization_matches_expectation(self) -> None:
        record = load_pytest_case_set(DEFAULT_PYTEST_TESTCASES, case_set="enabled", case_name="coverage_case_001")[0]
        case = load_case_config(EXAMPLE_CASE)
        self.assertEqual(record.perf_mapping["T"], 144)
        self.assertEqual(record.perf_mapping["He"], 7680)
        self.assertEqual(record.perf_mapping["q_head_num"], 128)
        self.assertEqual(record.perf_mapping["cache_mode"], "PA_BLK_BSND")
        self.assertEqual(record.perf_mapping["actual_seq_mode"], ACTUAL_SEQ_EN_Q_LEN)
        self.assertTrue(record.perf_mapping["smooth_scales_enabled"])
        self.assertEqual(case, case_from_mapping(record.perf_mapping))

    def test_example_case_equals_pytest_case(self) -> None:
        record = load_pytest_case_set(DEFAULT_PYTEST_TESTCASES, case_set="enabled", case_name="coverage_case_001")[0]
        example_case = load_case_config(EXAMPLE_CASE)
        self.assertEqual(example_case, case_from_mapping(record.perf_mapping))
        self.assertEqual(
            case_to_mapping(example_case, case_name="coverage_case_001"),
            case_to_mapping(case_from_mapping(record.perf_mapping), case_name=record.name),
        )

    def test_analyze_cli_rejects_mixed_case_sources(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(ANALYZE_SCRIPT),
                "--hw-config",
                str(HW_CONFIG),
                "--case-config",
                str(EXAMPLE_CASE),
                "--pytest-case-name",
                "coverage_case_001",
            ],
            capture_output=True,
            text=True,
            cwd=THIS_DIR.parent.parent.parent,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--case-config cannot be combined with pytest case flags", proc.stderr)

    def test_search_cli_rejects_mixed_case_sources(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SEARCH_SCRIPT),
                "--hw-config",
                str(HW_CONFIG),
                "--case-config",
                str(EXAMPLE_CASE),
                "--pytest-case-name",
                "coverage_case_001",
            ],
            capture_output=True,
            text=True,
            cwd=THIS_DIR.parent.parent.parent,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--case-config cannot be combined with pytest case flags", proc.stderr)

    def test_analyze_cli_batch_mode_over_enabled_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "analysis.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZE_SCRIPT),
                    "--hw-config",
                    str(HW_CONFIG),
                    "--pytest-case-set",
                    "enabled",
                    "--emit-json",
                    str(output_json),
                ],
                capture_output=True,
                text=True,
                cwd=THIS_DIR.parent.parent.parent,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 10)
            self.assertEqual(payload[0]["case_name"], "coverage_case_000")
            self.assertEqual(payload[1]["case_name"], "coverage_case_001")

    def test_search_cli_accepts_pytest_named_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "search.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SEARCH_SCRIPT),
                    "--hw-config",
                    str(HW_CONFIG),
                    "--pytest-case-name",
                    "coverage_case_001",
                    "--mode",
                    "host_legal",
                    "--emit-json",
                    str(output_json),
                ],
                capture_output=True,
                text=True,
                cwd=THIS_DIR.parent.parent.parent,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["case_name"], "coverage_case_001")


if __name__ == "__main__":
    unittest.main()
