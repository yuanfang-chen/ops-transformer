from __future__ import annotations

import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _perf_model import KV_QUANT_PER_TENSOR
from _perf_model import WEIGHT_QUANT_FULL
from _perf_model import QUERY_QUANT_PER_TOKEN_HEAD
from _perf_model import analyze_case
from _perf_model import apply_case_updates
from _perf_model import best_candidates
from _perf_model import current_host_tiling
from _perf_model import derive_path_info
from _perf_model import load_case_config
from _perf_model import load_hardware_config


DEFAULT_HW = THIS_DIR / "ascend950_hw.json"
DEFAULT_CASE = THIS_DIR / "example_case.json"


class PerfModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hw = load_hardware_config(DEFAULT_HW)
        cls.base_case = load_case_config(DEFAULT_CASE)

    def test_hw_peak_derivation(self) -> None:
        self.assertAlmostEqual(self.hw.cube_bf16_tflops_total, 432.5376, places=4)
        self.assertAlmostEqual(self.hw.cube_int8_tops_total, 865.0752, places=4)
        self.assertAlmostEqual(self.hw.vector_fp32_tflops_total, 432.5376, places=4)
        self.assertAlmostEqual(self.hw.vector_bf16_tflops_total, 865.0752, places=4)

    def test_current_host_tiling_matches_cpp_rules(self) -> None:
        tiling = current_host_tiling(self.base_case, self.hw)
        self.assertEqual(tiling.step_batch_size, 128)
        self.assertEqual(tiling.vector_block_num, 64)
        self.assertEqual(tiling.step_num_head_dequant, 64)
        self.assertEqual(tiling.mm1_single_core_n, 64)
        self.assertEqual(tiling.mm2_single_core_n, 64)
        self.assertEqual(tiling.mm3_single_core_n, 768)
        self.assertEqual(tiling.mm4_single_core_batch, 4)
        self.assertEqual(tiling.mm1_block_num, 24)
        self.assertEqual(tiling.mm2_block_num, 9)
        self.assertEqual(tiling.mm3_block_num, 32)
        self.assertEqual(tiling.mm4_block_num, 32)
        self.assertEqual(tiling.vector_core_num_runtime, 128)
        self.assertEqual(tiling.cur_vec_token_max, 1)

    def test_quant_path_activates_vector_postprocess(self) -> None:
        quant_case = apply_case_updates(
            self.base_case,
            {
                "weight_quant_mode": WEIGHT_QUANT_FULL,
                "kv_quant_mode": KV_QUANT_PER_TENSOR,
                "query_quant_mode": QUERY_QUANT_PER_TOKEN_HEAD,
            },
        )
        path = derive_path_info(quant_case, self.hw)
        self.assertTrue(path.enable_dequant_opt)
        self.assertTrue(path.need_dequant_or_cast_qc)
        self.assertTrue(path.need_dynamic_quant_qn_mul_qr)

        analysis = analyze_case(quant_case, self.hw)
        stages = {stage.name: stage for stage in analysis.steady_step_stages}
        self.assertGreater(stages["dequant_or_cast_qc"].duration_us, 0.0)
        self.assertGreater(stages["dynamic_quant_qn_mul_qr"].duration_us, 0.0)

    def test_bf16_large_prefers_cube_side(self) -> None:
        analysis = analyze_case(self.base_case, self.hw)
        self.assertEqual(analysis.critical_path_type, "cube_side")
        self.assertGreater(analysis.case.T, analysis.tiling.step_batch_size)
        self.assertEqual(analysis.tail_tokens, 16)

    def test_host_legal_search_reproduces_baseline(self) -> None:
        candidates = best_candidates(self.base_case, self.hw, mode="host_legal", top_k=3)
        self.assertEqual(len(candidates), 1)
        baseline = current_host_tiling(self.base_case, self.hw)
        self.assertEqual(candidates[0].tiling.step_batch_size, baseline.step_batch_size)
        self.assertEqual(candidates[0].tiling.mm3_base_n, baseline.mm3_base_n)

    def test_exploratory_search_runs(self) -> None:
        candidates = best_candidates(self.base_case, self.hw, mode="exploratory", top_k=3)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreater(candidates[0].search_objective_us, 0.0)


if __name__ == "__main__":
    unittest.main()
