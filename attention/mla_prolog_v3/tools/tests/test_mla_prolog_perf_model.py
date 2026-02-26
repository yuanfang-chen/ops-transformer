import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = ROOT / "attention/mla_prolog_v3/tools/mla_prolog_perf_model.py"

spec = importlib.util.spec_from_file_location("mla_prolog_perf_model", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)  # type: ignore[attr-defined]


class MlaPrologPerfModelTests(unittest.TestCase):
    def test_exact_matmul_flops(self):
        p = mod.ProblemSpec(T=2, He=4, Hcq=8, N=2, D=4, Dr=2, Hckv=6, Nkv=1, Dtile=6)
        p.validate()
        flops = mod.matmul_flops_exact(p)
        self.assertEqual(flops["MatmulCq"], 2 * 2 * 4 * 8)
        self.assertEqual(flops["MatmulCkvKr"], 2 * 2 * 4 * (6 + 2))
        self.assertEqual(flops["MatmulQcQr"], 2 * 2 * 8 * (2 * (4 + 2)))
        self.assertEqual(flops["MatmulQn"], 2 * 2 * 2 * 4 * 6)

    def test_tiling_derivation_no_quant(self):
        p = mod.ProblemSpec(T=8, He=7168, Hcq=1536, N=32, D=128, Dr=64, Hckv=512, Nkv=1, Dtile=512)
        p.validate()
        q = mod.quant_plan(0)
        hw = mod.HardwareProfile()
        tiling, base = mod.derive_tiling(p, hw, q)
        self.assertEqual(tiling.splitMFlag, 0)
        self.assertEqual(tiling.stepBatchSize, 8)
        self.assertEqual(tiling.mm2BlockNum, 9)
        self.assertEqual(base.headSizeQc, 4096)
        self.assertEqual(base.headSizeQr, 2048)

    def test_cli_json_runs(self):
        payload = {
            "T": 16,
            "He": 7168,
            "Hcq": 1536,
            "N": 32,
            "D": 128,
            "Dr": 64,
            "Hckv": 512,
            "Nkv": 1,
            "Dtile": 512,
            "quant_mode": 0,
            "cache_mode": "PA_BSND"
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--input-mode", "shape", "--input-json", str(p), "--report", "json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            out = json.loads(proc.stdout)
            self.assertIn("matmul_flops_exact", out)
            self.assertIn("timing_cold", out)
            self.assertIn("timing_warm", out)


if __name__ == "__main__":
    unittest.main()
