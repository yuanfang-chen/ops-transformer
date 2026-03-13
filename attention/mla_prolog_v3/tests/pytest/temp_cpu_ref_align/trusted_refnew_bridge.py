import re
import sys
import types
from pathlib import Path


PYTEST_DIR = Path(__file__).resolve().parents[1]
if str(PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTEST_DIR))


_REFNEW_PATH = PYTEST_DIR / "aclnnMlaPrologV3RefNew.py"
_TRUSTED_MODULE = None


def _sanitize_refnew_source(text: str) -> str:
    lines = text.splitlines()
    while lines and not re.match(r"^(from|import|def|class|try:|#)", lines[0]):
        lines.pop(0)
    text = "\n".join(lines)

    replacements = [
        (
            r"(?m)^import libs\.tools as tools$",
            "try:\n"
            "    import libs.tools as tools\n"
            "except Exception:\n"
            "    class _TrustedToolsStub:\n"
            "        @staticmethod\n"
            "        def ConfigFmk():\n"
            "            return types.SimpleNamespace(device_type='cpu')\n"
            "    tools = _TrustedToolsStub()",
        ),
        (
            r"(?m)^import tensorflow as tf$",
            "try:\n    import tensorflow as tf\nexcept Exception:\n    tf = None",
        ),
        (
            r"(?m)^import libs\.training\.run_aclnn as training$",
            "try:\n    import libs.training.run_aclnn as training\nexcept Exception:\n    training = None",
        ),
        (
            r"(?m)^cfg_fk = tools\.ConfigFmk\(\)$",
            "cfg_fk = tools.ConfigFmk() if tools is not None and hasattr(tools, 'ConfigFmk') "
            "else types.SimpleNamespace(device_type='cpu')",
        ),
        (
            r"(?m)^from ml_dtypes import float8_e4m3fn$",
            "try:\n    from ml_dtypes import float8_e4m3fn\nexcept Exception:\n    float8_e4m3fn = None",
        ),
        (
            r"(?m)^from en_dtypes import float8_e8m0$",
            "try:\n    from en_dtypes import float8_e8m0\nexcept Exception:\n    float8_e8m0 = None",
        ),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    text = text.replace(
        "scatter_pa_blk_nz(kv_cache, norm2_res, index_table, seq_len, scatter_size)",
        "scatter_pa_blk_nz(kv_cache, norm2_res, index_table, seq_len, B, scatter_size)",
    )
    text = text.replace(
        "scatter_pa_blk_nz(kr_cache, rotary2_res, index_table, seq_len, scatter_size)",
        "scatter_pa_blk_nz(kr_cache, rotary2_res, index_table, seq_len, B, scatter_size)",
    )
    return text


def load_trusted_refnew_module():
    global _TRUSTED_MODULE
    if _TRUSTED_MODULE is not None:
        return _TRUSTED_MODULE

    source = _sanitize_refnew_source(_REFNEW_PATH.read_text(encoding="utf-8"))
    module = types.ModuleType("trusted_refnew_bridge_module")
    module.__file__ = str(_REFNEW_PATH)
    module.__dict__["types"] = types
    exec(compile(source, str(_REFNEW_PATH), "exec"), module.__dict__)
    _TRUSTED_MODULE = module
    return module


def get_param(torch_tensor_list, params):
    module = load_trusted_refnew_module()
    return module.get_param(torch_tensor_list, params)


def cal_mlaprolog(mla_param):
    module = load_trusted_refnew_module()
    return module.cal_mlaprolog(mla_param)


def scatter_pa_blk_bsnd(cache, input_, index, seq_len, B):
    module = load_trusted_refnew_module()
    return module.scatter_pa_blk_bsnd(cache, input_, index, seq_len, B)


def scatter_pa_blk_nz(cache, input_, index, seq_len, B, data_size=16):
    module = load_trusted_refnew_module()
    return module.scatter_pa_blk_nz(cache, input_, index, seq_len, B, data_size)
