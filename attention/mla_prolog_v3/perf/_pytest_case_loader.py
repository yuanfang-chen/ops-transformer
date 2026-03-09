from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from _perf_model import ACTUAL_SEQ_DISABLED
from _perf_model import ACTUAL_SEQ_EN_Q_LEN


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PYTEST_TESTCASES = THIS_DIR.parent / "tests" / "pytest" / "testcases.py"


@dataclass(frozen=True)
class PytestCaseRecord:
    name: str
    source_set: str
    raw_params: Dict[str, Any]
    perf_mapping: Dict[str, Any]


def _attribute_to_string(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attribute_to_string(node.value)}.{node.attr}"
    raise ValueError(f"unsupported attribute expression: {ast.dump(node)}")


def _eval_ast(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_ast(item, env) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(item, env) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval_ast(key, env): _eval_ast(value, env) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Attribute):
        return _attribute_to_string(node)
    if isinstance(node, ast.Subscript):
        container = _eval_ast(node.value, env)
        key = _eval_ast(node.slice, env)
        return container[key]
    raise ValueError(f"unsupported AST node in pytest case loader: {ast.dump(node)}")


def _load_symbol_table(path: Path | str) -> Dict[str, Any]:
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    env: Dict[str, Any] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            continue
        env[statement.targets[0].id] = _eval_ast(statement.value, env)
    return env


def _unwrap_singleton_lists(params: Mapping[str, Any]) -> Dict[str, Any]:
    unwrapped: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"positive pytest case field {key!r} must contain exactly one value")
            unwrapped[key] = value[0]
        else:
            unwrapped[key] = value
    return unwrapped


def normalize_pytest_case(case_name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _unwrap_singleton_lists(params)
    cache_mode = str(raw.get("cache_mode", "BSND")).upper()
    bs_fused_flag = int(raw.get("bs_fused_flag", 0))
    actual_seq_mode = (
        ACTUAL_SEQ_EN_Q_LEN
        if bs_fused_flag == 1 and cache_mode in {"PA_BLK_BSND", "PA_BLK_NZ"}
        else ACTUAL_SEQ_DISABLED
    )
    normalized = dict(raw)
    normalized["case_name"] = case_name
    normalized["cache_mode"] = cache_mode
    normalized["T"] = int(raw["batch_size"]) * int(raw["q_seq"])
    normalized["actual_seq_mode"] = actual_seq_mode
    normalized["smooth_scales_enabled"] = bool(int(raw.get("smooth_scales_cq_flag", 0)))
    return normalized


def load_pytest_case_catalog(path: Path | str = DEFAULT_PYTEST_TESTCASES) -> Dict[str, Dict[str, Any]]:
    env = _load_symbol_table(path)
    test_params = env.get("TEST_PARAMS")
    if not isinstance(test_params, dict):
        raise ValueError("failed to resolve TEST_PARAMS from pytest testcases file")
    return {str(name): _unwrap_singleton_lists(params) for name, params in test_params.items()}


def _resolve_case_names(symbol_table: Mapping[str, Any], *, case_set: str) -> List[str]:
    test_params = symbol_table.get("TEST_PARAMS")
    if not isinstance(test_params, dict):
        raise ValueError("failed to resolve TEST_PARAMS from pytest testcases file")
    if case_set == "all":
        return list(test_params.keys())
    if case_set != "enabled":
        raise ValueError(f"unsupported pytest case set {case_set!r}")
    enabled = symbol_table.get("ENABLED_PARAMS")
    if not isinstance(enabled, Sequence):
        raise ValueError("failed to resolve ENABLED_PARAMS from pytest testcases file")
    names_by_identity = {id(params): str(name) for name, params in test_params.items()}
    ordered_names: List[str] = []
    for item in enabled:
        match = names_by_identity.get(id(item))
        if match is None:
            for name, params in test_params.items():
                if item == params:
                    match = str(name)
                    break
        if match is None:
            raise ValueError("failed to map ENABLED_PARAMS entry back to TEST_PARAMS case name")
        ordered_names.append(match)
    return ordered_names


def load_pytest_case_set(
    path: Path | str = DEFAULT_PYTEST_TESTCASES,
    *,
    case_set: str = "enabled",
    case_name: str | None = None,
) -> List[PytestCaseRecord]:
    symbol_table = _load_symbol_table(path)
    test_params = symbol_table.get("TEST_PARAMS")
    if not isinstance(test_params, dict):
        raise ValueError("failed to resolve TEST_PARAMS from pytest testcases file")
    selected_names = _resolve_case_names(symbol_table, case_set=case_set)
    if case_name is not None:
        if case_name not in selected_names:
            raise ValueError(f"pytest case {case_name!r} not found in case set {case_set!r}")
        selected_names = [case_name]
    records: List[PytestCaseRecord] = []
    for name in selected_names:
        raw = _unwrap_singleton_lists(test_params[name])
        records.append(
            PytestCaseRecord(
                name=str(name),
                source_set=case_set,
                raw_params=raw,
                perf_mapping=normalize_pytest_case(str(name), raw),
            )
        )
    return records
