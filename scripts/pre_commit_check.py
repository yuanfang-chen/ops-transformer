#!/usr/bin/env python3
"""Pre-commit check for ops-transformer, scoped to one op + one SoC target.

Runs build + tests for one operator on one target SoC on the current machine.
Whoever runs this is responsible for being on a host where the build toolchain
and (for the test gates) the NPU are actually usable — the script itself
assumes nothing about the host.

Gates (in order):
  compile   bash build.sh --pkg --ops=<op> --soc=<target>
  ut        bash build.sh -u   --ops=<op> --soc=<target>
  pytest    torch/torch_npu pytest suite under <op>/tests/pytest/ (no atk).
            Requires the run pkg for <target> installed + on LD_LIBRARY_PATH.
            Skipped if the op has no pytest dir or prereqs are missing.
  st        atk-driven system tests under <op>/tests/st/*/. Executors are atk
            plugins (`from atk.tasks.api_execute import register`, inherit
            AclnnBaseApi); they cannot be invoked directly. Requires atk +
            $ST_CMD. Skipped if the op has no ST dirs or prereqs are missing.

How to use
----------
Pick a target SoC (`ascend910b` or `ascend950`, same names as `build.sh --soc=`)
and, optionally, an op (default: fused_infer_attention_score). Everything else
defaults to "run every gate whose prereqs are present".

  # Full pre-commit run for FIAS on 910b:
  scripts/pre_commit_check.py ascend910b

  # Different op — script locates <family>/<op>/ or experimental/<class>/<op>/:
  scripts/pre_commit_check.py ascend910b --op=grouped_matmul

  # Fast iteration — just compile + ut, no runtime tests:
  scripts/pre_commit_check.py ascend910b --only=compile,ut

  # Only run pytest (assumes the run pkg is already installed):
  scripts/pre_commit_check.py ascend910b --only=pytest

  # CI mode — any SKIP becomes a failure:
  scripts/pre_commit_check.py ascend910b --strict

  # Parallelism + a venv python for pytest/ST:
  JOBS=32 PYTHON_BIN=~/venvs/npu/bin/python scripts/pre_commit_check.py ascend910b

  # Enable the ST gate (requires the atk harness — not shipped with this repo):
  ST_CMD='atk run --executor {DIR}/executor_*.py --cases {DIR}/all_*.json' \\
      scripts/pre_commit_check.py ascend910b --only=st

Each gate echoes the exact command it runs and tees its output to
build/precommit/<op>_<soc>.<gate>.log. Per-gate build dirs
(build/precommit deliberately isn't one — those live as build_<op>_<soc>_<gate>/
at the repo root) are kept separate so ccache stays warm between runs but
cmake's configuration can't leak across gates.

Wire it into git as a pre-commit hook if you want:
  ln -s ../../scripts/pre_commit_check.py .git/hooks/pre-commit
(a hook with no args will print the usage and exit nonzero — supply the target
in a small wrapper, e.g. `exec scripts/pre_commit_check.py ascend910b`).

Optional env:
  JOBS         parallel jobs                (default: 16)
  PYTHON_BIN   python for pytest/ST probes  (default: python3)
  ST_CMD       atk invocation template; {DIR} expands to each ST subdir, e.g.
                 ST_CMD='atk run --executor {DIR}/executor_*.py --cases {DIR}/all_*.json'
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

TARGETS = ("ascend910b", "ascend950")
GATES = ("compile", "ut", "pytest", "st")
DEFAULT_OP = "fused_infer_attention_score"

USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def red(s: str) -> str: return _c("1;31", s)
def grn(s: str) -> str: return _c("1;32", s)
def ylw(s: str) -> str: return _c("1;33", s)
def blu(s: str) -> str: return _c("1;34", s)


def step(msg: str) -> None:
    print(f"\n{blu('==')} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{grn('[OK]')} {msg}", flush=True)


def die(msg: str) -> None:
    print(f"{red('[FAIL]')} {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def skip(strict: bool, msg: str) -> None:
    if strict:
        die(f"{msg} (strict mode: skip not allowed)")
    print(f"{ylw('[SKIP]')} {msg}", flush=True)


def print_cmd(argv: Sequence[str], cwd: Path | None = None) -> None:
    """Echo a command in `set -x` style before it runs."""
    shown = shlex.join(argv)
    if cwd is not None:
        print(f"+ (cd {cwd} && {shown})", flush=True)
    else:
        print(f"+ {shown}", flush=True)


def has_module(python_bin: str, name: str) -> bool:
    argv = [python_bin, "-c", f"import {name}"]
    print_cmd(argv)
    rc = subprocess.run(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    print(f"  -> import {name}: {'ok' if rc == 0 else 'MISSING'}", flush=True)
    return rc == 0


def tee_run(label: str, log_path: Path, cwd: Path, argv: Sequence[str]) -> None:
    """Run `argv` in `cwd`, stream output to console *and* log_path. Die on nonzero."""
    step(label)
    print_cmd(argv, cwd)
    with log_path.open("wb") as log, subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    ) as proc:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
        rc = proc.wait()
    if rc != 0:
        die(f"{label} (exit={rc}, see {log_path})")
    ok(label)


def run_build(
    repo: Path, log_dir: Path, jobs: int,
    label: str, build_dirname: str, logname: str,
    *build_args: str,
) -> None:
    cmd = [
        "bash", "build.sh", "--ninja",
        f"--build-dir={repo / build_dirname}",
        f"-j{jobs}",
        *build_args,
    ]
    tee_run(label, log_dir / logname, repo, cmd)


ATK_BANNER = """
{bar}
{bar_title}
{bar}
## The ST executors ({op}/tests/st/*/executor_*.py) are atk plugins:
##   from atk.tasks.api_execute import register
##   class ... (AclnnBaseApi): ...
## They cannot be invoked as standalone python scripts and have no in-repo
## runner — only the external Ascend `atk` harness can drive them.
##
## To enable this gate:
##   1. install atk so `{python} -c "import atk"` succeeds
##   2. export ST_CMD with the atk invocation, using {{DIR}} for each ST dir, e.g.
##        ST_CMD='atk run --executor {{DIR}}/executor_*.py --cases {{DIR}}/all_*.json'
##   3. re-run this script (optionally with --only=st)
##
## Earlier gates still ran; only the ST gate was skipped.
{bar}
"""


def warn_atk_missing(python_bin: str, op_relpath: str) -> None:
    bar = "#" * 80
    title = "##  ST TESTS CANNOT RUN — atk is missing" + " " * 37 + "##"
    print(ylw(ATK_BANNER.format(bar=bar, bar_title=title, python=python_bin, op=op_relpath)),
          flush=True)


def find_repo_root(script_path: Path) -> Path:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(script_path.parent), "rev-parse", "--show-toplevel"],
            text=True,
        )
        return Path(out.strip())
    except subprocess.CalledProcessError:
        die("cannot find repo root (not a git repo?)")
        raise  # unreachable; keeps type-checkers happy


def find_op_dir(repo: Path, op: str) -> Path:
    """Locate <family>/<op>/ or experimental/<class>/<op>/ under the repo root.

    When an op exists in both a top-level family dir and under experimental/
    (i.e. it was promoted but the incubation copy still ships), prefer the
    promoted one — that's the canonical location per CONTRIBUTING.md.
    """
    promoted: list[Path] = []
    experimental: list[Path] = []
    for entry in sorted(repo.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name == "experimental":
            for cls in sorted(entry.iterdir()):
                if cls.is_dir():
                    cand = cls / op
                    if cand.is_dir():
                        experimental.append(cand)
        else:
            cand = entry / op
            if cand.is_dir():
                promoted.append(cand)

    if len(promoted) > 1:
        die(f"op '{op}' is ambiguous across family dirs: "
            f"{', '.join(str(c) for c in promoted)}")
    if promoted:
        if experimental:
            print(ylw(f"[INFO] op '{op}' also exists under experimental/ — using promoted copy "
                      f"{promoted[0].relative_to(repo)}"), flush=True)
        return promoted[0]
    if len(experimental) > 1:
        die(f"op '{op}' is ambiguous across experimental classes: "
            f"{', '.join(str(c) for c in experimental)}")
    if experimental:
        return experimental[0]
    die(f"op '{op}' not found under any family dir (looked for <family>/{op} "
        f"and experimental/<class>/{op})")
    raise AssertionError  # unreachable; satisfies type-checkers


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pre_commit_check.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("target", nargs="?",
                   help=f"SoC target (same name as build.sh --soc=): {' or '.join(TARGETS)}")
    p.add_argument("--op", default=DEFAULT_OP,
                   help=f"operator name (default: {DEFAULT_OP})")
    p.add_argument("--only", default="",
                   help=f"comma-separated subset of gates to run; one of: {', '.join(GATES)}")
    p.add_argument("--strict", action="store_true",
                   help="SKIPs become failures")
    p.add_argument("--skip-sync", action="store_true",
                   help=argparse.SUPPRESS)  # legacy no-op kept so old muscle memory doesn't error
    args = p.parse_args(argv)

    expected = " or ".join(TARGETS)
    if args.target is None:
        p.print_usage(sys.stderr)
        sys.stderr.write(f"error: target is required ({expected})\n")
        sys.exit(2)
    if args.target not in TARGETS:
        sys.stderr.write(f"unsupported target '{args.target}' (expected: {expected})\n")
        sys.exit(2)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    bad = only - set(GATES)
    if bad:
        sys.stderr.write(f"unknown gate(s) in --only: {', '.join(sorted(bad))}; "
                         f"known: {', '.join(GATES)}\n")
        sys.exit(2)
    args.only_set = only
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    soc = args.target
    op = args.op

    jobs = int(os.environ.get("JOBS", "16"))
    python_bin = os.environ.get("PYTHON_BIN", "python3")
    st_cmd_tmpl = os.environ.get("ST_CMD", "")

    repo = find_repo_root(Path(__file__).resolve())
    op_dir = find_op_dir(repo, op)
    op_relpath = str(op_dir.relative_to(repo))
    log_dir = repo / "build" / "precommit"
    log_dir.mkdir(parents=True, exist_ok=True)

    want = (lambda g: True) if not args.only_set else (lambda g: g in args.only_set)

    step(f"op: {grn(op)} ({op_relpath})  target: {grn(soc)}  repo: {repo}  logs: {log_dir}")

    tag = f"{op}_{soc}"  # shared suffix for per-(op, soc) build dirs and logs

    # --- gate: compile -------------------------------------------------------
    if want("compile"):
        run_build(
            repo, log_dir, jobs,
            f"{op} compile ({soc})",
            f"build_{tag}_compile",
            f"{tag}.compile.log",
            "--pkg", f"--ops={op}", f"--soc={soc}",
        )

    # --- gate: ut ------------------------------------------------------------
    if want("ut"):
        run_build(
            repo, log_dir, jobs,
            f"{op} ut ({soc})",
            f"build_{tag}_ut",
            f"{tag}.ut.log",
            "-u", f"--ops={op}", f"--soc={soc}",
        )

    # --- gate: pytest --------------------------------------------------------
    # `-m ci` picks the single-op direct-call subset; drop it to include graph
    # mode (which needs torchair). Requires torch + torch_npu and the op's run
    # pkg for <target> installed + on LD_LIBRARY_PATH.
    if want("pytest"):
        pytest_dir = op_dir / "tests" / "pytest"
        if not pytest_dir.is_dir():
            skip(args.strict, f"pytest: {op} has no tests/pytest/ dir")
        else:
            step(f"probe pytest prereqs for {op}")
            if not (has_module(python_bin, "torch") and has_module(python_bin, "torch_npu")):
                skip(args.strict, f"pytest: torch / torch_npu not importable by {python_bin}")
            else:
                pytest_argv = [python_bin, "-m", "pytest", "-rA", "-s", "-v", "-m", "ci"]
                # FIAS convention: entry point is test.py. pytest's default discovery
                # (test_*.py / *_test.py) would skip it, so pass it explicitly when
                # it exists; otherwise let pytest discover whatever's in the dir.
                if (pytest_dir / "test.py").is_file():
                    pytest_argv.append("test.py")
                tee_run(
                    f"{op} pytest ({soc})",
                    log_dir / f"{tag}.pytest.log",
                    pytest_dir,
                    pytest_argv,
                )

    # --- gate: st ------------------------------------------------------------
    if want("st"):
        st_root = op_dir / "tests" / "st"
        if not st_root.is_dir():
            skip(args.strict, f"st: {op} has no tests/st/ dir")
        else:
            # An ST subdir is any dir under tests/st/ that carries both an atk
            # executor and a case matrix. Avoids hard-coding aclnn<OpName>*.
            st_dirs = sorted(
                d for d in st_root.iterdir()
                if d.is_dir()
                and any(d.glob("executor_*.py"))
                and any(d.glob("all_*.json"))
            )
            if not st_dirs:
                skip(args.strict, f"st: no ST subdirs with executor_*.py + all_*.json under {st_root}")
            else:
                step(f"probe st prereqs for {op}")
                atk_ok = has_module(python_bin, "atk")
                st_cmd_ok = bool(st_cmd_tmpl)
                if not atk_ok:
                    warn_atk_missing(python_bin, op_relpath)
                    skip(args.strict, f"st: atk not importable by {python_bin} (see banner above)")
                elif not st_cmd_ok:
                    skip(args.strict,
                         "st: atk is installed, but $ST_CMD is not set "
                         "(e.g. ST_CMD='atk run --executor {DIR}/executor_*.py --cases {DIR}/all_*.json')")
                else:
                    for d in st_dirs:
                        # ST_CMD is an arbitrary shell template; route via bash -c.
                        rendered = st_cmd_tmpl.replace("{DIR}", str(d))
                        tee_run(
                            f"{op} st {d.name} ({soc})",
                            log_dir / f"{tag}.st.{d.name}.log",
                            repo,
                            ["bash", "-c", rendered],
                        )

    step(grn(f"pre-commit checks done for op={op} target={soc}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
