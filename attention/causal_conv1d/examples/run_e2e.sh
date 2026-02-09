#!/usr/bin/env bash
set -euo pipefail

# End-to-end workflow:
#   1) Generate a varlen prefill/extend test case (inputs as .bin)
#   2) Build/install the custom package and run the device example
#   3) Run PyTorch CPU reference on the same inputs
#   4) Compare device outputs vs PyTorch reference

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BUILD_SH="${BUILD_SH:-$REPO_ROOT/build.sh}"

# Keep these in sync with the hard-coded values in test_aclnn_causal_conv1d.cpp.
CASE_DIR="$SCRIPT_DIR/case"
BATCH_SIZE="32"
TOTAL_SEQLEN="8192"
DIM="1024"
WIDTH="4"
INIT_STATE_SEQS="1"
ACTIVATION_MODE="1"
PAD_SLOT_ID="-1"

python_cmd=""
if command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
elif command -v py >/dev/null 2>&1; then
  python_cmd="py -3"
else
  echo "ERROR: python not found. Need python3/python/py in PATH." >&2
  exit 1
fi

echo "[1/4] Generate case inputs -> $CASE_DIR"
mkdir -p "$CASE_DIR"
$python_cmd "$SCRIPT_DIR/gen_case.py" \
  --out_dir "$CASE_DIR" \
  --batch_size "$BATCH_SIZE" \
  --total_seqlen "$TOTAL_SEQLEN" \
  --dim "$DIM" \
  --width "$WIDTH" \
  --init_state_seqs "$INIT_STATE_SEQS" \
  --activation_mode "$ACTIVATION_MODE" \
  --pad_slot_id "$PAD_SLOT_ID"

required_inputs=(
  "$CASE_DIR/x.bin"
  "$CASE_DIR/weight.bin"
  "$CASE_DIR/bias.bin"
  "$CASE_DIR/conv_states.bin"
  "$CASE_DIR/query_start_loc.bin"
  "$CASE_DIR/cache_indices.bin"
  "$CASE_DIR/has_initial_state.bin"
)
for f in "${required_inputs[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required input: $f" >&2
    exit 2
  fi
done

echo
echo "[2/4] Build/install custom pkg + run device example"
echo "Expected outputs after this step:"
echo "  - $CASE_DIR/y_device.bin"
echo "  - $CASE_DIR/conv_states_device.bin"
echo

if [[ ! -f "$BUILD_SH" ]]; then
  echo "ERROR: build.sh not found at: $BUILD_SH" >&2
  echo "Set BUILD_SH=/path/to/build.sh or run from the repo checkout." >&2
  exit 2
fi

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_BUILD_JOBS="$(nproc)"
else
  DEFAULT_BUILD_JOBS="8"
fi

BUILD_JOBS="${BUILD_JOBS:-$DEFAULT_BUILD_JOBS}"
VENDOR_NAME="${VENDOR_NAME:-custom}"
SOC="${SOC:-}"
COMPILATION_LOG="${COMPILATION_LOG:-$REPO_ROOT/compilation.log}"
SKIP_PKG_BUILD="${SKIP_PKG_BUILD:-0}"
SKIP_PKG_INSTALL="${SKIP_PKG_INSTALL:-0}"
RUN_PKG_PATH="${RUN_PKG_PATH:-}"

echo "Start custom pkg building and executing"

if [[ "$SKIP_PKG_BUILD" != "1" ]]; then
  pkg_cmd=(bash "$BUILD_SH" --pkg --ops=causal_conv1d "-j${BUILD_JOBS}")
  if [[ -n "$SOC" ]]; then
    pkg_cmd+=("--soc=${SOC}")
  fi
  if [[ -n "$VENDOR_NAME" ]]; then
    pkg_cmd+=("--vendor_name=${VENDOR_NAME}")
  fi

  echo "Building custom package (log: $COMPILATION_LOG)"
  "${pkg_cmd[@]}" &> "$COMPILATION_LOG"
fi

if [[ "$SKIP_PKG_INSTALL" != "1" ]]; then
  if [[ -z "$RUN_PKG_PATH" ]]; then
    shopt -s nullglob
    vendor_candidates=(
      "$REPO_ROOT/build"/*"${VENDOR_NAME}"*.run
      "$REPO_ROOT/output"/*"${VENDOR_NAME}"*.run
      "$REPO_ROOT/build_out"/*"${VENDOR_NAME}"*.run
    )
    candidates=("${vendor_candidates[@]}")
    if (( ${#candidates[@]} == 0 )); then
      candidates=(
        "$REPO_ROOT/build"/*.run
        "$REPO_ROOT/output"/*.run
        "$REPO_ROOT/build_out"/*.run
      )
    fi
    shopt -u nullglob

    if (( ${#candidates[@]} == 0 )); then
      echo "ERROR: no .run package found under $REPO_ROOT/{build,output,build_out}." >&2
      echo "Set RUN_PKG_PATH=/path/to/package.run or run the build step." >&2
      exit 2
    fi

    RUN_PKG_PATH="${candidates[0]}"
    for f in "${candidates[@]}"; do
      if [[ "$f" -nt "$RUN_PKG_PATH" ]]; then
        RUN_PKG_PATH="$f"
      fi
    done
  fi

  if [[ ! -f "$RUN_PKG_PATH" ]]; then
    echo "ERROR: .run package not found: $RUN_PKG_PATH" >&2
    exit 2
  fi

  echo "Installing run package: $RUN_PKG_PATH"
  bash "$RUN_PKG_PATH"
fi

run_cmd=(bash "$BUILD_SH" --run_example causal_conv1d eager cust)
if [[ -n "$SOC" ]]; then
  run_cmd+=("--soc=${SOC}")
fi
if [[ -n "$VENDOR_NAME" ]]; then
  run_cmd+=("--vendor_name=${VENDOR_NAME}")
fi

echo "Running device example and dumping outputs to: $CASE_DIR"
CAUSAL_CONV1D_CASE_DIR="$CASE_DIR" "${run_cmd[@]}"

if [[ ! -f "$CASE_DIR/y_device.bin" || ! -f "$CASE_DIR/conv_states_device.bin" ]]; then
  echo "ERROR: device outputs not found in $CASE_DIR." >&2
  echo "Step [2/4] did not produce expected device outputs." >&2
  exit 2
fi

echo
echo "[3/4] Run CPU reference"
$python_cmd "$SCRIPT_DIR/cpu_causal_conv1d_ref.py" --case_dir "$CASE_DIR"

echo
echo "[4/4] Compare device vs CPU reference"
$python_cmd "$SCRIPT_DIR/compare_case.py" --case_dir "$CASE_DIR"

echo
echo "OK: E2E compare passed"
