#!/bin/bash
# cannsim_exec.sh — Run operator binary under cannsim simulation
CANN=/mnt/workspace/wangqi/Ascend/cann-9.0.0
VENDOR=custom_transformer

export ASCEND_HOME_PATH=$CANN
export ASCEND_TOOLKIT_HOME=$CANN
export ASCEND_OPP_PATH=$CANN/opp
export ASCEND_AICPU_PATH=$CANN

# Single path only (Rule 7)
export ASCEND_CUSTOM_OPP_PATH=$CANN/opp/vendors/$VENDOR

export PATH=$CANN/bin:$CANN/tools/ccec_compiler/bin:/usr/local/bin:/usr/bin:/bin:$PATH

# Enable verbose CANN logging
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1

CAMODEL=$CANN/tools/simulator/Ascend950PR_9599/camodel
export LD_LIBRARY_PATH=${CAMODEL}:$CANN/lib64:$CANN/aarch64-linux/lib64:$CANN/opp/vendors/$VENDOR/op_api/lib:$CANN/opp/vendors/$VENDOR/op_proto/lib/linux/aarch64:$CANN/opp/vendors/$VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$CANN/lib64/plugin/opskernel:$CANN/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver

BINARY_DIR="$(dirname "$1")"
mkdir -p "$BINARY_DIR"/{log/ub_log,log_ca,run_log}

cd "$BINARY_DIR"
echo "[cannsim] Running: $(basename "$1")"
echo "[cannsim] ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
./$(basename "$1") "${@:2}" 2>&1
EXIT_CODE=$?
echo "[cannsim] Exit code: $EXIT_CODE"
exit $EXIT_CODE
