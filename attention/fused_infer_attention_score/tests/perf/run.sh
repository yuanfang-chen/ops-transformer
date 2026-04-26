#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Build and run the FIAS NPU kernel-time benchmark.
#
# Prereqs:
#   - CANN toolkit installed; set_env.sh sourced (so $ASCEND_HOME_PATH is set).
#   - Custom run package built and installed: from the repo root,
#       bash build.sh --pkg --ops=fused_infer_attention_score --soc=ascend950
#       ./build_out/cann-ops-transformer-*.run --install-path=$(realpath $ASCEND_HOME_PATH/../)
#   - Run package's set_env.bash sourced (this script does that automatically
#     via $CUSTOM_PATH/vendors/custom_transformer/bin/set_env.bash if present).
#
# Usage: bash run.sh [--vendor=NAME]   # default vendor: custom

set -eu

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
VENDOR="custom"
for arg in "$@"; do
    case "$arg" in
        --vendor=*) VENDOR="${arg#--vendor=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ASCEND_HOME_PATH not set. Source CANN's set_env.sh first." >&2
    exit 1
fi

# Source the installed run package's vendor env (mirrors
# experimental/attention/fused_infer_attention_score/run.sh:47). The set_env.bash
# adds the vendor's op_api/lib to LD_LIBRARY_PATH and exposes ASCEND_CUSTOM_OPP_PATH.
CUSTOM_PATH=$(realpath "${ASCEND_HOME_PATH}/../")
VENDOR_ENV="${CUSTOM_PATH}/vendors/${VENDOR}_transformer/bin/set_env.bash"
if [ -f "$VENDOR_ENV" ]; then
    # shellcheck disable=SC1090
    source "$VENDOR_ENV"
else
    echo "vendor set_env.bash not found at $VENDOR_ENV — did you install the run package?" >&2
    exit 1
fi

ARCH_INFO=$(uname -m)
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
ACLNNOP_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

# Resolve vendor lib/include paths the same way build.sh:582-588 does.
if [ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
    CUST_VENDORS_PATH=$(dirname "${ASCEND_CUSTOM_OPP_PATH%%:*}")
    CUST_LIBRARY_PATH="${CUST_VENDORS_PATH}/${VENDOR}_transformer/op_api/lib"
    CUST_INCLUDE_PATH="${CUST_VENDORS_PATH}/${VENDOR}_transformer/op_api/include"
else
    CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/${VENDOR}_transformer/op_api/lib"
    CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/${VENDOR}_transformer/op_api/include"
fi

BIN="${CURRENT_DIR}/fias_perf"

echo "== Building fias_perf =="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "CUST_LIBRARY_PATH=${CUST_LIBRARY_PATH}"

# Compile flags mirror build.sh's --run_example PKG_MODE=cust path (build.sh:589-596).
g++ "${CURRENT_DIR}/fias_perf.cpp" \
    -O2 -std=c++17 \
    -I "${CUST_INCLUDE_PATH}" -I "${INCLUDE_PATH}" -I "${ACLNNOP_INCLUDE_PATH}" \
    -L "${CUST_LIBRARY_PATH}" -L "${EAGER_LIBRARY_PATH}" \
    -lopapi_math -lcust_opapi -lascendcl -lnnopbase -lc_sec \
    -Wl,-rpath="${CUST_LIBRARY_PATH}" \
    -o "${BIN}"

echo "== Running fias_perf =="
"${BIN}"
