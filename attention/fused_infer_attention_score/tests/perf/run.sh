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
#   - CANN toolkit installed somewhere — this script auto-discovers it via
#     /usr/local/Ascend/ascend-toolkit/set_env.sh or $HOME/Ascend/ascend-toolkit/set_env.sh.
#     If neither exists, set ASCEND_HOME_PATH manually before invoking.
#   - Custom run package installed: from the repo root,
#       bash build.sh --pkg --ops=fused_infer_attention_score --soc=ascend950
#       ./build_out/cann-ops-transformer-*.run --install-path=$(realpath ${ASCEND_HOME_PATH}/../)
#     The vendor's set_env.bash ends up under either:
#       ${ASCEND_OPP_PATH}/vendors/<vendor>_transformer/bin/set_env.bash   (CANN-opp install)
#       ${ASCEND_HOME_PATH}/../vendors/<vendor>_transformer/bin/set_env.bash (top-level install)
#
# Usage: bash run.sh [--vendor=NAME]   # default vendor: custom

set -e
# Don't `set -u` — CANN's set_env.sh references LD_LIBRARY_PATH /
# PYTHONPATH / CMAKE_PREFIX_PATH unconditionally and warns under `-u`,
# which is harmless but pollutes our log.

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
VENDOR="custom"
for arg in "$@"; do
    case "$arg" in
        --vendor=*) VENDOR="${arg#--vendor=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# 1. CANN env. Source set_env.sh if not already sourced.
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for cand in /usr/local/Ascend/ascend-toolkit/set_env.sh \
                "${HOME:-/root}/Ascend/ascend-toolkit/set_env.sh"; do
        if [ -f "$cand" ]; then
            # shellcheck disable=SC1090
            source "$cand"
            break
        fi
    done
fi
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ASCEND_HOME_PATH still unset after probing standard locations." >&2
    echo "Set it manually or source CANN's set_env.sh." >&2
    exit 1
fi

# 2. Vendor env (optional). Some run-pkg installations don't ship a
# set_env.bash — that's fine, build.sh's --run_example fallback path
# (build.sh:582-583) reads ${ASCEND_OPP_PATH}/vendors/.../op_api/{lib,include}
# directly without needing the vendor env to be sourced. If a set_env.bash
# IS present, source it so ASCEND_CUSTOM_OPP_PATH gets honored.
for cand in "${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/${VENDOR}_transformer/bin/set_env.bash" \
            "$(realpath "${ASCEND_HOME_PATH}/../")/vendors/${VENDOR}_transformer/bin/set_env.bash"; do
    if [ -f "$cand" ]; then
        # shellcheck disable=SC1090
        source "$cand"
        break
    fi
done

# 3. Resolve include/lib paths.
ARCH_INFO=$(uname -m)
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
ACLNNOP_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

# Sanity-check the resolved paths before invoking g++.
if [ ! -f "${ACLNNOP_INCLUDE_PATH}/aclnn_fused_infer_attention_score_v5.h" ]; then
    echo "FIAS V5 header not found at ${ACLNNOP_INCLUDE_PATH}/" >&2
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH} may not have aclnnop headers for arch ${ARCH_INFO}." >&2
    exit 1
fi

# CANN ships the FIAS aclnn entry points in libopapi_transformer.so (the
# standard CANN aclnn lib), not in the vendor's libcust_opapi.so — that
# only contains ops the run package actually emitted as new aclnn functions
# (e.g. ApplyRotaryPosEmb). FIAS V5 already exists in CANN proper; the
# custom run package overrides the device-side kernel via the op_master
# scheduler reading ${ASCEND_OPP_PATH}/vendors/config.ini, NOT via a
# replacement aclnn symbol. So we link against -lopapi_transformer.
TRANSFORMER_LIB_DIR="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64"
if [ ! -f "${TRANSFORMER_LIB_DIR}/libopapi_transformer.so" ]; then
    echo "libopapi_transformer.so not found at ${TRANSFORMER_LIB_DIR}/" >&2
    exit 1
fi

BIN="${CURRENT_DIR}/fias_perf"

echo "== Building fias_perf =="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-(unset)}"
echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-(unset)}"
echo "TRANSFORMER_LIB_DIR=${TRANSFORMER_LIB_DIR}"

# Compile flags mirror build.sh's --run_example PKG_MODE='' path
# (build.sh:570-576), which links against -lopapi_transformer for ops
# already present in CANN proper (FIAS is one). We don't need
# -lcust_opapi here because FIAS V5's aclnn entry points are not in the
# vendor's lib.
g++ "${CURRENT_DIR}/fias_perf.cpp" \
    -O2 -std=c++17 \
    -I "${INCLUDE_PATH}" -I "${ACLNNOP_INCLUDE_PATH}" \
    -L "${TRANSFORMER_LIB_DIR}" -L "${EAGER_LIBRARY_PATH}" \
    -lopapi_math -lopapi_transformer -lascendcl -lnnopbase -lc_sec \
    -Wl,-rpath="${TRANSFORMER_LIB_DIR}" \
    -o "${BIN}"

echo "== Running fias_perf =="
"${BIN}"
