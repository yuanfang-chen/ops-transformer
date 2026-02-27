#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
TARGET_LINE=2
SOC_VERSION="Ascend310P3"
PROFING_TYPE="OFF"

SHORT=c:,v:,i:,b:,p:,r:,n:,f:,
LONG=cmake-rebuild:,soc-version:,install-path:,build-type:,install-prefix:,run-test:,target-line:,profing-type:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -c | --cmake-rebuild):
        CMAKE_REBUILD="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    -r | --run-test):
        RUN_TEST="$2"
        shift 2
        ;;
    -n | --target-line):
        TARGET_LINE="$2"
        shift 2
        ;;
    -f | --profing-type):
        PROFING_TYPE="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4 Ascend910_9599"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

set -e

if [[ "$CMAKE_REBUILD" == "ON" ]]; then
    rm -rf build
    mkdir -p build
    cmake -B build \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
    cmake --build build -j
    cmake --install build
fi

if [[ "$RUN_TEST" == "ON" ]]; then    
    CSV_FILE="${CURRENT_DIR}/scripts/QuantAllReduce.csv"
    target_content=$(sed -n "${TARGET_LINE}p" "$CSV_FILE")
    IFS=',' read -r \
        case_name \
        case_type \
        bs \
        hidden_size \
        input_tensor_range \
        input_tensor_type \
        scales_type \
        scales_range \
        output_type \
        ranksize \
        reduce_op \
        mxfp \
        seed \
    <<< "$target_content"    
    echo "===== 执行第${TARGET_LINE}条用例 ====="
    echo "case_name : $case_name"
    echo "case_type : $case_type"
    echo "bs : $bs"
    echo "hidden_size : $hidden_size"
    echo "input_tensor_range : $input_tensor_range"
    echo "input_tensor_type : $input_tensor_type"
    echo "scales_range : $scales_range"
    echo "scales_type : $scales_type"
    echo "output_type : $output_type"
    echo "ranksize : $ranksize"
    echo "reduce_op : $reduce_op"
    echo "seed : $seed"
    echo "mxfp : $mxfp"
    echo "======================================"
    python3 ./scripts/data_gen.py \
            "$case_name" \
            "$bs" \
            "$hidden_size" \
            "$input_tensor_range" \
            "$input_tensor_type" \
            "$scales_range" \
            "$scales_type" \
            "$output_type" \
            "$ranksize" \
            "$reduce_op" \
            "$mxfp" \
            "$seed"
    if [[ "$PROFING_TYPE" == "ON" ]]; then
        msprof \
            --application="./start.sh -b $bs -h $hidden_size -t $case_type -s $scales_type -o $output_type -m $mxfp -n $case_name" \
            --output=./prof
    else
        ./start.sh -b $bs -h $hidden_size -t $case_type -s $scales_type -o $output_type -m $mxfp -n $case_name
    fi
    python3 ./scripts/data_compare.py \
            "$case_name" \
            "$bs" \
            "$hidden_size" \
            "$output_type" \
            "$ranksize" 
fi