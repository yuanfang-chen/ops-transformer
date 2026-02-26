#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
# 输入选项默认初值
BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
TARGET_LINE=2
SOC_VERSION="Ascend910_93"
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
# 编译芯片型号选择
VERSION_LIST="ascend950pr_9599"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi
# 获取环境变量ASCEND_HOME_PATH
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
# 设置环境变量
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

set -e
# 编译开关
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
# 运行用例开关
if [[ "$RUN_TEST" == "ON" ]]; then    
    # 读取目录下QuantAllReduce配置文件
    CSV_FILE="${CURRENT_DIR}/scripts/MoeDistributeDispatchV2.csv"
    target_content=$(sed -n "${TARGET_LINE}p" "$CSV_FILE")
    IFS=',' read -r \
        case_name \
        case_type \
        bs \
        h \
        k \
        x_dtype \
        quant_mode \
        graph_type \
        is_dispatch_scale \
        is_continue \
        tp \
        localmoe \
        shared_expert_rank_num \
        moe_expert_num \
        is_fullcore \
        aic_num \
        seed \
        activeMask_Dim \
        is_shared_expert_x \
        comm_mode \
        is_performance \
        is_variable_bs \
        shared_expert_num \
        is_elastic \
        share_broken_card_num \
        moe_broken_card_num \
        zero_expert_num \
        copy_expert_num \
        const_expert_num \
    <<< "$target_content"    
    echo "===== 执行第${TARGET_LINE}条用例 ====="
    echo "case_name : $case_name"
    echo "case_type : $case_type"
    echo "bs : $bs"
    echo "h : $h"
    echo "k : $k"
    echo "x_dtype : $x_dtype"
    echo "quant_mode : $quant_mode"
    echo "graph_type : $graph_type"
    echo "is_dispatch_scale : $is_dispatch_scale"
    echo "is_continue : $is_continue"
    echo "tp : $tp"
    echo "localmoe : $localmoe"
    echo "shared_expert_rank_num : $shared_expert_rank_num"
    echo "moe_expert_num : $moe_expert_num"
    echo "is_fullcore : $is_fullcore"
    echo "aic_num : $aic_num"
    echo "seed : $seed"
    echo "activeMask_Dim : $activeMask_Dim"
    echo "is_shared_expert_x : $is_shared_expert_x"
    echo "comm_mode : $comm_mode"
    echo "is_performance : $is_performance"
    echo "is_variable_bs : $is_variable_bs"
    echo "shared_expert_num : $shared_expert_num"
    echo "is_elastic : $is_elastic"
    echo "share_broken_card_num : $share_broken_card_num"
    echo "moe_broken_card_num : $moe_broken_card_num"
    echo "zero_expert_num : $zero_expert_num"
    echo "copy_expert_num : $copy_expert_num"
    echo "const_expert_num : $const_expert_num"
    echo "======================================"
    # 运行数据生成脚本
    python3 ${CURRENT_DIR}/scripts/data_generate.py \
            "$case_name" \
            "$case_type" \
            "$bs" \
            "$h" \
            "$k" \
            "$x_dtype" \
            "$quant_mode" \
            "$graph_type" \
            "$is_dispatch_scale" \
            "$is_continue" \
            "$tp" \
            "$localmoe" \
            "$shared_expert_rank_num" \
            "$moe_expert_num" \
            "$is_fullcore" \
            "$aic_num" \
            "$seed" \
            "$activeMask_Dim" \
            "$is_shared_expert_x" \
            "$comm_mode" \
            "$is_performance" \
            "$is_variable_bs" \
            "$shared_expert_num" \
            "$is_elastic" \
            "$share_broken_card_num" \
            "$moe_broken_card_num" \
            "$zero_expert_num" \
            "$copy_expert_num" \
            "$const_expert_num"
    chmod +x ${CURRENT_DIR}/start.sh
    # profing开关
    if [[ "$PROFING_TYPE" == "ON" ]]; then
        # 采集性能数据
        msprof \
            --application="./start.sh \
            -a $case_name \
            -b $case_type \
            -c $bs \
            -d $h \
            -e $k \
            -f $x_dtype \
            -g $quant_mode \
            -h $graph_type \
            -i $is_dispatch_scale \
            -j $is_continue \
            -k $tp \
            -l $localmoe \
            -m $shared_expert_rank_num \
            -n $moe_expert_num \
            -o $is_fullcore \
            -p $aic_num \
            -q $seed \
            -r $activeMask_Dim \
            -s $is_shared_expert_x \
            -t $comm_mode \
            -u $is_performance \
            -v $is_variable_bs \
            -w $shared_expert_num \
            -x $is_elastic \
            -y $share_broken_card_num \
            -z $moe_broken_card_num \
            -aa $zero_expert_num \
            -ab $copy_expert_num \
            -ac $const_expert_num" \
            --output=./prof
    else
        # 直接运行不采集性能数据
        ./start.sh \
            -a $case_name \
            -b $case_type \
            -c $bs \
            -d $h \
            -e $k \
            -f $x_dtype \
            -g $quant_mode \
            -h $graph_type \
            -i $is_dispatch_scale \
            -j $is_continue \
            -k $tp \
            -l $localmoe \
            -m $shared_expert_rank_num \
            -n $moe_expert_num \
            -o $is_fullcore \
            -p $aic_num \
            -q $seed \
            -r $activeMask_Dim \
            -s $is_shared_expert_x \
            -t $comm_mode \
            -u $is_performance \
            -v $is_variable_bs \
            -w $shared_expert_num \
            -x $is_elastic \
            -y $share_broken_card_num \
            -z $moe_broken_card_num \
            -aa $zero_expert_num \
            -ab $copy_expert_num \
            -ac $const_expert_num
    fi
    # 精度对比
    python3 ${CURRENT_DIR}/scripts/data_compare.py \
            "$case_name" \
            "$bs" \
            "$hidden_size" \
            "$output_type" \
            "$ranksize" 
fi