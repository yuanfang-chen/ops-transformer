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

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
echo "current dir: $CURRENT_DIR"

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
fi
echo "ASCEND_INSTALL_PATH: $_ASCEND_INSTALL_PATH"

PROJECT_ROOT=$CURRENT_DIR/../../
echo "PROJECT_ROOT: $PROJECT_ROOT"

function main {
    # 1. Clearing Residual Generated Files and Log Files
    rm -rf $HOME/ascend/log/*
    rm -rf $PROJECT_ROOT/build
    rm -rf $PROJECT_ROOT/build_out
    rm -rf $CURRENT_DIR/output
    echo "clear success!"

    # 2. Compile custom package
    cd $PROJECT_ROOT
    bash build.sh --pkg --experimental --soc=ascend910b --ops=svd

    # 3. Install custom package
    CUSTOM_PATH=$(realpath "$_ASCEND_INSTALL_PATH/../")
    ./build_out/cann* --install-path=$CUSTOM_PATH

    # 4. Compile and Install PyTorch package extension
    cd $CURRENT_DIR/torch_npu_linalg/
    bash build_and_install.sh

    # 5. source custom
    source $CUSTOM_PATH/vendors/custom_transformer/bin/set_env.bash

    # 6.运行测试文件（精度和性能）
    pytest ./tests/test_svd.py
}

main
