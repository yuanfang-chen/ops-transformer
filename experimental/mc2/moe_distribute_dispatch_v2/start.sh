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
BS=64
HIDDEN_SIZE=8192
RUN_TYPE=0
SCALES_TYPE="float32_t"
OUTPUT_TYPE="float16_t"
MXFP=0
CASE_NAME="int8_t-float32_t-float16_t-ID001"
RANK_SIZE=2

SHORT=a:,b:,c:,d:,e:,f:,g:,h:,i:,j:,k:,l:,m:,n:,o:,p:,q:,r:,s:,t:,u:,v:,w:,x:,y:,z:,aa:,ab:,ac:
LONG=case_name:,case_type:,bs:,h:,k:,x_dtype:,quant_mode:,graph_type:,is_dispatch_scale:,\
is_continue:,tp:,localmoe:,shared_expert_rank_num:,moe_expert_num:,is_fullcore:,aic_num:,\
seed:,activeMask_Dim:,is_shared_expert_x:,comm_mode:,is_performance:,is_variable_bs:,\
shared_expert_num:,is_elastic:,share_broken_card_num:,moe_broken_card_num:,zero_expert_num:,copy_expert_num:,const_expert_num:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -a | --case_name)
    CASE_NAME="$2"
    shift 2
    ;;
    -b | --case_type)
        CASE_TYPE="$2"
        shift 2
        ;;
    -c | --bs)
        BS="$2"
        shift 2
        ;;
    -d | --h)
        H="$2"
        shift 2
        ;;
    -e | --k)
        K="$2"
        shift 2
        ;;
    -f | --x_dtype)
        X_DTYPE="$2"
        shift 2
        ;;
    -g | --quant_mode)
        QUANT_MODE="$2"
        shift 2
        ;;
    -h | --graph_type)
        GRAPH_TYPE="$2"
        shift 2
        ;;
    -i | --is_dispatch_scale)
        IS_DISPATCH_SCALE="$2"
        shift 2
        ;;
    -j | --is_continue)
        IS_CONTINUE="$2"
        shift 2
        ;;
    -k | --tp)
        TP="$2"
        shift 2
        ;;
    -l | --localmoe)
        LOCALMOE="$2"
        shift 2
        ;;
    -m | --shared_expert_rank_num)
        SHARED_EXPERT_RANK_NUM="$2"
        shift 2
        ;;
    -n | --moe_expert_num)
        MOE_EXPERT_NUM="$2"
        shift 2
        ;;
    -o | --is_fullcore)
        IS_FULLCORE="$2"
        shift 2
        ;;
    -p | --aic_num)
        AIC_NUM="$2"
        shift 2
        ;;
    -q | --seed)
        SEED="$2"
        shift 2
        ;;
    -r | --activeMask_Dim)
        ACTIVEMASK_DIM="$2"
        shift 2
        ;;
    -s | --is_shared_expert_x)
        IS_SHARED_EXPERT_X="$2"
        shift 2
        ;;
    -t | --comm_mode)
        COMM_MODE="$2"
        shift 2
        ;;
    -u | --is_performance)
        IS_PERFORMANCE="$2"
        shift 2
        ;;
    -v | --is_variable_bs)
        IS_VARIABLE_BS="$2"
        shift 2
        ;;
    -w | --shared_expert_num)
        SHARED_EXPERT_NUM="$2"
        shift 2
        ;;
    -x | --is_elastic)
        IS_ELASTIC="$2"
        shift 2
        ;;
    -y | --share_broken_card_num)
        SHARE_BROKEN_CARD_NUM="$2"
        shift 2
        ;;
    -z | --moe_broken_card_num)
        MOE_BROKEN_CARD_NUM="$2"
        shift 2
        ;;
    -aa | --zero_expert_num)
        ZERO_EXPERT_NUM="$2"
        shift 2
        ;;
    -ab | --copy_expert_num)
        COPY_EXPERT_NUM="$2"
        shift 2
        ;;
    -ac | --const_expert_num)
        CONST_EXPERT_NUM="$2"
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
# 导入环境变量
export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
echo "========== 开始执行程序 =========="
echo "RANKSIZE 大小 "$RANK_SIZE""
RANKSIZE=$((RANK_SIZE))
for ((rank_id=0; rank_id<RANKSIZE; rank_id++))
do
    (
        nohup ./out/bin/test_quant_all_reduce \
            --rank_id ${rank_id} \
            --bs ${BS} \
            --hidden_size ${HIDDEN_SIZE} \
            --run_type ${RUN_TYPE} \
            --scales_type ${SCALES_TYPE} \
            --output_type ${OUTPUT_TYPE} \
            --mxfp ${MXFP} \
            --case_name ${CASE_NAME} \
    ) &
done

wait
echo "========== 程序执行完成 =========="