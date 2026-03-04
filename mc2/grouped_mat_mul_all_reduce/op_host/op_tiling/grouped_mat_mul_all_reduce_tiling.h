/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_all_reduce_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_ALL_REDUCE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_ALL_REDUCE_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t MAX_TENSOR_CONT = 64;
constexpr uint32_t MC2_MSG_SIZE = 128;

BEGIN_TILING_DATA_DEF(GMMAllReduceBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseM);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestSize);
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_TENSOR_CONT, mList);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_TENSOR_CONT, kList);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_TENSOR_CONT, nList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMAllReduceBaseParamsOp, GMMAllReduceBaseParams)

BEGIN_TILING_DATA_DEF(GMMAllReduceCoreTiling)
TILING_DATA_FIELD_DEF_STRUCT(GMMAllReduceBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
TILING_DATA_FIELD_DEF(uint32_t, notifyOff);
TILING_DATA_FIELD_DEF(uint32_t, debugMode); // hccl debug mode
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMAllReduceCoreTilingOp, GMMAllReduceCoreTiling)

BEGIN_TILING_DATA_DEF(KFCGroupedTiling) // 同aicpu_hccl_def KFCGroupTilingData
TILING_DATA_FIELD_DEF_ARR(uint8_t, MAX_TENSOR_CONT* MC2_MSG_SIZE, msg);
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, groupTilingMagicNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KFCGroupedTilingOp, KFCGroupedTiling)

BEGIN_TILING_DATA_DEF(GMMAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KFCGroupedTiling, aicpuTiling);
TILING_DATA_FIELD_DEF_STRUCT(GMMAllReduceCoreTiling, aicoreTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupedMatMulAllReduce, GMMAllReduceTilingData)

BEGIN_TILING_DATA_DEF(Mc2Msg)                        // 同aicpu_hccl_def KFCTilingData
    TILING_DATA_FIELD_DEF(uint32_t, preparePosition);  // 任务准备位置：0表示在host完成所有通信任务的准备，1表示在kernel侧完成
    TILING_DATA_FIELD_DEF(uint64_t, sendOff);        // 发送数据地址偏移，count * dataTypeSize
    TILING_DATA_FIELD_DEF(uint64_t, recvOff);        // 接收数据地址偏移, count * dataTypeSize
    TILING_DATA_FIELD_DEF(uint64_t, tailSendOff);    // 尾块发送数据地址偏移，count * dataTypeSize
    TILING_DATA_FIELD_DEF(uint64_t, tailRecvOff);    // 尾块接收数据地址偏移, count * dataTypeSize
    TILING_DATA_FIELD_DEF(uint64_t, sendCnt);        // 整块发送数据个数
    TILING_DATA_FIELD_DEF(uint64_t, recvCnt);        // 尾块接收数据个数
    TILING_DATA_FIELD_DEF(uint64_t, tailSendCnt);    // 尾块发送数据个数
    TILING_DATA_FIELD_DEF(uint64_t, tailRecvCnt);    // 尾块接收数据个数
    TILING_DATA_FIELD_DEF(uint64_t, totalCnt);       // 总数据个数
    TILING_DATA_FIELD_DEF(uint32_t, turnNum);        // 总轮次
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);        // 尾块的轮次
    TILING_DATA_FIELD_DEF(uint32_t, stride);         // 跳写间隔
    TILING_DATA_FIELD_DEF(uint32_t, workspaceOff);   // 使用workspace作为recvbuf时的workspace偏移
    TILING_DATA_FIELD_DEF(uint32_t, notifyOff);      // device notify write/read value偏移

    TILING_DATA_FIELD_DEF(uint16_t, notifyBeginCnt); // notify write value的使用个数
    TILING_DATA_FIELD_DEF(uint16_t, notifyEndCnt);   // notify write value的使用个数
    TILING_DATA_FIELD_DEF(uint8_t, useBufferType);    // recvBuf类型
    TILING_DATA_FIELD_DEF(uint8_t, funID);           // function ID
    TILING_DATA_FIELD_DEF(uint8_t, dataType);        // hccl 数据类型
    TILING_DATA_FIELD_DEF(uint8_t, groupNum);        // groupNum

    TILING_DATA_FIELD_DEF(uint8_t, reuseMode);       // 不复用填turnNum，内存优化选择复用的内存块个数
    TILING_DATA_FIELD_DEF(uint8_t, commType);        // 通信类型
    TILING_DATA_FIELD_DEF(uint8_t, reduceOp);        // reduce op type
    TILING_DATA_FIELD_DEF(uint8_t, commOrder);       // 通信顺序，0表示通信在前，1表示通信在后
    TILING_DATA_FIELD_DEF(uint8_t, waitPolicy);      // 等待任务启动的阻塞策略，2、首轮等待，1、每轮等待。
                                                     // KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    TILING_DATA_FIELD_DEF(uint8_t, rspPolicy);       // 任务执行结束时的响应策略， 2、最后通知一次，
                                                     // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    TILING_DATA_FIELD_DEF(uint8_t, exitPolicy);      // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
                                                     // 等待AIC通知退出(可以多次执行任务)。
    TILING_DATA_FIELD_DEF(uint8_t, commAlg);         // 用于指定具体通信算法。
                                                     // 0：default, 1：fullmesh, 2：doublering, 3：switchwing
    TILING_DATA_FIELD_DEF(uint8_t, taskType);        // 从参数获取通信任务，直接下发。AIC自己发Record激活
    TILING_DATA_FIELD_DEF(uint8_t, debugMode);       // 调测模式
                                                     // 1:单独执行CUBE
                                                     // 2:单独执行Vector
                                                     // 4:单独执行AICPU KFC算子
                                                     // 8:KFC等待通信结束
                                                     // 16:KFC统计各阶段耗时
    TILING_DATA_FIELD_DEF(uint8_t, stepSize);        // 用于指定通算频率步长
    TILING_DATA_FIELD_DEF(uint8_t, sendArgIndex);    // 发送数据参数索引，对应算子原型的参数顺序
    TILING_DATA_FIELD_DEF(uint8_t, recvArgIndex);    // 接收数据参数索引，对应算子原型的参数顺序
    TILING_DATA_FIELD_DEF(uint8_t, commOutArgIndex); // 通信输出参数索引，对应算子原型的参数顺序
    TILING_DATA_FIELD_DEF(uint8_t, hasCommOut);      // 是否有通信输出
    TILING_DATA_FIELD_DEF(uint8_t, reserve);
    TILING_DATA_FIELD_DEF(uint32_t, reserve2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2MsgOp, Mc2Msg)
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_ALL_REDUCE_H