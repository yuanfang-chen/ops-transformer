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
 * \file grouped_matmul_tiling_data_apt.h
 * \brief
 */
#ifndef GROUPED_MATMUL_TILING_DATA_H
#define GROUPED_MATMUL_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GroupedMatmulTilingData {
#pragma pack(push, 8)
struct GMMArray {
    // GroupedMatmul::MAX_TENSOR_CONT
    int32_t mList[128]; // 每个group的M维度大小列表，SPLIT_M模式下值为-1表示从groupList动态获取
    int32_t kList[128]; // 每个group的K维度大小列表，SPLIT_K模式下值为-1表示从groupList动态获取
    int32_t nList[128]; // 每个group的N维度大小列表
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMNoQuantBaseParams {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t singleWeight = 0;
    uint32_t singleX = 0;
    uint32_t singleY = 0;
    int32_t groupType = 0;
    uint32_t groupListType = 0;
    uint32_t hasBias = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t weightNoL2Cache = 0;
    uint32_t placeHolder = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantParams {
    uint32_t groupNum;   // 分组数量，控制kernel中group循环的次数
    uint32_t activeType; // 激活函数类型：0=None, 1=RELU, 2=GELU_TANH, 3=FASTGELU, 4=SILU
    uint32_t aQuantMode; // 输入矩阵X的量化模式：PERTENSOR/PERTOKEN/PERGROUP/PERBLOCK/MX_PERGROUP等
    uint32_t bQuantMode; // 权重矩阵W的量化模式：PERTENSOR/PERCHANNEL/PERGROUP/PERBLOCK/MX_PERGROUP等
    uint8_t singleX;     // 是否所有group共享同一个输入张量x（1=是，0=否，用于计算global buffer偏移）
    uint8_t singleW;     // 是否所有group共享同一个权重张量w（1=是，0=否，用于计算global buffer偏移）
    uint8_t singleY;     // 是否所有group输出到同一个输出张量y（1=是，0=否）
    int8_t
        groupType; // 分组类型：0=SPLIT_M（按M维度分组，m从groupList获取），2=SPLIT_K（按K维度分组，k从groupList获取）
    uint8_t groupListType; // groupList类型：0=累加值，1=单独值
    uint8_t hasBias;       // 是否有bias（1=有，0=无，影响是否调用mm_.SetBias）
    uint16_t reserved;     // 保留字段，用于内存对齐
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMWeightQuantParam {
    uint32_t groupNum;             // 分组数量
    uint32_t coreNum;              // 使用的AI Core数量
    uint64_t kSize;                // K维度全局大小
    uint64_t nSize;                // N维度全局大小
    uint8_t singleX;               // 是否所有group共享同一个输入张量x
    uint8_t singleWeight;          // 是否所有group共享同一个权重张量
    uint8_t singleY;               // 是否所有group输出到同一个输出张量y
    int8_t groupType;              // 分组类型
    uint8_t groupListType;         // groupList类型
    uint8_t hasBias;               // 是否有bias
    uint8_t cubeNumBlocksN;        // Cube在N方向的块数
    uint8_t reserved;              // 保留字段
    uint32_t groupSize;            // 分组大小
    uint32_t mainBlockSize;        // 主块大小
    uint64_t mainBlockCount;       // 主块数量
    uint16_t firstTailBlockSize;   // 第一个尾块大小
    uint16_t secondTailBlockSize;  // 第二个尾块大小
    uint16_t firstTailBlockCount;  // 第一个尾块数量
    uint16_t secondTailBlockCount; // 第二个尾块数量
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantTilingData {
    GMMQuantParams gmmQuantParams; // 核心量化参数：group数量、量化模式、分组类型等
    GMMArray gmmArray;             // 每个group的M/K/N维度大小列表
    TCubeTiling mmTilingData;      // Matmul高阶API的tiling参数：baseM/N/K、depthA1/B1、stepKa/Kb等
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMNoQuantTilingData {
    GMMNoQuantBaseParams gmmNoQuantParam;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMWeightQuantTilingData {
    GMMWeightQuantParam gmmWeightQuantParam;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

} // GroupedMatmulTilingData
#endif