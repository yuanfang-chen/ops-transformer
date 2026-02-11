/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing.h
 * \brief
 */
#ifndef __CHUNK_GATED_DELTA_RULE_H_
#define __CHUNK_GATED_DELTA_RULE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
    
using namespace AscendC;
    
constexpr int32_t BUFFER_NUM = 2;

struct ChunkGatedDeltaRuleInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gamma;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};


//template <typename T>
class ChunkGatedDeltaRule {
public:
    __aicore__ inline ChunkGatedDeltaRule() {};

    __aicore__ inline void Init(/* */);
    __aicore__ inline void Process(/* */);

private:
    __aicore__ inline void CopyIn(/* */);
    __aicore__ inline void CopyOut(/* */);
    __aicore__ inline void Compute(/* */);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> X;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Y;
};

__aicore__ inline void ChunkGatedDeltaRule::Init(/* */)
{
}

__aicore__ inline void ChunkGatedDeltaRule::CopyIn(/* */)
{
}

__aicore__ inline void ChunkGatedDeltaRule::CopyOut(/* */)
{
}

__aicore__ inline void ChunkGatedDeltaRule::Compute(/* */)
{
}

__aicore__ inline void ChunkGatedDeltaRule::Process()
{ 
}

} // namespace ChunkGatedDeltaRule
#endif  // __CHUNK_GATED_DELTA_RULE_H_