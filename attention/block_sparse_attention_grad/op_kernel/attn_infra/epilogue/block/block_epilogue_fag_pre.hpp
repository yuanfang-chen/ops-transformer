/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class UpdateType_,
    class InputType_>
class BlockEpilogue<
    EpilogueAtlasA2FAGPre,
    OutputType_,
    UpdateType_,
    InputType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        // Data members
        // GM_ADDR dq;
        // GM_ADDR dk;
        // GM_ADDR dv;
        // GM_ADDR blockSparseMask; 
        // GM_ADDR blockShape;
        GM_ADDR dqWrk; 
        GM_ADDR dkWrk;
        GM_ADDR dvWrk;
        GM_ADDR tilingData;

        // Methods
        __aicore__ inline
        Params() {}

        __aicore__ inline
        Params(
            // GM_ADDR dq_, GM_ADDR dv_, GM_ADDR dv_,
            GM_ADDR dqWrk_, GM_ADDR dkWrk_, GM_ADDR dvWrk_,
            GM_ADDR tilingData_
        ) : 
            // dq(dq_), dk(dk_), dv(dv_)
            dqWrk(dqWrk_), dkWrk(dkWrk_), dvWrk(dvWrk_),
            tilingData(tilingData_)
        {
            
        }    
    };

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    uint64_t cBlockIdx;

    uint64_t qPreBlockFactor = 0;
    uint64_t qPreBlockTotal = 0;
    uint64_t qPreTailNumTmp = 0;
    uint64_t qPreTailNum = 0;
    uint64_t qSizeAlign = 0;
    uint64_t initdqSize = 0;
    uint64_t dqOffset = 0;

    uint64_t kvPreBlockFactor = 0;
    uint64_t kvPreBlockTotal = 0;
    uint64_t kvPreTailNumTmp = 0;
    uint64_t kvPreTailNum = 0;
    uint64_t kvSizeAlign = 0;
    uint64_t initdkSize = 0;
    uint64_t dkvOffset = 0;

    uint64_t usedCoreNum = 0;

    __aicore__ inline
    BlockEpilogue(Params const &params)
    {
        cBlockIdx = GetBlockIdx();
        __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedCoreNum = tilingData->usedVecCoreNum; // 先按这个把，得适配
        if (cBlockIdx >= usedCoreNum) {
            return;
        }

        qSizeAlign = tilingData->dqSize;
        kvSizeAlign = tilingData->dkvSize;
        qPreBlockFactor = (qSizeAlign + usedCoreNum - 1) / usedCoreNum;
        qPreBlockTotal = (qSizeAlign + qPreBlockFactor - 1) / qPreBlockFactor;
        qPreTailNumTmp = qSizeAlign % qPreBlockFactor;
        qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

        kvPreBlockFactor = (kvSizeAlign + usedCoreNum - 1) / usedCoreNum;
        kvPreBlockTotal = (kvSizeAlign + kvPreBlockFactor - 1) / kvPreBlockFactor;
        kvPreTailNumTmp = kvSizeAlign % kvPreBlockFactor;
        kvPreTailNum = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dqWrk);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dkWrk);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dvWrk);

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreTailNum : qPreBlockFactor;
        dqOffset = ((uint64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreTailNum : kvPreBlockFactor;
        dkvOffset = ((uint64_t)cBlockIdx) * kvPreBlockFactor;
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline
    void operator()();

    template <>
    __aicore__ inline
    void operator()<AscendC::AIC>()
    {

    }

    template <>
    __aicore__ inline
    void operator()<AscendC::AIV>()
    {
        if (cBlockIdx >= usedCoreNum) {
            return;
        }

        // process clear dq dk dv workspace
        if (cBlockIdx < qPreBlockTotal) {
            matmul::InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }

        if (cBlockIdx < kvPreBlockTotal) {
            matmul::InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            matmul::InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
        }
    }

};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
