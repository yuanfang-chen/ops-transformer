/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_kernel.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_H
#define COMPRESSOR_KERNEL_H

#include "compressor_comm.h"
#include "compressor_vector_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"
#include "compressor_comm.h"
#include "compressor_block_cube.h"
#include "compressor_block_vec.h"

using namespace AscendC;

namespace Compressor {

template <typename COMP>
class CompressorKernel {
public:
    __aicore__ inline CompressorKernel(TPipe* pipe, const optiling::CompressorTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut,
        __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    // ================================Init functions==================================
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline uint32_t CalcTcSize();
    // ================================Process functions================================
    __aicore__ inline uint32_t GetSeqLength(uint32_t index);
    __aicore__ inline uint32_t GetStartPos(uint32_t index);
    __aicore__ inline void GetCurCoreStartIdx();
    __aicore__ inline void CalcParams(RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline void InitTilingData();
    __aicore__ inline bool IsNeedExcute(const RunInfo &info);
    __aicore__ inline uint32_t GetStartIdx();
    __aicore__ inline uint32_t GetEndIdx();
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::CompressorTilingData* __restrict tilingData_;
    // ================================Task Info====================================
    ConstInfo constInfo{};

    uint32_t tcStart = 0;

    uint32_t accSeqLength = 0;
    uint32_t curActSeqLength = 0;
    uint32_t curStartPos = 0;
    uint32_t preActSeqIdx = 0;
    uint32_t preStartPosIdx = 0;

    uint32_t curBStart = 0;
    uint32_t curBEnd = 0;
    uint32_t curSStart = 0;
    uint32_t curSEnd = 0;

    uint32_t aiCoreIdx = 0;

    // 常量
    static constexpr uint32_t N = 2;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 6;

    // ==============================Service Define==============================
    CompressorBlockCube<COMP> blockCube_;
    CompressorBlockVector<COMP> vectorService;
    static constexpr uint32_t PRELOAD_NUM = 2;
    
    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<int32_t> kvStateGm_;
    GlobalTensor<int32_t> scoreStateGm_;
    GlobalTensor<int32_t> apeGm_;
    GlobalTensor<X_T> ropeSinGm_;
    GlobalTensor<X_T> ropeCosGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;

    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> preMm1ResGm;
    GlobalTensor<MM1_OUT_T> curMm1ResGm;
    GlobalTensor<VEC1_OUT_T> vec1ResGm;
};

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut,
        __gm__ uint8_t *workspace) {
    // printf("[VERSION] 20260110-001\n");
    // printf("CompressorKernel::Init!!!!!\n");

    // TODO CV非1:2需处理 
    if ASCEND_IS_AIV {
        constInfo.aiCoreIdx = GetBlockIdx() / 2;
    } else {
        constInfo.aiCoreIdx = GetBlockIdx();
    }

    // GM Init
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    kvStateGm_.SetGlobalBuffer((__gm__ int32_t *)kvState);
    scoreStateGm_.SetGlobalBuffer((__gm__ int32_t *)scoreState);
    apeGm_.SetGlobalBuffer((__gm__ int32_t *)ape);
    ropeSinGm_.SetGlobalBuffer((__gm__ X_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ X_T *)ropeCos);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    if (seqUsed != nullptr) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);

    InitTilingData();
    InitWorkspace(workspace);

    // 初始化 curActSeqLength、start_pos TODO考虑为None， 
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        curActSeqLength = cuSeqlensGm_.GetValue(1);
        accSeqLength = curActSeqLength;
        // printf("[Init] curActSeqLength:%u\n", curActSeqLength);
    }
    curStartPos = startPosGm_.GetValue(0);

    // 计算分核基本信息
    constInfo.tcSize = CalcTcSize();
    constInfo.tcBaseSize = constInfo.mBaseSize / constInfo.cmpRatio;
    constInfo.tcBasicBlockNum = (constInfo.tcSize + constInfo.tcBaseSize - 1) / constInfo.tcBaseSize;       // TC方向的基本块
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;                           // D方向的基本块
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;                        // 核分为多少组
    constInfo.singleCoreDealTcBasicNum = (constInfo.tcBasicBlockNum + constInfo.coreGroupNum - 1) / constInfo.coreGroupNum; // 处理的最大基本块数量
    // printf("[BASEINFO] tcSize:%u tcBaseSize:%u tcBasicBlockNum:%u dBasicBlockNum:%u coreGroupNum:%u singleCoreDealTcBasicNum:%u\n", constInfo.tcSize, constInfo.tcBaseSize, constInfo.tcBasicBlockNum, constInfo.dBasicBlockNum, constInfo.coreGroupNum, constInfo.singleCoreDealTcBasicNum);
    if ASCEND_IS_AIC {
        blockCube_.InitParams(constInfo);
        blockCube_.Init(x, wKv, wGate, kvState, scoreState, ape, normWeight, ropeSin, ropeCos, blockTable,
            cuSeqlens, seqUsed, startPos, cmpKvOut, kvStateOut, scoreStateOut);
        blockCube_.InitBuffers(pipe_);
    } else {
        vectorService.InitParams(constInfo);
        vectorService.Init(x, wKv, wGate, kvState, scoreState, ape, normWeight, ropeSin, ropeCos, blockTable, 
                        cuSeqlens, seqUsed, startPos, cmpKvOut, kvStateOut, scoreStateOut);
        vectorService.InitVec1GlobalTensor(preMm1ResGm, curMm1ResGm, vec1ResGm);
    }
    
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::InitTilingData() {
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.dBaseSize = tilingData_->innerSplitParams.dBaseSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.hSize = tilingData_->baseParams.hiddenSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.ropeHeadDim = tilingData_->baseParams.ropeHeadDim;
    constInfo.normEps = tilingData_->baseParams.normEps;
    constInfo.reciprocalD = tilingData_->baseParams.reciprocalD;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;
    
    constInfo.blockNum = tilingData_->pageAttentionParams.blockNum;
    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    constInfo.preMm1ResSize = tilingData_->workspaceParams.preMm1ResSize;
    constInfo.curMm1ResSize = tilingData_->workspaceParams.curMm1ResSize;
    constInfo.vec1ResSize = tilingData_->workspaceParams.vec1ResSize * N;
    // printf("[TILINGDATA] cmpRatio:%u batchSize:%u mBaseSize:%u dBaseSize:%u\n", constInfo.cmpRatio, constInfo.batchSize, constInfo.mBaseSize, constInfo.dBaseSize);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::InitWorkspace(__gm__ uint8_t *workspace) {
    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;
    uint64_t offset = 0;
    // preMm1ResGm
    preMm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.aiCoreIdx * dbWorkspaceRatio * constInfo.preMm1ResSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.preMm1ResSize * sizeof(MM1_OUT_T);

    // curMm1ResGm
    curMm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.aiCoreIdx * dbWorkspaceRatio * constInfo.curMm1ResSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.curMm1ResSize * sizeof(MM1_OUT_T);

    // vec1Res
    vec1ResGm.SetGlobalBuffer(
        (__gm__ VEC1_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.vec1ResSize * sizeof(VEC1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.vec1ResSize * sizeof(VEC1_OUT_T);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::CalcTcSize() {
    uint32_t totalBasicNum = 0;

    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        curStartPos = GetStartPos(bIdx);
        curActSeqLength = GetSeqLength(bIdx);
        totalBasicNum += GetBasicNum();

        // printf("[CalcTcSize] curActSeqLength:%u totalBasicNum:%d\n", curActSeqLength, totalBasicNum);
    }
    
    return totalBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetSeqLength(uint32_t bIdx) {
    // printf("[GetSeqLength] preActSeqIdx:%u bIdx:%u\n", preActSeqIdx, bIdx);
    if (COMP::xLayout == X_LAYOUT::TH) {
        if (preActSeqIdx != bIdx) {
            preActSeqIdx = bIdx;
            if (bIdx == 0) {
                accSeqLength = cuSeqlensGm_.GetValue(bIdx + 1);
                return accSeqLength;
            } else {
                uint32_t tmpSeqLength = accSeqLength;
                accSeqLength = cuSeqlensGm_.GetValue(bIdx + 1);
                return accSeqLength - tmpSeqLength;
            }
        } else {
            return curActSeqLength;
        }
    } else {
        return constInfo.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetStartPos(uint32_t bIdx) {
    // printf("[GetStartPos] preStartPosIdx:%u index:%u\n", preStartPosIdx, index);
    if (preStartPosIdx != bIdx) {
        curStartPos = startPosGm_.GetValue(bIdx);
        preStartPosIdx = bIdx;
        return curStartPos;
    } else {
        return curStartPos;
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::GetCurCoreStartIdx() {
    // 获取当前核开始索引

    uint32_t totalBasicNum = 0;
    // 在当前batch的seq开始索引位置
    uint32_t startIdx = 0;
    // Tc的开始位置
    tcStart = (constInfo.aiCoreIdx / constInfo.dBasicBlockNum) * constInfo.tcBaseSize * constInfo.singleCoreDealTcBasicNum;
    if (tcStart >= constInfo.tcSize) {
        curBEnd = constInfo.batchSize;
        return;
    }
    // printf("[tcStart] aiCoreIdx:%u tcStart:%u\n", constInfo.aiCoreIdx, tcStart);
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        // TODO 考虑是否有其他情况
        if (totalBasicNum == tcStart) {
            // printf("[PRINT] b:%u tcStart:%u\n", bIdx, tcStart);
            curBEnd = bIdx;
            curSEnd = startIdx;
            return;
        }
        curStartPos = GetStartPos(bIdx);
        curActSeqLength = GetSeqLength(bIdx);

        // 加上头块，若有
        uint32_t curBasicNum = 0;
        uint32_t headSize = 0;
        if (curStartPos % constInfo.cmpRatio != 0) {
            headSize = constInfo.cmpRatio - curStartPos % constInfo.cmpRatio;
            headSize = headSize > curActSeqLength ? curActSeqLength : headSize;
            curBasicNum++;
        }
        // 加上中间整块及尾块
        curBasicNum += (curActSeqLength - headSize + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
        // printf("[PRINT] b:%u tcStart:%u headSize:%u curBasicNum:%u  curStartPos:%u, curActSeqLength:%u\n", bIdx, tcStart, headSize, curBasicNum, curStartPos, curActSeqLength);
        if (totalBasicNum + curBasicNum > tcStart) {
            uint32_t curBasicNumStart = tcStart - totalBasicNum;
            if (curBasicNumStart > 0 && headSize > 0) {
                startIdx = headSize + (curBasicNumStart - 1) * constInfo.cmpRatio;
            } else {
                startIdx = curBasicNumStart * constInfo.cmpRatio;
            }
            curBEnd = bIdx;
            curSEnd = startIdx;
            return;
        }
        totalBasicNum += curBasicNum;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetBasicNum() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos % constInfo.cmpRatio != 0) {
        headSize = constInfo.cmpRatio - curStartPos % constInfo.cmpRatio;
        headSize = headSize > curActSeqLength ? curActSeqLength : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength - headSize + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::CalcParams(RunInfo &info) {
    if (curBStart >= constInfo.batchSize) {
        return;
    }

    curBStart = curBEnd;
    curSStart = curSEnd;

    // sEnd到了seq末尾，下一个seq
    curActSeqLength = GetSeqLength(curBStart);
    if (curSStart == curActSeqLength) {
        curBStart++;
        curSStart = 0;
    }

    info.bStart = curBStart;
    info.sStart = curSStart;
    
    uint32_t dealTcNum = constInfo.tcBaseSize + tcStart <= constInfo.tcSize ? constInfo.tcBaseSize : constInfo.tcSize - tcStart;
    info.dealTcNum = dealTcNum;
    tcStart += dealTcNum;
    uint32_t accBasicNum = 0;
    
    for (uint32_t bIdx = curBStart; bIdx < constInfo.batchSize; ++bIdx) {
        curBEnd = bIdx;
        info.bEnd = curBEnd;
        if (bIdx == curBStart) {
            curActSeqLength = GetSeqLength(bIdx);
            curStartPos = GetStartPos(bIdx);
            uint32_t curRemainTcNum = 0;
            // 计算起始batch的剩余seq长度 起始位置计算头块
            uint32_t headSize = 0;
            if (curStartPos % constInfo.cmpRatio != 0) {
                headSize = (constInfo.cmpRatio - curStartPos % constInfo.cmpRatio);
                headSize = headSize > curActSeqLength ? curActSeqLength : headSize;
            }
            if (curSStart == 0) {
                curRemainTcNum = (curActSeqLength - headSize + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
                curRemainTcNum = headSize == 0 ? curRemainTcNum : curRemainTcNum + 1;
            } else {
                curRemainTcNum = (curActSeqLength - curSStart + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
            }
            // printf("[GetEndIdx]  bIdx:%u accBasicNum:%u dealTcNum:%u curRemainTcNum:%u headSize:%u curStartPos:%u curActSeqLength:%u \n", bIdx, accBasicNum, dealTcNum, curRemainTcNum, headSize, curStartPos, curActSeqLength);
            if (curRemainTcNum > dealTcNum) {
                if (curSStart == 0) {
                    if (headSize == 0) {
                        curSEnd = curSStart + dealTcNum * constInfo.cmpRatio;
                    } else {
                        curSEnd = curSStart + headSize + (dealTcNum - 1) * constInfo.cmpRatio;
                    }
                    info.sEnd = curSEnd;
                    return;
                } else {
                    curSEnd = curSStart + dealTcNum * constInfo.cmpRatio;
                    info.sEnd = curSEnd;
                    return;
                }
            } else if (curRemainTcNum == dealTcNum || bIdx == constInfo.batchSize - 1) {
                curSEnd = curActSeqLength;
                info.sEnd = curSEnd;
                return;
            } else {
                accBasicNum += curRemainTcNum;
            }
            
        } else {
            curActSeqLength = GetSeqLength(bIdx);
            curStartPos = GetStartPos(bIdx);
            uint32_t curBasicNum = GetBasicNum();
            // printf("[GetEndIdx] accBasicNum:%u curBasicNum:%u dealTcNum:%u\n", accBasicNum, curBasicNum, dealTcNum);
            if (accBasicNum + curBasicNum > dealTcNum) {
                uint32_t headSize = 0;
                if (curStartPos % constInfo.cmpRatio != 0) {
                    headSize = constInfo.cmpRatio - curStartPos % constInfo.cmpRatio;
                    // 处理seq不足head大小的情况
                    headSize = headSize > curActSeqLength ? curActSeqLength : headSize;
                }
                uint32_t curBasicNumEnd = dealTcNum - accBasicNum;
                if (headSize == 0) {
                    curSEnd = curBasicNumEnd * constInfo.cmpRatio;
                } else {
                    curSEnd = headSize + (curBasicNumEnd - 1) * constInfo.cmpRatio;
                }
                curSEnd = curSEnd > curActSeqLength ? curActSeqLength : curSEnd;
                info.sEnd = curSEnd;
                return;
            } else if (accBasicNum + curBasicNum == dealTcNum) {
                curSEnd = curActSeqLength;
                info.sEnd = curSEnd;
                return;
            }
            accBasicNum += curBasicNum;
        }
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernel<COMP>::IsNeedExcute(const RunInfo &info) {
    if (info.bStart == constInfo.batchSize) {
        return false;
    }
    return true;
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeMm1(const RunInfo &info) {
    printf("[COMPUTE] MM1 curBStart:%d curBEnd:%d curSStart:%d curSEnd:%d\n", curBStart, curBEnd, curSStart, curSEnd);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeVec1(const RunInfo &info) {
    printf("[COMPUTE] VEC1 curBStart:%d curBEnd:%d curSStart:%d curSEnd:%d\n", curBStart, curBEnd, curSStart, curSEnd);
    vectorService.ComputeVec1(info);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeVec2(const RunInfo &info) {
    printf("[COMPUTE] VEC2 curBStart:%d curBEnd:%d curSStart:%d curSEnd:%d\n", curBStart, curBEnd, curSStart, curSEnd);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Process() {
    // printf("CompressorKernel::Process!!!!!\n");
    if ASCEND_IS_AIC {
        blockCube_.AllocEventID(pipe_);
    }

    RunInfo extraInfo[1];
    GetCurCoreStartIdx();

    
    for (uint32_t i = 0; i < constInfo.singleCoreDealTcBasicNum; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        
        // 获取各切分轴的起始核结束索引
        CalcParams(extraInfo0);
        bool isNeedExcute = IsNeedExcute(extraInfo0);
        if ASCEND_IS_AIC {
            if (isNeedExcute) {
                ComputeMm1(extraInfo0);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
            }
        } else {
            if (isNeedExcute) {
                CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
                ComputeVec1(extraInfo0);
            }
            // 累积N个基本块/最后一次循环
            if ((i + 1) % N == 0 || (i + 1) == constInfo.singleCoreDealTcBasicNum) {
                SyncAll();
                if (isNeedExcute) {
                    ComputeVec2(extraInfo0);
                }
            }
        }
    }

    if ASCEND_IS_AIC {
        blockCube_.FreeEventID(pipe_);
    }

}

} // namespace Compressor

#endif // COMPRESSOR_KERNEL_H