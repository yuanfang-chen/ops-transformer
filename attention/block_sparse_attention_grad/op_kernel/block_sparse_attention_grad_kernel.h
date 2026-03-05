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
 * \file block_sparse_attention_grad_kernel.h
 * \brief Block Sparse Attention Grad Kernel Implementation
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H
#define BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

#include "attn_infra/base_defs.hpp"
#include "attn_infra/arch/arch.hpp"
#include "attn_infra/arch/cross_core_sync.hpp"
#include "attn_infra/arch/resource.hpp"
#include "attn_infra/layout/layout.hpp"

#include "attn_infra/gemm/block/block_mmad.hpp"
#include "attn_infra/gemm/dispatch_policy.hpp"
#include "attn_infra/gemm/gemm_type.hpp"
#include "attn_infra/epilogue/block/block_epilogue.hpp"
#include "attn_infra/epilogue/dispatch_policy.hpp"

using namespace NpuArch;

namespace BSA {

    constexpr int32_t PRE_LAUNCH = 2;

    template <
        class BlockMmadBSAG1_,
        class BlockMmadBSAG2_,
        class BlockMmadBSAG3_,
        class EpilogueFAGPre_,
        class EpilogueFAGSfmg_,
        class EpilogueFAGOp_,
        class EpilogueFAGPost_,
        uint32_t INPUT_LAYOUT>
    class BlockSparseAttentionGradKernel {
    public:
        using BlockMmadBSAG1 = BlockMmadBSAG1_;
        using BlockMmadBSAG2 = BlockMmadBSAG2_;
        using BlockMmadBSAG3 = BlockMmadBSAG3_;
        using EpilogueFAGPre = EpilogueFAGPre_;
        using EpilogueFAGSfmg = EpilogueFAGSfmg_;
        using EpilogueFAGOp = EpilogueFAGOp_;
        using EpilogueFAGPost = EpilogueFAGPost_;
        using ArchTag = typename BlockMmadBSAG1_::ArchTag;
        
        using ElementInput = typename BlockMmadBSAG1::ElementA;
        /// Parameters structure
        struct Params {
            // Data members
            GM_ADDR dout;
            GM_ADDR q;
            GM_ADDR k;
            GM_ADDR v;
            GM_ADDR out;
            GM_ADDR softmaxLse;
            GM_ADDR blockSparseMask; 
            GM_ADDR blockShape;
            GM_ADDR attentionMask;
            GM_ADDR actualQseqlen; 
            GM_ADDR actualKvseqlen;
            GM_ADDR dq;
            GM_ADDR dk;
            GM_ADDR dv;
            GM_ADDR workspace;
            GM_ADDR tiling;

            // Methods
            __aicore__ inline
            Params() {}

            __aicore__ inline
            Params(
                GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR softmaxLse_, GM_ADDR blockSparseMask_,
                GM_ADDR blockShape_, GM_ADDR attentionMask_, GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
                GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR workspace_, GM_ADDR tiling_data_
            ) : dout(dout_), q(q_), k(k_), v(v_), out(out_), softmaxLse(softmaxLse_), blockSparseMask(blockSparseMask_),
                blockShape(blockShape_), attentionMask(attentionMask_), actualQseqlen(actualQseqlen_), actualKvseqlen(actualKvseqlen_),
                dq(dq_), dk(dk_), dv(dv_), workspace(workspace_), tiling(tiling_data_)
            {
            }    
        };

        struct TaskInfo {
            uint32_t curBatchIdx;
            uint32_t curHeadIdx;
            uint32_t curQSeqIdx;
            uint32_t curQBlcokSparseSeqIdx;
            uint32_t curCalQSize;
            uint32_t curCalKVSize;
            uint32_t qSeqlen;
            uint32_t kvSeqlen;
            uint64_t qOffset;
            uint64_t kvOffset;
        };

        __aicore__ inline void UpdateTaskInfoCalQSize(uint32_t blockShapeX, uint32_t basicQBlockSize, TaskInfo &taskInfo) {
            taskInfo.curQBlcokSparseSeqIdx = taskInfo.curQSeqIdx / blockShapeX * blockShapeX;
            if (taskInfo.curQBlcokSparseSeqIdx + blockShapeX <= taskInfo.qSeqlen) {
                if (taskInfo.curQSeqIdx + basicQBlockSize <= taskInfo.curQBlcokSparseSeqIdx + blockShapeX) {
                    taskInfo.curCalQSize = basicQBlockSize;
                } else {
                    taskInfo.curCalQSize = taskInfo.curQBlcokSparseSeqIdx + blockShapeX - taskInfo.curQSeqIdx;
                }
            } else {
                if (taskInfo.curQSeqIdx + basicQBlockSize <= taskInfo.qSeqlen) {
                    taskInfo.curCalQSize = basicQBlockSize;
                } else {
                    taskInfo.curCalQSize = taskInfo.qSeqlen - taskInfo.curQSeqIdx;
                }
            }
        }

        __aicore__ inline void initTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen, AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                            __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                            uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                            uint32_t maxQSeqlen, uint32_t maxKvSeqlen, uint32_t blockShapeX,
                                            uint32_t basicQBlockSize, uint32_t inputLayout, uint32_t coreIdx,
                                            TaskInfo &taskInfo) {
            //BNSD:curBatch * numHeads * maxQSeqlen + curQSeqOffset; TND:cusum(gActualQseqlen[0:curBatch-1]) + curQSeqOffset
            uint32_t preQSeqLengths = tilingData->preQSeqLengths[coreIdx];
            //BNSD:curBatch * kvHeads * maxKvSeqlen; TND:cusum(gActualKvseqlen[0:curBatch-1])
            uint32_t preKVSeqLengths = tilingData->preKVSeqLengths[coreIdx];

            taskInfo.curBatchIdx =  tilingData->beginBatch[coreIdx];
            taskInfo.curHeadIdx = tilingData->beginHead[coreIdx];
            taskInfo.curQSeqIdx = tilingData->beginQSeqOffset[coreIdx];
            if (inputLayout == 0) {
                taskInfo.qOffset = preQSeqLengths * numHeads * headDim + taskInfo.curHeadIdx * headDim;
                taskInfo.kvOffset = preKVSeqLengths * kvHeads * headDim + taskInfo.curHeadIdx / groupSize * headDim;
                taskInfo.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(taskInfo.curBatchIdx)));
                taskInfo.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(taskInfo.curBatchIdx)));
            } else {
                taskInfo.qOffset = preQSeqLengths * headDim;
                taskInfo.kvOffset = preKVSeqLengths * headDim;
                taskInfo.qSeqlen = maxQSeqlen;
                taskInfo.kvSeqlen = maxKvSeqlen;
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, taskInfo);
        }

        __aicore__ inline void updateNextTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen, AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                                  uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                                  uint32_t blockShapeX, uint32_t basicQBlockSize, uint32_t inputLayout,
                                                  const TaskInfo &taskInfo, TaskInfo &nextTask) {
            nextTask = taskInfo;
            if (inputLayout == 0) { // TND, Traverse N-axis first
                if (taskInfo.curHeadIdx == numHeads - 1) {
                    nextTask.qOffset = taskInfo.qOffset + (taskInfo.curCalQSize - 1) * numHeads * headDim;
                    nextTask.kvOffset = taskInfo.kvOffset + (taskInfo.kvSeqlen - 1) * kvHeads * headDim;
                    if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                        nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                        nextTask.curHeadIdx = 0;
                        nextTask.curQSeqIdx = 0;
                        nextTask.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(nextTask.curBatchIdx)));
                        nextTask.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(nextTask.curBatchIdx)));
                    } else { // batch 不变
                        nextTask.curHeadIdx = 0;
                        nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                    }
                } else { // batch/QSeqIdx 不变
                    nextTask.qOffset = taskInfo.qOffset + headDim;
                    nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                    if (nextTask.curHeadIdx % groupSize == 0) {
                        nextTask.kvOffset = taskInfo.kvOffset + headDim;
                    }
                }
            } else { // BNSD, Traverse S-axis first
                nextTask.qOffset = taskInfo.qOffset + taskInfo.curCalQSize * headDim;
                if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                    nextTask.curQSeqIdx = 0;
                    if (taskInfo.curHeadIdx == numHeads - 1) {
                        nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                        nextTask.curHeadIdx = 0;
                        nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                    } else {
                        nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                        if (nextTask.curHeadIdx % groupSize == 0) {
                            nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                        }
                    }
                } else {
                    nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                }
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, nextTask);
        }

        // Methods
        __aicore__ inline
        BlockSparseAttentionGradKernel() {}

        template <int32_t CORE_TYPE = g_coreType>
        __aicore__ inline
        void operator()(Params const &params);

        template <>
        __aicore__ inline
        void operator()<AscendC::AIC>(Params const &params)
        {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t batch = tilingData->batch;
            uint32_t numHeads = tilingData->numHeads;
            uint32_t kvHeads = tilingData->kvHeads;
            uint32_t groupSize = numHeads / kvHeads;
            uint32_t headDim = tilingData->headDim;
            uint32_t maxQSeqlen = tilingData->maxQSeqlen;
            uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;
            uint32_t inputLayout = tilingData->inputLayout;
            uint32_t blockShapeX = tilingData->blockShapeX;
            uint32_t blockShapeY = tilingData->blockShapeY;

            uint32_t basicQBlockSize = tilingData->basicQBlockSize;
            uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
            uint32_t taskNumPerCore = tilingData->taskNumPerCore;
            uint32_t tailTaskNum = tilingData->tailTaskNum;
            uint32_t taskLength = tailTaskNum >= coreIdx ? taskNumPerCore : taskNumPerCore + 1;

            // Initialize global tensors
            AscendC::GlobalTensor<ElementInput> gDout;
            gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
            AscendC::GlobalTensor<ElementInput> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
            AscendC::GlobalTensor<ElementInput> gK;
            gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
            AscendC::GlobalTensor<ElementInput> gV;
            gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
            AscendC::GlobalTensor<bool> gBlcokSpaseMask;
            gBlcokSpaseMask.SetGlobalBuffer((__gm__ bool *)params.blockSparseMask);
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

            AscendC::GlobalTensor<float> gS;
            gS.SetGlobalBuffer((__gm__ float *)params.workspace); // 128 * 128 * 20 * 2 * sizeof(float)
            AscendC::GlobalTensor<ElementInput> gP;
            gP.SetGlobalBuffer((__gm__ ElementInput *)params.workspace); // 和 S 复用
            AscendC::GlobalTensor<float> gDp;
            gDp.SetGlobalBuffer((__gm__ float *)params.workspace); // 128 * 128 * 20 * 2 * sizeof(float)
            AscendC::GlobalTensor<ElementInput> gDs;
            gDs.SetGlobalBuffer((__gm__ ElementInput *)params.workspace); // 和 dp 复用
            AscendC::GlobalTensor<float> gDq;
            gDq.SetGlobalBuffer((__gm__ float *)params.workspace); // TND or BNSD * sizeof(float)
            AscendC::GlobalTensor<float> gDk;
            gDk.SetGlobalBuffer((__gm__ float *)params.workspace); // TND or BNSD * sizeof(float)
            AscendC::GlobalTensor<float> gDv;
            gDv.SetGlobalBuffer((__gm__ float *)params.workspace); // TND or BNSD * sizeof(float)

            TaskInfo taskInfo[2];
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim,
                maxQSeqlen, maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfo[0]);
            uint32_t qBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
            uint32_t kvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
            uint32_t batchBlocks = numHeads * qBlockNum * kvBlockNum;
            uint32_t headBlocks = qBlockNum * kvBlockNum;

            for (uint32_t i = 0; i < taskLength; i++) {
                TaskInfo& curInfo = taskInfo[i % 2];
                // loadQGM
                // loadDoutGM
                uint64_t kvBlockOffset = 0;
                for (uint32_t idx = 0; idx < kvBlockNum; idx++) {
                    uint64_t kvBlockBasicOffset = 0;
                    uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks + curInfo.curQBlcokSparseSeqIdx * kvBlockNum + idx;
                    if (gBlcokSpaseMask.GetValue(maskOffset)) {
                        uint32_t kvBlockSize = (idx != kvBlockNum - 1) ? blockShapeY : maxKvSeqlen % blockShapeY;
                        uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                        for (uint32_t loop = 0; loop < kvLoop; loop++) {
                            curInfo.curCalKVSize = (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize % basicKVBlockSize;
                            if (inputLayout == 0) {
                                curInfo.kvOffset += (kvBlockOffset * blockShapeY + kvBlockBasicOffset * basicKVBlockSize) * kvHeads * headDim;
                            } else {
                                curInfo.kvOffset += (kvBlockOffset * blockShapeY + kvBlockBasicOffset * basicKVBlockSize) * headDim;
                            }
                            // cube1(q,k)
                            // cube1(dout,v)
                            // AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
                            // AscendC::WaitEvent(VEC2CUBE);
                            // cube2(ds,k)
                            // cube3(ds,q)
                            // cube3(p,dout)
                        }
                        kvBlockBasicOffset += basicKVBlockSize;
                    }
                    kvBlockOffset += kvBlockNum;
                }
                updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim,
                    blockShapeX, basicQBlockSize, inputLayout, curInfo, taskInfo[(i + 1) % 2]);

            }
            // AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);

            // for(uint32_t i = 0; i < PRE_LAUNCH; i++) {
            //     loadQGM
            //     loadDoutGM
            //     //blockSparseMask : [B, N , ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)]
            //     for (uint32_t idx = 0; idx < ceilDiv(maxKVS, blockShapeY); idx++) {
            //         if (params.blockSparseMask[batchId][headId][seqId][idx] == 1) {
            //             cube1(q,k)
            //             cube1(dout,v)
            //             AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
            //         }
            //     }
            // }
            
            // for (uint32_t i = 2; i < taskLength; i++) {
            //     AscendC::WaitEvent(VEC2CUBE);
            //     AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
            //     cube2(ds,k)
            //     cube3(ds,q)
            //     cube3(p,dout)
            //     cube1(q,k)
            //     cube1(dout,v)
            // }

            // for(uint32_t i = 0; i < PRE_LAUNCH; i++) {
            //     AscendC::WaitEvent(VEC2CUBE);
            //     if(i == 0) {
            //         AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
            //     }
            //     cube2(ds,k)
            //     cube3(ds,q)
            //     cube3(p,dout)
            // }
            // AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
        }

        template <>
        __aicore__ inline
        void operator()<AscendC::AIV>(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
        }

    private:
        NpuArch::Arch::Resource<ArchTag> resource;
        // NpuArch::Arch::CrossCoreFlag qkReady{QK_READY_ID};
        // NpuArch::Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        // NpuArch::Arch::CrossCoreFlag pvReady{PV_READY_ID};
    };

} // namespace BSA

#endif // BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

