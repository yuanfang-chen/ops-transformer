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

using namespace NpuArch;

namespace BSA {
    template <
        class BlockMmadFAGCube1_,
        class BlockMmadFAGCube2_,
        class BlockMmadFAGCube3_,
        class EpilogueFAGPre_,
        class EpilogueFAGSfmg_,
        class EpilogueFAGOp_,
        class EpilogueFAGPost_
        uint32_t INPUT_LAYOUT>
    class BlockSparseAttentionGradKernel {
    public:
        using BlockMmadFAGCube1 = BlockMmadFAGCube1_;
        using BlockMmadFAGCube2 = BlockMmadFAGCube2_;
        using BlockMmadFAGCube3 = BlockMmadFAGCube3_;
        using EpilogueFAGPre = EpilogueFAGPre_;
        using EpilogueFAGSfmg = EpilogueFAGSfmg_;
        using EpilogueFAGOp = EpilogueFAGOp_;
        using EpilogueFAGPost = EpilogueFAGPost_;
        using ArchTag = typename BlockMmadFAGCube1_::ArchTag;
        
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
            GM_ADDR tiling_data;

            // Methods
            CATLASS_DEVICE
            Params() {}

            CATLASS_DEVICE
            Params(
                GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR softmaxLse_, GM_ADDR blockSparseMask_,
                GM_ADDR blockShape_, GM_ADDR attentionMask_, GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
                GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR workspace_, GM_ADDR tiling_data_
            ) : dout(dout_), q(q_), k(k_), v(v_), out(out_), softmaxLse(softmaxLse_), blockSparseMask(blockSparseMask_),
                blockShape(blockShape_), attentionMask(attentionMask_), actualQseqlen_(actualQseqlen), actualKvseqlen(actualKvseqlen_),
                dq(dq_), dk(dk_), dv(dv_), workspace(workspace_), tiling_data(tiling_data_)
            {
            }    
        };

        // Methods
        CATLASS_DEVICE
        BlockSparseAttentionGradKernel() {}

        template <int32_t CORE_TYPE = g_coreType>
        CATLASS_DEVICE
        void operator()(Params const &params);

        template <>
        CATLASS_DEVICE
        void operator()<AscendC::AIC>(Params const &params)
        {
        }

        template <>
        CATLASS_DEVICE
        void operator()<AscendC::AIV>(Params const &params)
        {
        }

    private:
        NpuArch::Arch::Resource<ArchTag> resource;
        // NpuArch::Arch::CrossCoreFlag qkReady{QK_READY_ID};
        // NpuArch::Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        // NpuArch::Arch::CrossCoreFlag pvReady{PV_READY_ID};
    };

} // namespace BSA

#endif // BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

