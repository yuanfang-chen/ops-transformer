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
 * \file flash_attention_tiling_basic.cpp
 * \brief FlashAttention arch35 基础切分策略（非量化，非PA场景，框架桩）
 *
 * 参照flash_attention_score/op_host/arch35/flash_attention_score_tiling_basic.cpp，
 * 实现基础的S1S2Const切分策略。具体tiling计算逻辑待补充。
 */

#include "flash_attention_tiling_regbase.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FA {

class FlashAttentionTilingBasic : public FlashAttentionTilingRegbase {
public:
    explicit FlashAttentionTilingBasic(gert::TilingContext *context)
        : FlashAttentionTilingRegbase(context)
    {
        this->templateName = "S1S2Const";
        this->regbase = true;
    }
    ~FlashAttentionTilingBasic() override = default;

protected:
    bool IsCapable() override
    {
        if (npuArch != NpuArch::DAV_3510) {
            OP_LOGD(opName, "FlashAttentionTilingBasic: current npu arch is not dav-3510, skip.");
            return false;
        }
        return true;
    }

    void CalcDBasicBlock() override
    {
        // 按64对齐选择dBasicBlock，D > 256时用768，其余按64档位对齐
        dBasicBlock = FA_AlignUp(dSize, FA_D_TEMPLATE_SPLIT);
        if (dBasicBlock > FA_NUM_256) {
            dTemplateType = FADTemplateType::ALIGNED_768;
            dBasicBlock   = FA_NUM_768;
            return;
        }
        switch (dBasicBlock) {
            case FA_NUM_64:  dTemplateType = FADTemplateType::ALIGNED_64;  break;
            case FA_NUM_128: dTemplateType = FADTemplateType::ALIGNED_128; break;
            case FA_NUM_192: dTemplateType = FADTemplateType::ALIGNED_192; break;
            case FA_NUM_256: dTemplateType = FADTemplateType::ALIGNED_256; break;
            default:
                dTemplateType = FADTemplateType::BOTTOM;
                OPS_REPORT_VECTOR_INNER_ERR(opName, "dSize(%ld) not in range (0, 768].", dSize);
                break;
        }
    }

    void CalcS1S2BasicBlock() override
    {
        // TODO: 根据D大小和场景（isPA等）选择S1/S2基本块
        // 参照flash_attention_score_tiling_basic.cpp中的CalcS1S2BasicBlock实现
        if (dSize > FA_NUM_256) {
            s1BasicBlock = FA_NUM_128;
            s2BasicBlock = FA_NUM_128;
        } else {
            s1BasicBlock = FA_NUM_128;
            s2BasicBlock = FA_NUM_128;
        }
    }

    int64_t CalcTotalSize() override
    {
        int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        return totalSize;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        // TODO: 当D较大时需要workspace存放BMM2的中间结果
        // 参照flash_attention_score_tiling_basic.cpp中的GetWorkspaceSize实现
        int64_t bmm2Bytes = 0LL;
        if (dSize > FA_MIN_D_WORKSPACE) {
            bmm2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
        }
        bmm2Bytes = FA_AlignUp(bmm2Bytes, FA_GM_ALIGN);
        workspaces[0] = static_cast<size_t>(bmm2Bytes * FA_PING_PONG * multiCoreParamsRegbase_->get_coreNum());
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        uint8_t layout    = static_cast<uint8_t>(tilingKeyLayout);
        uint16_t s1Type   = static_cast<uint16_t>(s1BasicBlock);
        uint16_t s2Type   = static_cast<uint16_t>(s2BasicBlock);
        uint16_t dType    = static_cast<uint16_t>(dTemplateType);
        uint16_t dvType   = static_cast<uint16_t>(dVTemplateType);
        uint8_t atten     = static_cast<uint8_t>(hasAttenMask ? 1 : 0);
        uint8_t paFlag    = static_cast<uint8_t>(isPA ? 1 : 0);
        uint8_t lseFlag   = static_cast<uint8_t>(returnSoftmaxLse != 0 ? 1 : 0);
        uint8_t impl      = static_cast<uint8_t>(implMode);

        if (dType == static_cast<uint16_t>(dVType)) {
            return GET_TPL_TILING_KEY(0, impl, layout, s1Type, s2Type, dType,
                                      static_cast<uint16_t>(FADTemplateType::NONALIGNED),
                                      atten, paFlag, lseFlag, 1);
        }
        return GET_TPL_TILING_KEY(0, impl, layout, s1Type, s2Type, dType, dvType,
                                  atten, paFlag, lseFlag, 1);
    }

    ge::graphStatus PostTiling() override
    {
        FlashAttentionTilingRegbase::PostTiling();
        return ge::GRAPH_SUCCESS;
    }

private:
    uint16_t dVType = static_cast<uint16_t>(FADTemplateType::NONALIGNED);
    bool regbase = true;
};

// 注册到arch35 (DAV_3510)，优先级83（与flash_attention_score保持一致）
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttention, FlashAttentionTilingBasic,
                                    static_cast<int32_t>(NpuArch::DAV_3510), 83);

} // namespace FA
} // namespace optiling
