/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mx_quant_matmul.h
 * \brief
 */

#ifndef MC2_MX_QUANT_MATMUL_H
#define MC2_MX_QUANT_MATMUL_H

#include "base_quant_matmul.h"

namespace MC2KernelTemplate {

// QuantExtraData 结构体保持不变，定义该模式需要的字段
struct MXQuantExtraData {
    uint64_t a_offset;
    uint64_t b_offset;
    uint64_t c_offset;
    uint64_t x1_scale_offset;
    GM_ADDR x1_scale;
    GM_ADDR x2_scale;
};

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
class MXQuantMatmul : public BaseQuantMatmul<MXQuantMatmul<MMKernel, ExtraDataType, TilingDataType>, MMKernel,
                                             ExtraDataType, TilingDataType> {
    using Base = BaseQuantMatmul<MXQuantMatmul<MMKernel, ExtraDataType, TilingDataType>, MMKernel, ExtraDataType,
                                 TilingDataType>;
    friend Base;

public:
    __aicore__ inline MXQuantMatmul(AscendC::TPipe *tPipe);

protected:
    /**
     * @brief 单独的Init，传递的参数需要特殊定义
     *
     * @return __aicore__
     */
    __aicore__ inline void Init();

    __aicore__ inline void ShiftQuantParams();

    __aicore__ inline void UpdateQuantParams(ExtraDataType *extraData);
};

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline MXQuantMatmul<MMKernel, ExtraDataType, TilingDataType>::MXQuantMatmul(AscendC::TPipe *tPipe)
    : Base(tPipe)
{
}

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void MXQuantMatmul<MMKernel, ExtraDataType, TilingDataType>::Init()
{
    this->tPipe_->Reset();
    this->mmOp_.Init(this->baseAddrs_.aGM, this->baseAddrs_.bGM, this->baseAddrs_.biasGM, this->extraData_.x2_scale,
                     this->extraData_.x1_scale, this->baseAddrs_.cGM, nullptr, this->tilingData_, this->tPipe_);
}

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void MXQuantMatmul<MMKernel, ExtraDataType, TilingDataType>::ShiftQuantParams()
{
    // MXQuant 模式下特有的 x1_scale 更新逻辑
    this->extraData_.x1_scale = (GM_ADDR)((uint64_t)this->extraData_.x1_scale + this->extraData_.x1_scale_offset);
}

template <typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void
MXQuantMatmul<MMKernel, ExtraDataType, TilingDataType>::UpdateQuantParams(ExtraDataType *extraData)
{
    this->extraData_.x1_scale_offset = extraData->x1_scale_offset;
    this->extraData_.x1_scale = extraData->x1_scale;
    this->extraData_.x2_scale = extraData->x2_scale;
}
} // namespace MC2KernelTemplate
#endif
