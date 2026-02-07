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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_mte.h
 * \brief qbmm_reduce_scatter_add_rms_norm_cast mte通信kernel代码逻辑
 */

#ifndef QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H
#define QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "kernel_operator.h"



namespace QbmmReduceScatterAddRmsNormCastImpl {

using namespace AscendC;
#define TemplateTypeClass typename X1Type, typename X2Type, typename Y1Type
#define TemplateType X1Type, X2Type, Y1Type

template<TemplateTypeClass>
class QbmmReduceScatterAddRmsNormCastMte {
public:
    __aicore__ inline QbmmReduceScatterAddRmsNormCastMte() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, 
                                GM_ADDR scale, GM_ADDR bias,  GM_ADDR perTokenScale,
                                GM_ADDR y1, GM_ADDR y2, GM_ADDR x, TPipe *tPipe, 
                                const QbmmReduceScatterAddRmsNormCastTilingData *tilingData);
    __aicore__ inline void Process();
};

template <TemplateTypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateType>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, 
                                                                 GM_ADDR scale, GM_ADDR bias,  GM_ADDR perTokenScale,
                                                                 GM_ADDR y1, GM_ADDR y2, GM_ADDR x, TPipe *tPipe, 
                                                                 const QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    AscendC::printf("OK");
}

template <TemplateTypeClass>
__aicore__ inline void QbmmReduceScatterAddRmsNormCastMte<TemplateType>::Process()
{
    AscendC::printf("OK");
    return;
}
} // QbmmReduceScatterAddRmsNormCastImpl
#endif  // QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_MTE_H