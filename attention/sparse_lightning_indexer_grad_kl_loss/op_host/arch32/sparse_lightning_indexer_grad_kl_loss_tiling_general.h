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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_general.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_H

#include <numeric>
#include <algorithm>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_type.h"
#include "../sparse_lightning_indexer_grad_kl_loss_tiling_common.h"
#include "../../op_kernel/arch32/sparse_lightning_indexer_grad_kl_loss_template_tiling_key.h"
#include "../../op_kernel/arch32/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "../sparse_lightning_indexer_grad_kl_loss_tiling_basic.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

class SparseLightningIndexerGradKLLossTilingBase : public SparseLightningIndexerGradKLLossTilingGeneral {
public:
    explicit SparseLightningIndexerGradKLLossTilingBase(gert::TilingContext *context) : SparseLightningIndexerGradKLLossTilingGeneral(context)
    {
        this->templateName = "sligklloss_base";
    }
    ~SparseLightningIndexerGradKLLossTilingBase() override = default;

protected:
    // 获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    
    bool CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShap);
    bool AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &queryIndexShape,
                         const gert::Shape &topKShape, size_t layoutLen, const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape);
    bool AnalyzeLayout();
    void SetSparseParamsRegbase();
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);

    TopKRange topkSize;
    SparseLightningIndexerGradKLLossTilingData *tilingData = context_->GetTilingData<SparseLightningIndexerGradKLLossTilingData>();
    SLIGradKLLossBaseParams *sliGradkllossBaseParams_ = &tilingData->baseParams;
    SLIGradKLLossInitOutputParams *initoutput =  &tilingData->initOutputParams;
    SLIGradKLLossMultiCoreParams *sliGradkllossMultiCoreParams_ = &tilingData->multiCoreParams;
};

} // optiling
#endif