/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_general_regbase.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_REGBASE_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_REGBASE_H

#include <numeric>
#include <algorithm>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_type.h"
#include "../../op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_template_tiling_key_regbase.h"
#include "../../op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_tiling_regbase.h"
#include "../sparse_lightning_indexer_grad_kl_loss_tiling_basic.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

class SparseLightningIndexerGradKLLossTilingBaseRegbase : public SparseLightningIndexerGradKLLossTilingGeneral {
public:
    explicit SparseLightningIndexerGradKLLossTilingBaseRegbase(gert::TilingContext *context) : SparseLightningIndexerGradKLLossTilingGeneral(context)
    {
        this->templateName = "sligklloss_regbase";
    }
    ~SparseLightningIndexerGradKLLossTilingBaseRegbase() override = default;

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

    SparseLightningIndexerGradKLLossRegBaseTilingData *tilingData = context_->GetTilingData<SparseLightningIndexerGradKLLossRegBaseTilingData>();
    SLIGradKLLossBaseParamsRegbase *sliGradkllossBaseParams_ = &tilingData->baseParams;
    SLIGradKLLossMultiCoreParamsRegbase *sliGradkllossMultiCoreParams_ = &tilingData->multiCoreParams;
    SLIGradKLLossInitOutputParamsRegbase *initoutput =  &tilingData->initOutputParams;
    SLIGradKLLossWorkSpaceOffsetParamsRegbase *sliGradkllossWorkSpaceOffsetParams_ = &tilingData->workSpaceOffsetParams;
};

} // optiling
#endif