/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_tiling.h
 * \brief
 */
#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_INVERSE_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_INVERSE_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcPreTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, outFlag);
TILING_DATA_FIELD_DEF(uint32_t, hasGamma);
TILING_DATA_FIELD_DEF(uint32_t, chunkTSize);
TILING_DATA_FIELD_DEF(uint32_t, v1ChunkDSize);
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, nD);
TILING_DATA_FIELD_DEF(uint64_t, fusionSize);
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint64_t, D);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(float, scaleMean);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPre, MhcPreTilingData);

struct MhcPreCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
};

class MhcPreBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPreBaseTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~MhcPreBaseTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    ge::graphStatus GetInputShape();
    ge::graphStatus ParseBsndFormat(const gert::Tensor *xTensor);
    ge::graphStatus ParseTndFormat(const gert::Tensor *xTensor);
    ge::graphStatus ValidateAndSetTilingParams(const gert::Tensor *xTensor);
    ge::graphStatus InitPlatformMemory();
    ge::graphStatus ParseOutputFlags();
    ge::graphStatus ParseEpsAttributes();
    void PrintTilingData();
    ge::graphStatus ParseInputAndAttr();
    void FillTilingData();
    ge::graphStatus TilingProcess();


private:
    MhcPreTilingData tilingData_;
    uint32_t blockDim_; // AIC
    uint64_t totalLength_;
    uint64_t m_;
    uint64_t ubSize_;
    uint64_t l1Size_;
    uint64_t matM_;
    uint64_t matK_;
    uint64_t matN_;
    uint64_t nD_;
    uint64_t N_;
    uint64_t D_;
    float normEps_;
    float hcEps_;
    uint32_t outFlag_;
    uint32_t hasGamma_;
    uint32_t chunkTSize_;
    uint32_t v1ChunkDSize_;

protected:
    matmul_tiling::MultiCoreMatmulTiling mm_;
};

} // namespace optiling
#endif // __OP_HOST_CHUNK_GATED_DELTA_RULE_INVERSE_H__