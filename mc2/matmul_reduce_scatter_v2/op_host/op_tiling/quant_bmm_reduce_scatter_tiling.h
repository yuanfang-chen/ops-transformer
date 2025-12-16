/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_bmm_reduce_scatter_tiling.h
 * \brief
 */
#ifndef __QUANT_BMM_REDUCE_SCATTER_TILING_H__
#define __QUANT_BMM_REDUCE_SCATTER_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_c_tiling.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "matmul_reduce_scatter_tiling_base.h"
#include "tiling/mc2_tiling_utils.h"
#include "kernel/mc2_tiling_struct.h"

namespace optiling {

class QuantBmmReduceScatterTiling : public MatmulReduceScatterTilingBase {
public:
    explicit QuantBmmReduceScatterTiling(gert::TilingContext *context);
    ~QuantBmmReduceScatterTiling() override = default;
    gert::TilingContext *GetContext()
    {
        return context_;
    }
    mc2tiling::TilingArgs &GetArgs()
    {
        return args_;
    }
    const char *GetOpName()
    {
        return opName_;
    }
    uint64_t GetMyWorkSpaceSize()
    {
        return myWorkSpaceSize_;
    }
    uint64_t SetMyWorkSpaceSize(uint64_t workspaceSize)
    {
        return myWorkSpaceSize_ = workspaceSize;
    }
    uint64_t GetLibApiWorkSpaceSize()
    {
        return libApiWorkSpaceSize_;
    }
    mc2tiling::Mc2QuantMode GetQuantMode()
    {
        return quantMode_ ;
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ::TCubeTiling &MutableTCubeTileTilingData();
    DequantBmm::Mc2QuantBatchMatmulV3DataParams &MutableTCubeTilingParam();
    DequantBmm::Mc2L2cacheTileParams &MutableTCubeTilingL2cache();
    DequantBmm::Mc2SlidingWindowParams &MutableTCubeTilingSlidingWindow();

    ::TCubeTiling &MutableTCubeTailTilingData();
    DequantBmm::Mc2QuantBatchMatmulV3DataParams &MutableTailTCubeTilingParam();
    DequantBmm::Mc2L2cacheTileParams &MutableTailTCubeTilingL2cache();
    DequantBmm::Mc2SlidingWindowParams &MutableTailTCubeTilingSlidingWindow();

    Mc2Tiling::Mc2Msg &MutableMc2MsgDataA5();
    Mc2Tiling::RCSTiling &MutableRCSTilingDataA5();

    ge::graphStatus DoAdaptSlidWindowTiling();
    void SetMc2Hcomm();
    ge::graphStatus CheckInput() override;
    bool CommonParamCheck();
    bool PerblockSceneParamCheck(const gert::StorageShape *x1ScaleShape, const gert::StorageShape *x2ScaleShape);
    bool PertensorSceneParamCheck(const gert::StorageShape *x1ScaleShape, const gert::StorageShape *x2ScaleShape);
    bool MxfpSceneParamCheck(const gert::StorageShape *x1ScaleShape, const gert::StorageShape *x2ScaleShape);
    void SetScene();
    bool CheckPerblockM();
    ge::graphStatus CheckGroupSize();
    ge::graphStatus CheckScale();
    ge::graphStatus CheckMxScaleDim(const gert::StorageShape *x1ScaleShape, const gert::StorageShape *x2ScaleShape);

private:
    Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData quantBmmMatmulReducescatterTilingDataSelf_;
    Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData* quantBmmMatmulReducescatterTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    mc2tiling::Mc2QuantMode quantMode_{mc2tiling::Mc2QuantMode::DEFAULT};
};

class QuantBmmReduceScatterHelper : public Mc2AdaptiveSlidingWindowTiling {
public:
    QuantBmmReduceScatterHelper(QuantBmmReduceScatterTiling &quantBmmReduceScatterTiling,
                                DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &out);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    const gert::StorageShape *GetOffsetShape(const size_t index);
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    QuantBmmReduceScatterTiling &tilingProcesser_;
    mc2tiling::TilingArgs &tilingArgs_;
};
} // namespace optiling


#endif //__QUANT_BMM_REDUCE_SCATTER_TILING_H__