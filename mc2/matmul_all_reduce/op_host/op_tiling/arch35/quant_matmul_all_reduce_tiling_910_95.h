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
 * \file quant_matmul_all_reduce_tiling_910_95.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H

#include "../matmul_all_reduce_tiling_base.h"
#include "../../../op_kernel/arch35/matmul_all_reduce_tiling_struct_ar35.h"

namespace optiling {
class QuantMatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase
{
    friend class QuantTilingTransferHelperA5;

public:
    explicit QuantMatmulAllReduceTilingA5(gert::TilingContext* context);
    QuantMatmulAllReduceTilingA5(
        gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, Mc2Tiling::QuantMatmulAllReduceTilingDataA5* out);
    ~QuantMatmulAllReduceTilingA5() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::Mc2Msg& MutableMc2MsgData() override;

    Mc2Tiling::RCSTiling& MutableRCSTilingData() override;

    ::TCubeTiling& MutableTCubeTileTilingData() override;

    ::TCubeTiling& MutableTCubeTailTilingData() override;

    void PrintExtendMatmulTiling(bool isTail) override;

    ge::graphStatus DoQuantTiling();

    void SetMc2Hcomm();

    ge::graphStatus CheckInput() override;

    ge::graphStatus CheckDequantScaleType();
    ge::graphStatus CheckCommQuantScale();
    ge::graphStatus CheckBias();
    ge::graphStatus CheckX1X2();
    ge::graphStatus CheckA8W8ScenarioScaleType();
    ge::graphStatus CheckMXFPScenarioScaleType();
    ge::graphStatus CheckQuantGroupSize();
    ge::graphStatus GetDynamicQuantTempBuffSize();

private:
    ge::graphStatus CheckAxisSize();
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 quantMatmulAllReduceTilingDataSelf_{};
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5& quantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isCommInt8Enable_ = false;
    bool isCommFp8Enable_ = false;
};

class QuantTilingTransferHelperA5 : public Mc2AdaptiveSlidingWindowTiling
{
public:
    QuantTilingTransferHelperA5(
        QuantMatmulAllReduceTilingA5& quantMatmulAllReduceTiling, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& data)
        : Mc2AdaptiveSlidingWindowTiling(quantMatmulAllReduceTiling.context_, &data),
          tilingProcesser_(quantMatmulAllReduceTiling)
    {}

    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape& GetScaleShape(const size_t index) override;
    const gert::StorageShape* GetOffsetShape(const size_t index); // matmulV3还未回合
    const gert::StorageShape* GetPertokenShape(const size_t index) override;
    const gert::StorageShape* GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    QuantMatmulAllReduceTilingA5& tilingProcesser_;
};

} // namespace optiling
#endif // QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H