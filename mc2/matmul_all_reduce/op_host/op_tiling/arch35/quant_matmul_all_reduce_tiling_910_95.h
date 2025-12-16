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
 * \file quant_matmul_all_reduce_tiling_910_95.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H

#include "../matmul_all_reduce_tiling_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QuantMatmulAllReduceTilingDataA5)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommInt8Cfg);
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(Mc2QuantBatchMatmulV3TilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(Mc2QuantBatchMatmulV3TilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000000000, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000000001, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000002000, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000002001, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000000010, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000000011, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000002010, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000002011, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000004000, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_1000000000000004001, QuantMatmulAllReduceTilingDataA5);
REGISTER_TILING_DATA_CLASS(QuantMatmulAllReduceTilingDataOp, QuantMatmulAllReduceTilingDataA5);

class QuantMatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase
{
    friend class QuantTilingTransferHelperA5;

public:
    explicit QuantMatmulAllReduceTilingA5(gert::TilingContext* context);
    QuantMatmulAllReduceTilingA5(
        gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, QuantMatmulAllReduceTilingDataA5* out);
    ~QuantMatmulAllReduceTilingA5() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg& MutableMc2MsgData() override;

    RCSTiling& MutableRCSTilingData() override;

    TCubeTiling& MutableTCubeTileTilingData() override;

    TCubeTiling& MutableTCubeTailTilingData() override;

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
    QuantMatmulAllReduceTilingDataA5 quantMatmulAllReduceTilingDataSelf_;
    QuantMatmulAllReduceTilingDataA5& quantMatmulAllReduceTilingData_;
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