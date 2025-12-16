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
 * \file weight_quant_matmul_all_reduce_tiling_910_95.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H
#include "../matmul_all_reduce_tiling_base.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.h"

namespace optiling {
using Mc2weight_quant_batch_matmul_v2::Mc2WeightQuantBatchMatmulV2ASTilingData;
using Mc2weight_quant_batch_matmul_v2::Mc2WeightQuantBatchMatmulV2TilingAS;

BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceA5TilingData)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfg);
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(Mc2WeightQuantBatchMatmulV2RegBaseTilingData, tileRegBaseMmTiling);
TILING_DATA_FIELD_DEF_STRUCT(Mc2WeightQuantBatchMatmulV2RegBaseTilingData, tailRegBaseMmTiling);
END_TILING_DATA_DEF;
// weight int8
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_100200, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_101200, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_100210, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_101210, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_100300, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_100310, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_101300, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_101310, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_100100, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_100110, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_101100, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_101110, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000000008, WeightQuantMatmulAllReduceA5TilingData);
REGISTER_TILING_DATA_CLASS(WeightQuantMatmulAllReduceA5TilingDataOp, WeightQuantMatmulAllReduceA5TilingData);

BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceA5Fp8TilingData)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfg);
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(Mc2WeightQuantBatchMatmulV2ASTilingData, tileMmASTiling);
TILING_DATA_FIELD_DEF_STRUCT(Mc2WeightQuantBatchMatmulV2ASTilingData, tailMmASTiling);
END_TILING_DATA_DEF;
// weight fp8/hif8
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000012100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000012120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000002100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000002120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020000000012100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020001000012100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020002000012100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020003000012100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020000000012120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020001000012120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020002000012120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020003000012120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000012140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000002140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000012160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000002160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020000000012140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020001000012140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020002000012140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020003000012140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020000000012160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020001000012160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020002000012160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000020003000012160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000011100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000011120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000001100, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000001120, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000001140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030003000001160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000011140, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_2000030004000011160, WeightQuantMatmulAllReduceA5Fp8TilingData);
REGISTER_TILING_DATA_CLASS(WeightQuantMatmulAllReduceA5Fp8TilingDataOp, WeightQuantMatmulAllReduceA5Fp8TilingData);

class WeightQuantMatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase
{
    friend class WeightQuantTilingTransferHelperA5;
    friend class WeightQuantAsTilingTransferHelper;

public:
    explicit WeightQuantMatmulAllReduceTilingA5(gert::TilingContext* context);
    ~WeightQuantMatmulAllReduceTilingA5() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg& MutableMc2MsgData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.msg;
        }
        return weightQuantMatmulAllReduceA5TilingData_.msg;
    }

    RCSTiling& MutableRCSTilingData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.param;
        }
        return weightQuantMatmulAllReduceA5TilingData_.param;
    }

    TCubeTiling& MutableTCubeTileTilingData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.tileMmASTiling.matmulTiling;
        }
        return weightQuantMatmulAllReduceA5TilingData_.tileRegBaseMmTiling.matmulTiling;
    }

    TCubeTiling& MutableTCubeTailTilingData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.tailMmASTiling.matmulTiling;
        }
        return weightQuantMatmulAllReduceA5TilingData_.tailRegBaseMmTiling.matmulTiling;
    }

    void PrintExtendMatmulTiling(bool isTail) override;
    void PrintMatmulAsTiling(bool isTail);

    ge::graphStatus DoWeightQuantTiling();
    ge::graphStatus DoWeightQuantAsTiling();

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

    void SetMc2Hcomm();

private:
    ge::graphStatus CheckBiasInput();
    ge::graphStatus CheckAxisSize();
    WeightQuantMatmulAllReduceA5TilingData weightQuantMatmulAllReduceA5TilingDataSelf_;
    WeightQuantMatmulAllReduceA5Fp8TilingData weightQuantMatmulAllReduceA5Fp8TilingDataSelf_;

    WeightQuantMatmulAllReduceA5TilingData& weightQuantMatmulAllReduceA5TilingData_;
    WeightQuantMatmulAllReduceA5Fp8TilingData& weightQuantMatmulAllReduceA5Fp8TilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isWeightFp8Hif8_{false};
};

class WeightQuantTilingTransferHelperA5 : public Mc2WeightQuantBatchMatmulV2RegBase
{
public:
    WeightQuantTilingTransferHelperA5(
        WeightQuantMatmulAllReduceTilingA5& weightQuantMatmulAllReduceTiling,
        Mc2WeightQuantBatchMatmulV2RegBaseTilingData& data)
        : Mc2WeightQuantBatchMatmulV2RegBase(weightQuantMatmulAllReduceTiling.context_),
          tilingProcesser_(weightQuantMatmulAllReduceTiling),
          data_(data)
    {}
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus MatmulDoTiling()
    {
        if (DoTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        data_.set_cubeBlockDimN(tilingData_->get_cubeBlockDimN());
        data_.set_cubeBlockDimM(tilingData_->get_cubeBlockDimM());
        data_.set_reserve1(tilingData_->get_reserve1());
        data_.set_vecCoreParallel(tilingData_->get_vecCoreParallel());
        data_.set_AL1Pingpong(tilingData_->get_AL1Pingpong());
        data_.set_BL1Pingpong(tilingData_->get_BL1Pingpong());
        data_.set_kSize(tilingData_->get_kSize());
        data_.set_nSize(tilingData_->get_nSize());
        data_.set_groupSize(tilingData_->get_groupSize());
        data_.set_mSize(tilingData_->get_mSize());
        data_.set_nBubSize(tilingData_->get_nBubSize());
        data_.set_kBubSize(tilingData_->get_kBubSize());

        data_.matmulTiling.set_M(tilingData_->matmulTiling.get_M());
        data_.matmulTiling.set_Ka(tilingData_->matmulTiling.get_Ka());
        data_.matmulTiling.set_N(tilingData_->matmulTiling.get_N());
        data_.matmulTiling.set_Kb(tilingData_->matmulTiling.get_Kb());
        data_.matmulTiling.set_singleCoreM(tilingData_->matmulTiling.get_singleCoreM());
        data_.matmulTiling.set_singleCoreN(tilingData_->matmulTiling.get_singleCoreN());
        data_.matmulTiling.set_singleCoreK(tilingData_->matmulTiling.get_singleCoreK());
        data_.matmulTiling.set_baseM(tilingData_->matmulTiling.get_baseM());
        data_.matmulTiling.set_baseN(tilingData_->matmulTiling.get_baseN());
        data_.matmulTiling.set_baseK(tilingData_->matmulTiling.get_baseK());
        data_.matmulTiling.set_dbL0A(tilingData_->matmulTiling.get_dbL0A());
        data_.matmulTiling.set_dbL0B(tilingData_->matmulTiling.get_dbL0B());
        data_.matmulTiling.set_dbL0C(tilingData_->matmulTiling.get_dbL0C());
        data_.matmulTiling.set_stepM(tilingData_->matmulTiling.get_stepM());
        data_.matmulTiling.set_stepN(tilingData_->matmulTiling.get_stepN());
        data_.matmulTiling.set_stepKa(tilingData_->matmulTiling.get_stepKa());
        data_.matmulTiling.set_stepKb(tilingData_->matmulTiling.get_stepKb());
        data_.matmulTiling.set_depthA1(tilingData_->matmulTiling.get_depthA1());
        data_.matmulTiling.set_depthB1(tilingData_->matmulTiling.get_depthB1());
        data_.matmulTiling.set_iterateOrder(tilingData_->matmulTiling.get_iterateOrder());
        data_.matmulTiling.set_isBias(tilingData_->matmulTiling.get_isBias());
        data_.matmulTiling.set_shareMode(tilingData_->matmulTiling.get_shareMode());
        data_.matmulTiling.set_shareL1Size(tilingData_->matmulTiling.get_shareMode());
        data_.matmulTiling.set_shareL0CSize(tilingData_->matmulTiling.get_shareL0CSize());
        return ge::GRAPH_SUCCESS;
    }

private:
    WeightQuantMatmulAllReduceTilingA5& tilingProcesser_;
    Mc2WeightQuantBatchMatmulV2RegBaseTilingData& data_;
};

class WeightQuantAsTilingTransferHelper : public Mc2WeightQuantBatchMatmulV2TilingAS
{
public:
    WeightQuantAsTilingTransferHelper(
        WeightQuantMatmulAllReduceTilingA5& weightQuantMatmulAllReduceTiling,
        Mc2WeightQuantBatchMatmulV2ASTilingData& data)
        : Mc2WeightQuantBatchMatmulV2TilingAS(weightQuantMatmulAllReduceTiling.context_),
          tilingProcesser_(weightQuantMatmulAllReduceTiling),
          mmASTilingdata_(data)
    {}
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    void PrintTilingInputParam(std::unique_ptr<Mc2WeightQuantBatchMatmulInfo>& matmulInfo);
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus MatmulDoTiling()
    {
        if (DoTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        mmASTilingdata_.set_cubeBlockDimM(tilingData_->get_cubeBlockDimM());
        mmASTilingdata_.set_cubeBlockDimN(tilingData_->get_cubeBlockDimN());
        mmASTilingdata_.set_hasBias(tilingData_->get_hasBias());
        mmASTilingdata_.set_firstTailBlockCount(tilingData_->get_firstTailBlockCount());
        mmASTilingdata_.set_secondTailBlockCount(tilingData_->get_secondTailBlockCount());
        mmASTilingdata_.set_weightL2Cacheable(tilingData_->get_weightL2Cacheable());
        mmASTilingdata_.set_mainBlockL1Size(tilingData_->get_mainBlockL1Size());
        mmASTilingdata_.set_firstTailBlockL1Size(tilingData_->get_firstTailBlockL1Size());
        mmASTilingdata_.set_secondTailBlockL1Size(tilingData_->get_secondTailBlockL1Size());
        mmASTilingdata_.set_aPreloadSize(tilingData_->get_aPreloadSize());
        mmASTilingdata_.set_groupSize(tilingData_->get_groupSize());
        mmASTilingdata_.set_mainBlockCount(tilingData_->get_mainBlockCount());
        mmASTilingdata_.set_mSize(tilingData_->get_mSize());
        mmASTilingdata_.set_kSize(tilingData_->get_kSize());
        mmASTilingdata_.set_nSize(tilingData_->get_nSize());

        mmASTilingdata_.matmulTiling.set_usedCoreNum(tilingData_->matmulTiling.get_usedCoreNum());
        mmASTilingdata_.matmulTiling.set_M(tilingData_->matmulTiling.get_M());
        mmASTilingdata_.matmulTiling.set_Ka(tilingData_->matmulTiling.get_Ka());
        mmASTilingdata_.matmulTiling.set_N(tilingData_->matmulTiling.get_N());
        mmASTilingdata_.matmulTiling.set_Kb(tilingData_->matmulTiling.get_Kb());
        mmASTilingdata_.matmulTiling.set_singleCoreM(tilingData_->matmulTiling.get_singleCoreM());
        mmASTilingdata_.matmulTiling.set_singleCoreN(tilingData_->matmulTiling.get_singleCoreN());
        mmASTilingdata_.matmulTiling.set_singleCoreK(tilingData_->matmulTiling.get_singleCoreK());
        mmASTilingdata_.matmulTiling.set_baseM(tilingData_->matmulTiling.get_baseM());
        mmASTilingdata_.matmulTiling.set_baseN(tilingData_->matmulTiling.get_baseN());
        mmASTilingdata_.matmulTiling.set_baseK(tilingData_->matmulTiling.get_baseK());
        mmASTilingdata_.matmulTiling.set_dbL0A(tilingData_->matmulTiling.get_dbL0A());
        mmASTilingdata_.matmulTiling.set_dbL0B(tilingData_->matmulTiling.get_dbL0B());
        mmASTilingdata_.matmulTiling.set_dbL0C(tilingData_->matmulTiling.get_dbL0C());
        mmASTilingdata_.matmulTiling.set_stepM(tilingData_->matmulTiling.get_stepM());
        mmASTilingdata_.matmulTiling.set_stepN(tilingData_->matmulTiling.get_stepN());
        mmASTilingdata_.matmulTiling.set_stepKa(tilingData_->matmulTiling.get_stepKa());
        mmASTilingdata_.matmulTiling.set_stepKb(tilingData_->matmulTiling.get_stepKb());
        mmASTilingdata_.matmulTiling.set_depthA1(tilingData_->matmulTiling.get_depthA1());
        mmASTilingdata_.matmulTiling.set_depthB1(tilingData_->matmulTiling.get_depthB1());
        mmASTilingdata_.matmulTiling.set_iterateOrder(tilingData_->matmulTiling.get_iterateOrder());
        mmASTilingdata_.matmulTiling.set_isBias(tilingData_->matmulTiling.get_isBias());
        mmASTilingdata_.matmulTiling.set_shareMode(tilingData_->matmulTiling.get_shareMode());
        mmASTilingdata_.matmulTiling.set_shareL1Size(tilingData_->matmulTiling.get_shareMode());
        mmASTilingdata_.matmulTiling.set_shareL0CSize(tilingData_->matmulTiling.get_shareL0CSize());
        return ge::GRAPH_SUCCESS;
    }

private:
    WeightQuantMatmulAllReduceTilingA5& tilingProcesser_;
    Mc2WeightQuantBatchMatmulV2ASTilingData& mmASTilingdata_;
};

} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_910_95_H
