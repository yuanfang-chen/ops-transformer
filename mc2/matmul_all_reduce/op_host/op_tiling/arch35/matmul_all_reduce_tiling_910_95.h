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
 * \file matmul_all_reduce_tiling_910_95.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_910_95_H
#define MATMUL_ALL_REDUCE_TILING_910_95_H

#include "../matmul_all_reduce_tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "new_mc2_matmul_tiling_cfg.h"

namespace optiling {
using namespace mc2_matmul_v3_advanced;
using namespace Mc2Tiling;

BEGIN_TILING_DATA_DEF(MatmulAllReduce910TilingDataA5)
    TILING_DATA_FIELD_DEF(uint32_t, version);
    TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
    TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
    TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfg);
    TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
    TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
    TILING_DATA_FIELD_DEF_STRUCT(MC2MatmulV3TilingData, mC2Mmv3TileTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(MC2MatmulV3TilingData, mC2Mmv3TailTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000000001, MatmulAllReduce910TilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000001100, MatmulAllReduce910TilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000000009, MatmulAllReduce910TilingDataA5);

class MatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase
{
public:
    explicit MatmulAllReduceTilingA5(gert::TilingContext* context);
    MatmulAllReduceTilingA5(gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, MatmulAllReduce910TilingDataA5* out);
    ~MatmulAllReduceTilingA5() override = default;

protected:
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::NewMc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     optiling::MC2MatmulV3TilingData &tilingData);
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus Do910Tiling();

    Mc2Msg& MutableMc2MsgData() override;

    RCSTiling& MutableRCSTilingData() override;

    TCubeTiling &MutableTCubeTileTilingData() override
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TileTilingData.matmulTiling;
    }

    TCubeTiling &MutableTCubeTailTilingData() override
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TailTilingData.matmulTiling;
    }

    inline optiling::MC2MatmulV3TilingData &MutableMC2MmV3TileTilingData()
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TileTilingData;
    }

    inline optiling::MC2MatmulV3TilingData &MutableMC2MmV3TailTilingData()
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    }

    void PrintExtendMatmulTiling(bool isTail) override;
    void DoEmptyTensorTiling() override;
    void SetMc2Hcomm();
    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus CheckAxisSize();
    ge::graphStatus CheckX1X2();
    MatmulAllReduce910TilingDataA5 matmulAllReduce910TilingDataSelf_;
    MatmulAllReduce910TilingDataA5& matmulAllReduce910TilingData_;
    uint64_t myWorkSpaceSize_{0U};
    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo compileInfo_;
};

} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_910_95_H