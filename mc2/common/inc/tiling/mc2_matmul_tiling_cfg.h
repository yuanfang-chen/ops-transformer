/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file mc2_matmul_tiling_cfg.h
 * \brief
 */

#ifndef __MC2_MATMUL_TILING_CFG_H__
#define __MC2_MATMUL_TILING_CFG_H__

#pragma once
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_data.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MC2MatmulV3TilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, mTailCnt);
  TILING_DATA_FIELD_DEF(uint32_t, nTailCnt);
  TILING_DATA_FIELD_DEF(uint32_t, kTailCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mBaseTailSplitCnt);
  TILING_DATA_FIELD_DEF(uint32_t, nBaseTailSplitCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mTailMain);
  TILING_DATA_FIELD_DEF(uint32_t, nTailMain);
  TILING_DATA_FIELD_DEF(uint32_t, isHf32);
  TILING_DATA_FIELD_DEF(uint32_t, aswWindowLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2MMV3TilingDataOp, MC2MatmulV3TilingData);
}  // namespace optiling

namespace Mc2MatmulHelper {
class Mc2MatmulTilingCfg : public optiling::Mc2MatMulTilingCfg
{
public:
    Mc2MatmulTilingCfg(const void* compileInfoIn, const void* argsIn, uint32_t baseMLimit = 0, bool needUpdateIn = true)
        : Mc2MatMulTilingCfg(needUpdateIn, compileInfoIn, argsIn), baseMLimit_(baseMLimit)
    {
    }

    void SetMatMulV3TilingData(optiling::MC2MatmulV3TilingData& tilingData);
    void Update(const optiling::Mc2TilingResult& result) override;
    void SetRankDim(uint64_t rankDim);
    void SetCommCnt(uint64_t commCnt);

private:
    void SetMMTilingData() const;
    void SetTailCntAndType() const;
    void DealBaseBlock() const;

    int32_t baseMLimit_{0};
    uint64_t rankDim_{0};
    uint64_t commCnt_{0};
    optiling::MC2MatmulV3TilingData* mc2MmV3TilingData_{nullptr};
    Mc2MatMulV3TilingData* mmv3TilingData_{nullptr};
};

}  // namespace Mc2MatmulHelper

#endif
