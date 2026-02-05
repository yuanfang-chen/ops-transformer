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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_helper.h
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
// #include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_key.h"
#include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"
#include "add_rms_norm_dynamic_quant_v2.h"
#include "tiling/platform/platform_ascendc.h"

namespace MC2Tiling {
// 如何直接使用matmul高阶api进行拷贝
struct Mc2MatMulArgs {
    const char *opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool hasBias = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType x3Type = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint64_t mValue = 0UL;
    uint64_t mOriValue = 0UL;
    uint64_t nOriValue = 0UL;
    uint64_t kValue = 0UL;
    uint64_t nValue = 0UL;
    uint64_t aDtypeSize = 1UL;
    uint64_t bDtypeSize = 1UL;
    uint64_t fusedOpType = 0UL;
    uint64_t batchX3 = 1UL;
    bool hasX3Input = false;
    MatMulV3BatchInfo *batchInfo = nullptr;
};

struct Mc2MatmulCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t btSize{0UL};
    float cubeFreq{0};
    platform_ascendc::SocVersion socVersion;
    std::string socVersionStr = "";
    bool supportL0c2out = false;
    bool supportL12BtBf16 = false;
};

class MmTilingHelper
{
public:
    MmTilingHelper(gert::TilingContext* context):context_(context){}
    ~MmTilingHelper() = default;
    ge::graphStatus InitTCubeTilingData(TCubeTiling &tCubeTiling) const;
    ge::graphStatus getMamtulArgs();
    void InitCompileInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay,
                                   uint64_t dtypeSize, ge::Format matFormat) const
private:
    ge::graphStatus GetMoreArgs();
private:
    gert::TilingContext* context_;
    const Mc2MatMulArgs args_;
    const Mc2MatmulCompileInfo compileInfo_;
    matmul_tiling::MultiCoreMatmulTiling mm_;
};
} // namespace MC2Tiling
