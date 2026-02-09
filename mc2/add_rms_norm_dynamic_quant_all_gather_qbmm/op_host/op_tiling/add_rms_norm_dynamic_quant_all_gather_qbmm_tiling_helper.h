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
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace MC2Tiling {
// 如何直接使用matmul高阶api进行拷贝
struct Mc2MatMulArgs {
    const char *opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool isForceGrpAccForFp32 = false;
    bool hasBias = false;
    bool nd2nzA = false;
    bool nd2nzB = false;
    bool isNzA = false;
    bool isNzB = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint8_t unAlignProcessType = 0;
    uint64_t mValue = 0L;
    uint64_t mOriValue = 0L;
    uint64_t nOriValue = 0L;
    uint64_t kValue = 0L;
    uint64_t nValue = 0L;
    double l2Ratio = 0;
    // MatMulV3BatchInfo *batchInfo = nullptr;
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

// 选择不同的计算Tiling的方法
enum class Mc2TilingCalcSelect : int32_t {
    ALL = 0,
    BASE = 1,
    SINGLE_CORE_SPLIT_K = 2,
    DETERMINISTIC_SPLIT_K = 3
};

class MmTilingHelper {
public:
    explicit MmTilingHelper(gert::TilingContext* context):context_(context) {}
    ~MmTilingHelper() = default;
    ge::graphStatus InitTCubeTilingData(TCubeTiling &tCubeTiling);
    ge::graphStatus getMamtulArgs();
    void InitCompileInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetMoreArgs();
    bool NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay,
                                   uint64_t dtypeSize, ge::Format matFormat) const;
private:
    gert::TilingContext* context_;
    Mc2MatMulArgs args_;
    Mc2MatmulCompileInfo compileInfo_;
    matmul_tiling::MultiCoreMatmulTiling mm_;
    Mc2TilingCalcSelect tilingSelect_ = Mc2TilingCalcSelect::ALL;
};
} // namespace MC2Tiling
