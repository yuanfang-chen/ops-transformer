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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_helper.cpp
 * \brief host侧tiling实现
 */

#include <string>
#include <register/op_def_registry.h>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_infos_def.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_helper.h"
#include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"

namespace MC2Tiling {

constexpr uint64_t SCALE_INDEX = 5;
constexpr uint64_t BIAS_INDEX = 7;
constexpr uint64_t ONE_BATCH_DIM = 1;
constexpr uint64_t TWO_BATCH_DIM = 2;
constexpr uint64_t THREE_BATCH_DIM = 3;
constexpr uint64_t FOUR_BATCH_DIM = 4;
constexpr uint64_t LAST_DIM = 1;
constexpr uint64_t LAST_SECOND_DIM = 2;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t ND2NZ_ON_THE_FLY_LIMIT = 65535;

const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP =
{
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
};

enum class MatmulV3Trans : int32_t
{
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

template<typename T>
inline bool Is256BAlign(T base, uint64_t dTypeSize) {
    if (base * dTypeSize % 256 == 0) { // 256: align byte size
        return true;
    }
    return false;
};

ge::graphStatus MmTilingHelper::GetPlatformInfo() // 检查平台信息是否支持
{
    
    auto compileInfoPtr = reinterpret_cast<const Mc2MatmulCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
    compileInfo_ = *compileInfoPtr;

    if (compileInfo_.aicNum == 0) {
        OP_LOGE(context_->GetNodeName(), "compileInfo.aicNum is zero.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MmTilingHelper::InitCompileInfo() // 检查输入属性是否支持
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGW(context_->GetNodeName(), "platformInfo is null");
        return;
    }
    Mc2MatmulCompileInfo compileInfo;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfo.socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfo.supportL0c2out = !val.empty();
    compileInfo.supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfo.aicNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    compileInfo.aivNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    compileInfo.socVersion = ascendcPlatform.GetSocVersion();
    compileInfo.btSize = compileInfo.supportL0c2out ? 1024UL : 0UL;                    // 1024 is btSize
    compileInfo.btSize = compileInfo.supportL12BtBf16 ? 4096 : compileInfo.btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo.l2Size);

    // TilingPrepareForOpCache(context_);
    OP_LOGI(context_->GetNodeName(),
        "parse compile info soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
        static_cast<int32_t>(compileInfo.socVersion), compileInfo.l1Size, compileInfo.l2Size, compileInfo.aicNum,
        compileInfo.supportL0c2out, compileInfo.supportL12BtBf16);
    // compileInfoInit_ = true;
    compileInfo_ = compileInfo;
}

ge::graphStatus MmTilingHelper::InitTCubeTilingData(TCubeTiling &tCubeTiling)
{
    auto aFormat = args_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto bFormat = args_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto cFormat = args_.outFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    mm_.SetAType(matmul_tiling::TPosition::GM, aFormat, DTYPE_MAP.at(args_.aType), args_.isATrans);
    mm_.SetBType(matmul_tiling::TPosition::GM, bFormat, DTYPE_MAP.at(args_.bType), args_.isBTrans);
    mm_.SetCType(matmul_tiling::TPosition::GM, cFormat, DTYPE_MAP.at(args_.cType));
    mm_.SetDim(compileInfo_.aicNum);
    mm_.SetShape(args_.mValue, args_.nValue, args_.kValue);
    mm_.SetOrgShape(args_.mValue, args_.nValue, args_.kValue);
    if (args_.hasBias) {
        mm_.SetBias(true);
        mm_.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(args_.biasType));
    }
    mm_.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0CSize, compileInfo_.ubSize);
    if (mm_.GetTiling(tCubeTiling) == -1) {
        OP_LOGE(args_.opName, "MatmulV3 Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus GetInputDims(const gert::Shape &storageShape, ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - ONE_BATCH_DIM];
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus SetMatmulDimensions(
    const gert::TilingContext& context, Mc2MatMulArgs& args, int64_t m, int64_t k, int64_t n)
{
    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(m) || !isValidDimValue(k) || !isValidDimValue(n)) {
        OP_LOGE(args.opName, "illegal value: m[%ld], k[%ld], n[%ld]", m, k, n);
        return ge::GRAPH_FAILED;
    }
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(k);
    args.nValue = static_cast<uint64_t>(n);

    // get origin (m, n)
    const gert::Shape& cShape = context.GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE(args.opName, "illegal value: output dim num (%zu)", cDimNum);
        return ge::GRAPH_FAILED;
    }
    args.nOriValue = cShape[cDimNum - LAST_DIM];
    args.mOriValue = cShape[cDimNum - LAST_SECOND_DIM];

    if (args.aFormat == ge::FORMAT_FRACTAL_NZ) {
        args.mValue = args.mOriValue;
    }

    if (args.bFormat == ge::FORMAT_FRACTAL_NZ) {
        args.nValue = args.nOriValue;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShape(const gert::TilingContext &context, Mc2MatMulArgs &args)
{
    // get transpose
    args.isBTrans = *context.GetAttrs()->GetAttrPointer<bool>(2);

    // get (m, k, n)
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];
    if ((GetInputDims(context.GetInputShape(0)->GetStorageShape(), args.aFormat, mkDims) != ge::GRAPH_SUCCESS) ||
        (GetInputDims(context.GetInputShape(1)->GetStorageShape(), args.bFormat, knDims) != ge::GRAPH_SUCCESS)) {
        OP_LOGE(args.opName, "invalid input dim num");
        return ge::GRAPH_FAILED;
    }
    int64_t kDimsIndex = args.isATrans ? 0 : 1;
    int64_t k = mkDims[kDimsIndex];
    int64_t kRight = 0;
    if (args.aFormat == ge::FORMAT_FRACTAL_NZ) {
        auto aOriginShape = context.GetInputShape(0)->GetOriginShape();
        int64_t aDimNum = aOriginShape.GetDimNum();
        k = aOriginShape.GetDim(args.isATrans ? aDimNum - TWO_BATCH_DIM : aDimNum - ONE_BATCH_DIM);
    }
    if (args.bFormat == ge::FORMAT_FRACTAL_NZ) {
        auto bOriginShape = context.GetInputShape(1)->GetOriginShape();
        int64_t bDimNum = bOriginShape.GetDimNum();
        kRight = bOriginShape.GetDim(args.isBTrans ? bDimNum - ONE_BATCH_DIM: bDimNum - TWO_BATCH_DIM);
    } else {
        int64_t dimIndex = args.isBTrans ? 1 : 0;
        kRight = knDims[dimIndex];
    }
    if (k != kRight) {
        OP_LOGE(args.opName, "unequal input kDim values: k_left[%ld], k_right[%ld]", k, kRight);
        return ge::GRAPH_FAILED;
    }
    int64_t mDimsIndex = args.isATrans ? 1 : 0;
    int64_t m = mkDims[mDimsIndex];
    int64_t nDimsIndex = args.isBTrans ? 0 : 1;
    int64_t n = knDims[nDimsIndex];

    return SetMatmulDimensions(context, args, m, k, n);
}

static inline void GetFormat(const gert::TilingContext &context, Mc2MatMulArgs &args)
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

static inline void GetDtype(const gert::TilingContext &context, Mc2MatMulArgs &args)
{
    OP_LOGD(args.opName, "Hf32 flag is: %d", args.isHf32);

    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    args.hasBias = context.GetOptionalInputDesc(BIAS_INDEX) != nullptr;
    OP_LOGD(args.opName, "hasBias is: %d", static_cast<int>(args.hasBias));
    if (args.hasBias) {
        args.biasType = context.GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    }
}

ge::graphStatus MmTilingHelper::getMamtulArgs()
{
    GetFormat(*context_, args_);
    GetDtype(*context_, args_);
    if (GetShape(*context_, args_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline bool GetNd2nzA(const Mc2MatMulArgs& args_, const Mc2MatmulCompileInfo& compileInfo_)
{
    constexpr uint64_t nMataThread = 16384;
    constexpr uint64_t mMataThread = 4096;
    constexpr uint64_t kMataThread = 6656;
    constexpr uint64_t compileNum = 24;
    return !args_.isBTrans && args_.nValue % nMataThread == 0 &&
           (args_.mValue > mMataThread || (args_.mValue == mMataThread && compileInfo_.aicNum >= compileNum)) &&
           args_.kValue >= kMataThread && args_.bFormat == ge::FORMAT_ND &&
           (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16);
}

inline bool GetNd2nzB(const Mc2MatMulArgs& args_, const Mc2MatmulCompileInfo& compileInfo_)
{
    constexpr uint64_t kMataCond = 16384;
    constexpr uint64_t nMataCond = 7168;
    constexpr uint64_t mMataCondMax = 4480;
    constexpr uint64_t mMataCondMin = 4096;
    constexpr uint64_t compileNum = 24;
    return !args_.isATrans && args_.isBTrans && args_.kValue == kMataCond && args_.mValue >= mMataCondMin &&
           args_.mValue <= mMataCondMax && args_.nValue >= nMataCond && args_.bFormat == ge::FORMAT_ND &&
           (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) && compileInfo_.aicNum >= compileNum;
}

bool MmTilingHelper::NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay,
                                    uint64_t dtypeSize, ge::Format matFormat) const
{
    if (dtypeSize == 0UL) {
        return false;
    }
    if (matFormat == ge::FORMAT_ND) {
        bool innerAlign = Is256BAlign(innerSize, dtypeSize);
        // 外轴<=8192 会导致数据量增大减慢搬运, 192B为最大奇数内轴长度, 384B为最大偶数内轴长度
        bool willFitVnchwCond = outerSize > 8192UL && (innerSize > 1UL) && (innerSize * dtypeSize <= 192UL ||
                                (innerSize * dtypeSize <= 384UL && innerSize % 2UL == 0) ||
                                (innerSize * dtypeSize <= CACHELINE && innerSize % 4UL == 0UL));
        bool willInnerSizeEqualC0 = (innerSize == (32UL / dtypeSize));
        return (willFitVnchwCond && !innerAlign && !supportNd2NzOnTheWay && !willInnerSizeEqualC0);
    }
    return false;
}

ge::graphStatus MmTilingHelper::GetMoreArgs()
{
    uint64_t aDtypeSize = ge::GetSizeByDataType(args_.aType);
    uint64_t bDtypeSize = ge::GetSizeByDataType(args_.bType);
    uint64_t cDtypeSize = ge::GetSizeByDataType(args_.cType);
    uint64_t m256Align = Is256BAlign(args_.mValue, aDtypeSize); // A矩阵 m轴256B对齐
    uint64_t kA256Align = Is256BAlign(args_.kValue, aDtypeSize); // A矩阵 k轴256B对齐
    uint64_t kB256Align = Is256BAlign(args_.kValue, bDtypeSize); // B矩阵 k轴256B对齐
    uint64_t n256Align = Is256BAlign(args_.nValue, bDtypeSize);  // B矩阵 n轴256B对齐
    bool innerAlignA = kA256Align;
    bool innerAlignB = n256Align;
    uint64_t innerSizeA = args_.kValue;
    uint64_t innerSizeB = args_.nValue;
    uint64_t outerSizeA = args_.mValue;
    uint64_t outerSizeB = args_.kValue;
    auto trans = MatmulV3Trans::NO_TRANS;
    if (args_.isATrans) {
        trans = MatmulV3Trans::A_TRANS;
        innerAlignA = m256Align;
        innerSizeA = args_.mValue;
        outerSizeA = args_.kValue;
    }
    if (args_.isBTrans) {
        trans = MatmulV3Trans::B_TRANS;
        innerAlignB = kB256Align;
        innerSizeB = args_.kValue;
        outerSizeB = args_.nValue;
    }
    if (args_.isATrans && args_.isBTrans) {
        trans = MatmulV3Trans::AB_TRANS;
    }
    // uint64_t calcMBasic = compileInfo_.l0CSize == L0C_SIZE_256_KB ? CALC_MN_BASIC_L0C_256 : CALC_M_BASIC;
    // calcMNBasic_ = compileInfo_.l0CSize == L0C_SIZE_256_KB ? CALC_MN_BASIC_L0C_256 : CALC_MN_BASIC;
    // l2TileLength_ = L2_TILE_LENGTH;
    OP_TILING_CHECK(!compileInfo_.supportL0c2out, tilingSelect_ = Mc2TilingCalcSelect::BASE, return ge::GRAPH_SUCCESS);

    // check the size is equaled to {32, 64, 96, 128, 160, 192, 224, 256, 384}.
    bool supportNd2NzOnTheWayA = false;
    bool supportNd2NzOnTheWayB = false;
    args_.nd2nzA = ((!innerAlignA || innerSizeA > ND2NZ_ON_THE_FLY_LIMIT) && (args_.aFormat == ge::FORMAT_ND) &&
        (!supportNd2NzOnTheWayA) &&
        !(args_.aType == ge::DT_FLOAT && !args_.isHf32 && innerSizeA < ND2NZ_ON_THE_FLY_LIMIT) &&
        !(args_.aType == ge::DT_FLOAT && args_.isHf32 && innerSizeA * aDtypeSize < CACHELINE));
    args_.nd2nzB = ((!innerAlignB || innerSizeB > ND2NZ_ON_THE_FLY_LIMIT) && (args_.bFormat == ge::FORMAT_ND) &&
        (!supportNd2NzOnTheWayB) &&
        !(args_.bType == ge::DT_FLOAT && !args_.isHf32 && innerSizeB < ND2NZ_ON_THE_FLY_LIMIT) &&
        !(args_.bType == ge::DT_FLOAT && args_.isHf32 && innerSizeB * bDtypeSize < CACHELINE));

    OP_LOGD(args_.opName, "After judging nd2nz tiling condition, matrix A need normal mode nd2nz = %u, matrix B = %u.",
            static_cast<uint32_t>(args_.nd2nzA), static_cast<uint32_t>(args_.nd2nzB));
    args_.nd2nzA = args_.nd2nzA || NeedNd2NzVnchw(outerSizeA, innerSizeA, supportNd2NzOnTheWayA,
                                                  aDtypeSize, args_.aFormat);
    args_.nd2nzB = args_.nd2nzB || NeedNd2NzVnchw(outerSizeB, innerSizeB, supportNd2NzOnTheWayB,
                                                  bDtypeSize, args_.bFormat);
    // (k, n) n为16384的倍数时，mata冲突严重，m越大，右矩阵重复载入越多，冲突影响越大，将右矩阵先做nd2nz
    // 限制为fp16、bf16场景
    bool mataConflictFlag = GetNd2nzA(args_, compileInfo_);
    //  B 矩阵转置场景
    bool mataConflictFlag2 = GetNd2nzB(args_, compileInfo_);
    args_.nd2nzB = args_.nd2nzB || mataConflictFlag || mataConflictFlag2;
    OP_LOGI(args_.opName, "After judging nd2nz tiling condition, matrix A need vnchw mode nd2nz = %u, matrix B = %u.",
            static_cast<uint32_t>(args_.nd2nzA), static_cast<uint32_t>(args_.nd2nzB));
    if (args_.nd2nzA && NeedNd2NzVnchw(outerSizeA, innerSizeA, supportNd2NzOnTheWayA, aDtypeSize, args_.aFormat)) {
        args_.unAlignProcessType = 1;
    } else {
        args_.unAlignProcessType = 0;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
