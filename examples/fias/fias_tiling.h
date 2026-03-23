/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef __FIAS_TILING_H__
#define __FIAS_TILING_H__

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "tiling/platform/platform_ascendc.h"
#include "op_kernel/fias_tilingdata.h"

namespace fias {

// ---------- data types ----------
enum FiasDataType {
    FDT_FLOAT = 0,
    FDT_FLOAT16 = 1,
    FDT_INT8 = 2,
    FDT_INT32 = 3,
    FDT_UINT8 = 4,
    FDT_INT64 = 9,
    FDT_BOOL = 12,
    FDT_BF16 = 27,
    FDT_MAX
};

// ---------- tiling key constants ----------
// Format: FIAS_<dtype>_<layout>_C<config>_<mask>_<pa>_<fd>
// Config: 1 = D64, 3 = D128, 6 = D256, 7 = D512
enum FiasTilingKey : uint64_t {
    FIAS_BF16_BNSD_C1_NOMASK_NOPA_NOFD = 1,
    FIAS_BF16_BNSD_C1_MASK_NOPA_NOFD   = 2,
    FIAS_BF16_BNSD_C1_NOMASK_PA_NOFD   = 3,
    FIAS_BF16_BNSD_C1_MASK_PA_NOFD     = 4,
    FIAS_BF16_BNSD_C3_NOMASK_NOPA_NOFD = 5,
    FIAS_BF16_BNSD_C3_MASK_NOPA_NOFD   = 6,
    FIAS_BF16_BNSD_C3_NOMASK_PA_NOFD   = 7,
    FIAS_BF16_BNSD_C3_MASK_PA_NOFD     = 8,
};

// ---------- tensor descriptor ----------
struct FiasTensorDescTiling {
    std::vector<int64_t> tShape;
    FiasDataType tDataType = FDT_MAX;
    bool tIsNull = false;
};

// ---------- attribute struct ----------
struct FiasAttrTiling {
    int64_t numHeads = 1;
    int64_t numKVHeads = 1;
    double scaleValue = 1.0;
    int64_t preTokens = 65536;
    int64_t nextTokens = 0;
    std::string inputLayout = "BNSD";
    int64_t blockSize = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 0;
    bool softmaxLseFlag = false;
};

// ---------- tiling class ----------
class FiasTiling {
public:
    explicit FiasTiling(FiasTilingData *tilingData) : tilingData_(tilingData) {}

    bool DoTiling(const std::vector<FiasTensorDescTiling> &inTensors, const FiasAttrTiling &attrs,
                  const std::vector<FiasTensorDescTiling> &outTensors, uint64_t &tilingKey, uint64_t &workspaceSize,
                  uint64_t &blockDim)
    {
        if (!GetPlatformInfo()) {
            return false;
        }
        if (!SetInputParams(inTensors, attrs)) {
            return false;
        }
        if (!SetMultiCoreParams(inTensors, attrs)) {
            return false;
        }
        if (!SetInitOutputParams(inTensors, attrs)) {
            return false;
        }

        tilingKey = ComputeTilingKey(inTensors, attrs);
        workspaceSize = ComputeWorkspaceSize(inTensors, attrs);
        blockDim = blockDim_;
        return true;
    }

private:
    bool GetPlatformInfo()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        curSocName_ = ascendcPlatform->GetSocVersion();
        totalCore_ = ascendcPlatform->GetCoreNumAic();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        return (totalCore_ != 0 && ubSize_ != 0);
    }

    bool SetInputParams(const std::vector<FiasTensorDescTiling> &inTensors, const FiasAttrTiling &attrs)
    {
        // inTensors: [query, key, value, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, block_table]
        auto &inputParams = tilingData_->inputParamsRegbase;
        const auto &queryShape = inTensors[0].tShape;

        // Parse BNSD layout: [B, N, S1, D]
        int64_t bSize = queryShape[0];
        int64_t n2Size = queryShape[1];
        int64_t s1Size = queryShape[2];
        int64_t dSize = queryShape[3];

        const auto &keyShape = inTensors[1].tShape;
        int64_t s2Size = keyShape[2];

        const auto &valueShape = inTensors[2].tShape;
        int64_t dSizeV = valueShape[3];

        int64_t gSize = n2Size / attrs.numKVHeads;  // GQA group size

        inputParams.set_bSize(bSize);
        inputParams.set_n2Size(attrs.numKVHeads);
        inputParams.set_gSize(gSize);
        inputParams.set_s1Size(s1Size);
        inputParams.set_s2Size(s2Size);
        inputParams.set_dSize(dSize);
        inputParams.set_dSizeV(dSizeV);
        inputParams.set_scaleValue(static_cast<float>(attrs.scaleValue));
        inputParams.set_preTokens(attrs.preTokens);
        inputParams.set_nextTokens(attrs.nextTokens);
        inputParams.set_layoutType(3);  // BNSD = 3 in IFA internal enum
        inputParams.set_implMode(attrs.innerPrecise == 0 ? 0 : 1);
        inputParams.set_sparseType(static_cast<uint8_t>(attrs.sparseMode));
        inputParams.set_isSoftMaxLseEnable(attrs.softmaxLseFlag ? 1 : 0);
        inputParams.set_fromFused(1);  // mark as coming from FIAS

        // Attention mask
        bool hasMask = !inTensors[3].tIsNull;
        if (hasMask) {
            inputParams.set_attenMaskShapeType(2);  // (1,1,1,S1,S2)
            inputParams.set_attenMaskDataType(1);    // bool
            inputParams.set_attenMaskCompressMode(0); // ALL
        }

        // Actual seq lengths
        inputParams.set_isActualSeqLengthsNull(inTensors[4].tIsNull ? 1 : 0);
        inputParams.set_isActualSeqLengthsKVNull(inTensors[5].tIsNull ? 1 : 0);
        if (!inTensors[4].tIsNull) {
            inputParams.set_actualSeqLengthsSize(static_cast<uint32_t>(inTensors[4].tShape[0]));
        }
        if (!inTensors[5].tIsNull) {
            inputParams.set_actualSeqLengthsKVSize(static_cast<uint32_t>(inTensors[5].tShape[0]));
        }

        // PagedAttention
        bool isPa = (attrs.blockSize > 0) && !inTensors[6].tIsNull;
        if (isPa) {
            inputParams.set_blockSize(static_cast<int32_t>(attrs.blockSize));
            inputParams.set_blockTableDim2(static_cast<int32_t>(inTensors[6].tShape.back()));
        }

        // GQA
        inputParams.set_isGqa(gSize > 1 ? 1 : 0);
        inputParams.set_headNumRatio(static_cast<uint32_t>(gSize));

        // Dropout disabled for inference
        inputParams.set_keepProb(1.0f);
        inputParams.set_needDropMaskOp(0);

        return true;
    }

    bool SetMultiCoreParams(const std::vector<FiasTensorDescTiling> &inTensors, const FiasAttrTiling &attrs)
    {
        auto &multiCoreParams = tilingData_->multiCoreParamsRegbase;
        auto &inputParams = tilingData_->inputParamsRegbase;

        int64_t bSize = inputParams.get_bSize();
        int64_t n2Size = inputParams.get_n2Size();
        int64_t totalBN = bSize * n2Size;

        // Simple core splitting: divide B*N across cores
        uint32_t usedCores = std::min(static_cast<uint32_t>(totalBN), totalCore_);
        if (usedCores == 0) usedCores = 1;
        blockDim_ = usedCores;

        multiCoreParams.set_coreNum(static_cast<int32_t>(usedCores));
        multiCoreParams.set_totalSize(totalBN);
        multiCoreParams.set_s1OuterSize(1);  // decode: s1=1
        multiCoreParams.set_splitFactorSize(1);
        multiCoreParams.set_splitFactorTailSize(1);
        multiCoreParams.set_splitCoreMode(0);

        // Fill bnStartIdx: each core gets totalBN/usedCores items
        uint32_t bnStartIdx[48] = {0};
        int64_t perCore = totalBN / usedCores;
        int64_t remainder = totalBN % usedCores;
        uint32_t idx = 0;
        for (uint32_t c = 0; c < usedCores; c++) {
            idx += static_cast<uint32_t>(perCore + (c < remainder ? 1 : 0));
            bnStartIdx[c] = idx;
        }
        // Fill remaining entries
        for (uint32_t c = usedCores; c < 48; c++) {
            bnStartIdx[c] = static_cast<uint32_t>(totalBN);
        }
        multiCoreParams.set_bnStartIdx(bnStartIdx);

        // sparseStartIdx: not used for decode, set to 0
        int64_t sparseStartIdx[48] = {0};
        multiCoreParams.set_sparseStartIdx(sparseStartIdx);
        multiCoreParams.set_firstFullLoadS1OuterIdx(0);

        return true;
    }

    bool SetInitOutputParams(const std::vector<FiasTensorDescTiling> &inTensors, const FiasAttrTiling &attrs)
    {
        auto &initOutput = tilingData_->initOutputParams;
        auto &inputParams = tilingData_->inputParamsRegbase;

        int64_t bSize = inputParams.get_bSize();
        int64_t n2Size = inputParams.get_n2Size();
        int64_t gSize = inputParams.get_gSize();
        int64_t s1Size = inputParams.get_s1Size();
        int64_t dSizeV = inputParams.get_dSizeV();

        int64_t totalOutputSize = bSize * n2Size * gSize * s1Size * dSizeV;
        initOutput.set_needInit(0);
        initOutput.set_isOneN(n2Size == 1 ? 1 : 0);
        initOutput.set_totalOutputSize(totalOutputSize);
        initOutput.set_totalSoftMaxLseOutputSize(0);
        initOutput.set_singleCoreSize(0);

        return true;
    }

    uint64_t ComputeTilingKey(const std::vector<FiasTensorDescTiling> &inTensors, const FiasAttrTiling &attrs)
    {
        bool hasMask = !inTensors[3].tIsNull;
        bool isPa = (attrs.blockSize > 0) && !inTensors[6].tIsNull;
        int64_t dSize = inTensors[0].tShape[3];

        // Determine config based on D alignment
        // Config 1: D=64 aligned, Config 3: D=128 aligned
        bool isConfig3 = (dSize % 128 == 0) && (dSize >= 128);

        if (isConfig3) {
            if (!hasMask && !isPa) return FIAS_BF16_BNSD_C3_NOMASK_NOPA_NOFD;
            if (hasMask && !isPa) return FIAS_BF16_BNSD_C3_MASK_NOPA_NOFD;
            if (!hasMask && isPa) return FIAS_BF16_BNSD_C3_NOMASK_PA_NOFD;
            if (hasMask && isPa) return FIAS_BF16_BNSD_C3_MASK_PA_NOFD;
        } else {
            if (!hasMask && !isPa) return FIAS_BF16_BNSD_C1_NOMASK_NOPA_NOFD;
            if (hasMask && !isPa) return FIAS_BF16_BNSD_C1_MASK_NOPA_NOFD;
            if (!hasMask && isPa) return FIAS_BF16_BNSD_C1_NOMASK_PA_NOFD;
            if (hasMask && isPa) return FIAS_BF16_BNSD_C1_MASK_PA_NOFD;
        }
        return 0;
    }

    uint64_t ComputeWorkspaceSize(const std::vector<FiasTensorDescTiling> &inTensors, const FiasAttrTiling &attrs)
    {
        // Workspace for matmul lib and intermediate results
        // 16MB is a safe default (same as grouped_matmul)
        return 16ULL * 1024 * 1024;
    }

    FiasTilingData *tilingData_;
    uint32_t totalCore_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t blockDim_ = 1;
    std::string curSocName_;
};

}  // namespace fias

#endif  // __FIAS_TILING_H__
