/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file mask_checker.h
 * \brief
 */

#ifndef MASK_CHECKER_H
#define MASK_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class MaskChecker : public BaseChecker {
public:
    MaskChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~MaskChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    struct MaskInfo {
        int64_t attenMaskN = 1U;
        uint32_t attenMaskBatch = 1;
        uint32_t attenMaskQSize = 0;
        uint32_t attenMaskSize = 0;
        std::string strMaskShape;
    };
    bool CheckDtypeAndFormat(const FiaTilingInfo &fiaInfo);
    bool CheckSparseMode(const FiaTilingInfo &fiaInfo);
    bool CheckNoQuantIFAMLA(const FiaTilingInfo &fiaInfo);
    bool CheckFullQuantIFAMLA(const FiaTilingInfo &fiaInfo);
    bool CheckQKVDDifferent(const FiaTilingInfo &fiaInfo);
    bool CheckPretokenAndNexttoken(const FiaTilingInfo &fiaInfo);
    bool CheckIFADimAndShape(const FiaTilingInfo &fiaInfo);
    bool GetMaskInfo(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo);
    bool CheckDimAndShape(const FiaTilingInfo &fiaInfo);

private:
    bool enableIFAMLA = false;
    bool isIFAFlag = false;
};

}  // namespace optiling
#endif  // MASK_CHECKER_H