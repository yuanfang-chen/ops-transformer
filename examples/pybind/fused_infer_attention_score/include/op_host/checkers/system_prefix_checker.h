/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file system_prefix_checker.h
 * \brief
 */

#ifndef SYSTEM_REPFIX_CHECKER_H
#define SYSTEM_REPFIX_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class SystemPrefixChecker : public BaseChecker {
public:
    SystemPrefixChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~SystemPrefixChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // singlepara
    bool CheckSharedPrefixDim(const FiaTilingInfo &fiaInfo);
    bool CheckSharedPrefixDataType(const FiaTilingInfo &fiaInfo);
    bool CheckSharedPrefixShape(const FiaTilingInfo &fiaInfo);
    bool CheckActualSharedPrefixLenData(const FiaTilingInfo &fiaInfo);

    // existence
    bool CheckSharedPrefixExistence(const FiaTilingInfo &fiaInfo);

    // feature
    bool CheckUnSupportFeature(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureAntiquant(const FiaTilingInfo &fiaInfo);

    // multipara
private:
};

} // namespace optiling
#endif // SYSTEM_REPFIX_CHECKER_H