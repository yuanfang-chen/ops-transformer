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
 * \file softmax_lse_checker.h
 * \brief
 */

#ifndef SOFTMAX_LSE_CHECKER_H
#define SOFTMAX_LSE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class SoftmaxLSEChecker : public BaseChecker {
public:
    SoftmaxLSEChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~SoftmaxLSEChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    bool CheckSingleDtype(const FiaTilingInfo &fiaInfo);
    bool CheckExistenceShapeAndDesc(const FiaTilingInfo &fiaInfo);
    bool CheckMultiParaDimAndShape(const FiaTilingInfo &fiaInfo);

private:
};

} // namespace optiling
#endif // SOFTMAX_LSE_CHECKER_H