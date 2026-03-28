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
 * \file left_padding_checker.h
 * \brief
 */

#ifndef LEFT_PADDING_CHECKER_H
#define LEFT_PADDING_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class LeftPaddingChecker : public BaseChecker {
public:
    LeftPaddingChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~LeftPaddingChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    bool CheckShapeSupport(const gert::Tensor *tensor, const std::vector<int64_t> &expectShapeList) const;
    bool CheckSingleDesc(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureActualLen(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureLayout(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureAlibiPse(const FiaTilingInfo &fiaInfo);
    bool CheckFeaturePageAttention(const FiaTilingInfo &fiaInfo);

    // enableNonQuant 相关校验函数
    bool CheckMultiParaShapeAndDim(const FiaTilingInfo &fiaInfo);

private:
};

}  // namespace optiling
#endif  // LEFT_PADDING_CHECKER_H