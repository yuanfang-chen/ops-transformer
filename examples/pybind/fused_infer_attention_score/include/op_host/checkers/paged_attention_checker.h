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
 * \file paged_attention_checker.h
 * \brief
 */

#ifndef PAGED_ATTENTION_CHECKER_H
#define PAGED_ATTENTION_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PagedAttentionChecker : public BaseChecker {
public:
    PagedAttentionChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PagedAttentionChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数

    bool CheckBlockTableDtype(const FiaTilingInfo &fiaInfo);
    bool CheckQDtypeSupport(const FiaTilingInfo &fiaInfo);
    bool CheckBlockTableExistence(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureExistence(const FiaTilingInfo &fiaInfo);
    bool CheckSeqLengthKVExistence(const FiaTilingInfo &fiaInfo);

    int64_t GetMaxBlockNumPerBatch(const FiaTilingInfo &fiaInfo);
    bool CheckMaskShape(const FiaTilingInfo &fiaInfo);

    bool CheckBlockTableShape(const FiaTilingInfo &fiaInfo);
    bool CheckBlockSize(const FiaTilingInfo &fiaInfo);
    bool CheckPADimNum(const FiaTilingInfo &fiaInfo);

    bool CheckPACacheShape(const FiaTilingInfo &fiaInfo, const gert::Shape tempShape,
        const std::string& inputName);
};

} // namespace optiling
#endif // PAGED_ATTENTION_CHECKER_H