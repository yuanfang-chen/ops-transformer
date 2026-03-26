/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_checker.h
 * \brief
 */

#ifndef ROPE_CHECKER_H
#define ROPE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class RopeChecker : public BaseChecker {
public:
    RopeChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~RopeChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    bool CheckQDsizeSupport(const FiaTilingInfo &fiaInfo);
    bool CheckRopeDSizeSupport(const FiaTilingInfo &fiaInfo);
    bool CheckRopeDtype(const FiaTilingInfo &fiaInfo);
    bool CheckRopeDtypeConsistency(const FiaTilingInfo &fiaInfo);
    bool CheckQKAndQKRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
        const gert::Shape shape, const gert::Shape ropeShape, const std::string &inputName);
    bool CheckPAKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
        const gert::Shape &keyShape, const gert::Shape &keyRopeShape);
    bool CheckTensorlistKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo);
    bool CheckRopeExistence(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureDecodeMLA(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureSupport(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureAntiQuant(const FiaTilingInfo &fiaInfo);
    bool CheckShapeSupport(const FiaTilingInfo &fiaInfo);
    bool CheckAxisSupport(const FiaTilingInfo &fiaInfo);

private:
};

} // namespace optiling
#endif // ROPE_CHECKER_H