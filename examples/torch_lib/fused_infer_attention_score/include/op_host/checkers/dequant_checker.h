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
 * \file dequant_checker.h
 * \brief
 */

#ifndef DEQUANT_CHECKER_H
#define DEQUANT_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class DequantChecker : public BaseChecker {
public:
    DequantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~DequantChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // enableNonQuant 相关校验函数
    bool CheckExistenceNoquant(const FiaTilingInfo &fiaInfo);

    // enableFullQuant 相关校验函数
    bool CheckDataTypeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleDtypeMLAFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleDtypeGQAPerblock(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleDtypeGQAPertensor(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleDtypeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDequantModeMLAFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDequantModeGQAPerblock(const FiaTilingInfo &fiaInfo);
    bool CheckDequantModeGQAPertensor(const FiaTilingInfo &fiaInfo);
    bool CheckDequantModeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleDimNumGQAPerblock(const FiaTilingInfo &fiaInfo);

    bool CheckTensorExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName);
    bool CheckTensorNotExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName);
    bool CheckExistencePertensorFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckExistenceMLAFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckExistencePerblockFullquant(const FiaTilingInfo &fiaInfo);

    bool CheckFeaturePertensorFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckFeaturePerblockFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureMLAFullquant(const FiaTilingInfo &fiaInfo);

    bool CheckDequantScaleShapeMLAFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleShapePertensor(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleShapePerblock(const FiaTilingInfo &fiaInfo);
    bool CheckDequantScaleShapeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckInputDTypeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckInputLayoutPerblock(const FiaTilingInfo &fiaInfo);
    bool CheckInputLayoutPertensor(const FiaTilingInfo &fiaInfo);
    bool CheckInputLayoutMLAFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckInputLayoutFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckInputAxisFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckN1SizeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckN2SizeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckQSSizeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckGSizeFullquant(const FiaTilingInfo &fiaInfo);
    bool CheckDSizeFullquant(const FiaTilingInfo &fiaInfo);

    // enableAntiQuant 相关校验函数
    // SinglePara
    bool CheckSingleParaForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckAntiquantModeForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckInputKVTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    
    // Existence
    bool CheckExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckScaleExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckDescExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckOffsetExistenceForAntiquant(const FiaTilingInfo &fiaInfo);

    // Feature
    bool CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckFeatureExtendForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo);

    // MultiPara
    bool CheckMultiParaForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckScaleTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckScaleShapeForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckOffsetTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    bool CheckOffsetShapeForAntiquant(const FiaTilingInfo &fiaInfo);

    bool CheckScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo);
    bool CheckScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerTensorHeadMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerTokenHeadMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerTokenPAMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerTokenHeadPAMode(const FiaTilingInfo &fiaInfo);
    bool CheckKScaleShapeForPerTokenGroupMode(const FiaTilingInfo &fiaInfo);
    bool CheckVScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo);

private:
    bool enablePerblockQuant_ = false;
    bool enablePertensorQuant_ = false;
    bool enableIFAMLAFullQuant_ = false;
};

} // namespace optiling
#endif // DEQUANT_CHECKER_H