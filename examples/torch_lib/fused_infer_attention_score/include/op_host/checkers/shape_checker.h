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
 * \file shape_checker.h
 * \brief
 */

#ifndef SHAPE_CHECKER_H
#define SHAPE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

using std::map;
namespace optiling {
class ShapeChecker : public BaseChecker {
public:
    ShapeChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~ShapeChecker() override = default;

    bool CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    bool CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    bool CheckFeature(const FiaTilingInfo &fiaInfo) override;
    bool CheckMultiPara(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    bool CheckInputFormat(const FiaTilingInfo &fiaInfo);
    bool CheckParaExistenceImpl(const FiaTilingInfo &fiaInfo);
    bool CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                     std::map<std::string, std::vector<DataType>> dataMap);
    bool CheckPAKeyValue(const FiaTilingInfo &fiaInfo);
    bool CheckEmptyTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckMultiDtype(const FiaTilingInfo &fiaInfo);
    bool CheckAxis(const FiaTilingInfo &fiaInfo);
    bool CheckQueryOutConsistency(const FiaTilingInfo &fiaInfo);
    bool CheckKeyValueConsistency(const FiaTilingInfo &fiaInfo);
    bool CheckQueryShape(const FiaTilingInfo &fiaInfo);
    bool CheckKeyNHVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape);
    bool CheckKeyDVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape);
    bool CheckKeyShape(const FiaTilingInfo &fiaInfo);
    bool CheckQueryKeyConsistency(const FiaTilingInfo &fiaInfo);
    bool CheckQueryKeyTensorlistConsistency(const FiaTilingInfo &fiaInfo);
    bool CheckMultiAttr(const FiaTilingInfo &fiaInfo);
    void GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
        const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim, uint32_t i);

    // enableNonQuant 相关校验函数
    bool CheckNonQuantDataType(const FiaTilingInfo &fiaInfo);
    bool CheckNonQuantAttr(const FiaTilingInfo &fiaInfo);
    bool CheckNonQuantHeadNum(const FiaTilingInfo &fiaInfo);
    bool CheckNonQuantInputLayout(const FiaTilingInfo &fiaInfo);
    bool CheckNonQuantInnerPrecise(const FiaTilingInfo &fiaInfo);
    bool CheckTNDLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckNTDLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckTransposeLayoutCrossover(const FiaTilingInfo &fiaInfo);

private:
};

} // namespace optiling
#endif // SHAPE_CHECKER_H