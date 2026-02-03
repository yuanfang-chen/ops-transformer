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
 * \file gmm_a2a_int8_tiling.h
 * \brief INT8 tiling class for allto_allv_grouped_mat_mul on ascend950
 */
#ifndef GMM_A2A_INT8_TILING_H
#define GMM_A2A_INT8_TILING_H

#include "../3rd/gmm_qbmm_tiling.h"

namespace optiling {

class GmmA2AInt8Tiling : public GroupedQbmmTiling {
public:
    explicit GmmA2AInt8Tiling(gert::TilingContext *context);
    ~GmmA2AInt8Tiling() override = default;

    void Reset(gert::TilingContext *context) override;

protected:
    // Override methods for gmmA2A-specific behavior
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;

    // New methods specific to gmmA2A INT8
    virtual bool AnalyzeHcclTiling();
    virtual void CombineGmmAndHcclTiling();

private:
    // Additional tiling data specific to gmmA2A INT8
    uint64_t hcclTilingKey_ = 0;
    bool isInt8Mode_ = true;
};

} // namespace optiling

#endif // GMM_A2A_INT8_TILING_H
