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
 * \file mhc_post_tiling.h
 * \brief MhcPost tiling header
 */

#ifndef MHC_POST_TILING_H
#define MHC_POST_TILING_H

#include "graph/tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling_data_base.h"

namespace optiling {
struct MhcPostTilingData {
    uint32_t totalLength;      // Total elements in input tensor
    uint32_t coreNum;          // Number of AI cores to use
    uint32_t singleCoreLength; // Elements per core
};

class MhcPostTiling : public TilingData<MhcPostTilingData> {
public:
    MhcPostTiling() = default;
    ~MhcPostTiling() = default;

    void set_totalLength(uint32_t totalLength) { data_.totalLength = totalLength; }
    void set_coreNum(uint32_t coreNum) { data_.coreNum = coreNum; }
    void set_singleCoreLength(uint32_t singleCoreLength) { data_.singleCoreLength = singleCoreLength; }

    uint32_t get_totalLength() const { return data_.totalLength; }
    uint32_t get_coreNum() const { return data_.coreNum; }
    uint32_t get_singleCoreLength() const { return data_.singleCoreLength; }

    MhcPostTilingData &get_data() { return data_; }
    const MhcPostTilingData &get_data() const { return data_; }
};
} // namespace optiling

#endif // MHC_POST_TILING_H