/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

<<<<<<<< HEAD:mc2/common/op_tiling/aclnn_util.h
/*!
 * \file aclnn_util.h
 * \brief
 */
#ifndef COMMON_NN_ACLNN_UTIL_H
#define COMMON_NN_ACLNN_UTIL_H

#define ACLNN_API __attribute__((visibility("default")))

#endif  // COMMON_NN_ACLNN_UTIL_H
========
#ifndef GEMM_TILE_COPY_L1_TO_BT_HPP
#define GEMM_TILE_COPY_L1_TO_BT_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/arch.hpp"
#include "../../../attn_infra/layout/layout.hpp"
#include "../../../attn_infra/gemm/gemm_type.hpp"


namespace NpuArch::Gemm::Tile {


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L1_TO_BT_HPP
>>>>>>>> upstream/master:attention/block_sparse_attention/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_bt.hpp
