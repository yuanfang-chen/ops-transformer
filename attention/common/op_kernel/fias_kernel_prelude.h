/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Header set re-parsed identically across every FIAS arch32 device-kernel
// translation unit. Intended to be precompiled once per (SoC, ASC_DEVKIT_MAJOR)
// and reused across the ~1k tiling-key fan-out via -include-pch.
//
// Invariant: nothing here may read ORIG_DTYPE_*, TILING_KEY_VAR, or any other
// per-binary macro the kernel build framework injects. Adding such an include
// silently freezes one specialization into the PCH.

#ifndef FIAS_KERNEL_PRELUDE_H
#define FIAS_KERNEL_PRELUDE_H

#include <type_traits>

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/quantization/ascend_quant.h"

#include "../../fused_infer_attention_score/op_kernel/attn_infra/base_defs.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/arch/arch.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/arch/cross_core_sync.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/arch/resource.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/layout/layout.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/gemm/dispatch_policy.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/gemm/gemm_type.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/gemm/block/block_mmad.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/epilogue/dispatch_policy.hpp"
#include "../../fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue.hpp"

#include "../../fused_infer_attention_score/op_kernel/kernel_common.hpp"
#include "../../fused_infer_attention_score/op_kernel/flash_attention_regular.h"
#include "../../fused_infer_attention_score/op_kernel/flash_attention_regular_decode.h"

#endif  // FIAS_KERNEL_PRELUDE_H
