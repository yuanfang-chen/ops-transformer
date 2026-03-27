# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# ============================================================================

set(fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/fused_k_rms_norm_rope_store_kv_cache_mx_quant_base_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.cpp
    )

if (UT_TEST_ALL OR OP_KERNEL_UT)
    AddOpTestCase(fused_k_rms_norm_rope_store_kv_cache_mx_quant "ascend950pr_9599" "-DDTYPE_QKV=bfloat16_t" "${fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling_files}")
endif()
