# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.  You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Lazy import to avoid JIT compiling all operators when only one is needed.
# Each operator is loaded only when explicitly accessed.

_LAZY_IMPORTS = {
    "npu_moe_distribute_dispatch_v2": ".moe_distribute_dispatch_v2",
    "npu_moe_distribute_combine_v2": ".moe_distribute_combine_v2",
    "npu_moe_distribute_dispatch_v3": ".moe_distribute_dispatch_v3",
    "npu_moe_distribute_combine_v3": ".moe_distribute_combine_v3",
    "MoeDistributeBuffer": ".deep_ep",
    "converter_moe_distribute_dispatch_v3": ".graph_convert.graph_convert_moe_distribute_dispatch_v3",
    "convert_npu_moe_distribute_combine_v3": ".graph_convert.graph_convert_moe_distribute_combine_v3",
    "npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant": ".fused_k_rms_norm_rope_store_kv_cache_mx_quant",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())
