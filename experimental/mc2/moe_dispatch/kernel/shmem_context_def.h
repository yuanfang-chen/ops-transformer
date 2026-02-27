/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
"License").
 * Please refer to the License for details. You may not use this file except in
compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
License.
 */

/*!
 * \file shmem_context_def.h
 * \brief
 */

#include "shmem.h"

constexpr int64_t SHMEM_DATA_OFFSET = 0;
// 给数据区200兆，后面给状态区
constexpr int64_t SHMEM_SIGNAL_OFFSET = 1022 * 1024 * 1024;

__aicore__ inline int64_t GetShmemDataAddr(__gm__ ShmemContext *context, int32_t pe) {
    return (int64_t)aclshmem_ptr(context, pe) + SHMEM_DATA_OFFSET;
}

__aicore__ inline int64_t GetShmemSignalAddr(__gm__ ShmemContext *context, int32_t pe) {
    return (int64_t)aclshmem_ptr(context, pe) + SHMEM_SIGNAL_OFFSET;
}