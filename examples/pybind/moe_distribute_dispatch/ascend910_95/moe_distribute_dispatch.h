/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// <<<>>>调用函数声明
void moe_distribute_dispatch_demo(
    uint32_t tilingkey, uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* expertIds , uint8_t* scales,
    uint8_t* expandXOut, uint8_t* dynamicScalesOut, uint8_t* expandIdxOut, 
    uint8_t* expertTokenNumsOut, uint8_t* epSendCountsOut, uint8_t* tpSendCountsOut, 
    uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM);
