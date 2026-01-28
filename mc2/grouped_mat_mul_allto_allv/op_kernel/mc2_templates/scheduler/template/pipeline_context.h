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
 * \file pipeline_context.h
 * \brief
 */

#ifndef QGMMATAV_PIPELINE_CONTEXT_H
#define QGMMATAV_PIPELINE_CONTEXT_H

namespace ATAVKernelTemplate {
//todo 后续可以按节点拆成对应的上下文复用
template <typename TilingDataType>
struct PipelineContext {

GM_ADDR gmmxGM;
GM_ADDR gmmweightGM;
GM_ADDR sendCntsGM;
GM_ADDR recvCntsGM;
GM_ADDR mmxGM;
GM_ADDR mmweightGM;
GM_ADDR gmmbiasGM;
GM_ADDR mmbiasGM;
GM_ADDR gmmyGM;
GM_ADDR gmmOutGM;
GM_ADDR mmyGM;
GM_ADDR workspaceGM;
GM_ADDR gmmxScaleGM;
GM_ADDR gmmWeightScaleGM;
GM_ADDR mmxScaleGM;
GM_ADDR mmWeightScaleGM;
GM_ADDR tilingGM;

TilingDataType* tilingData;

};
};

#endif