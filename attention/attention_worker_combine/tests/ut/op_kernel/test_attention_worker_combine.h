/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_attention_worker_combine.h
 * \brief
 */

#ifndef _TEST_ATTENTION_WORKER_COMBINE_H_
#define _TEST_ATTENTION_WORKER_COMBINE_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct AttentionWorkerCombineTilingData {
  int64_t usedCoreNum = 32; // 使用的核数
  int64_t BS = 32; // batchSize
  int64_t K = 8; // TopK 选出的专家数
  int64_t H = 7168; // hiddenSize
  int64_t needSchedule = 0; // 是否需要扫描 schedule_context
  int64_t BsSplitFactor = 1; // bs轴核内切分
  int64_t BsSplitCoreNum = 32; // bs轴核间切分
  int64_t mainCoreBsLoopNum = 1; // 主核 bs 循环次数
  int64_t tailCoreBsLoopNum = 1; // 尾核 bs 循环次数
  int64_t HSplitFactor = 7168; // hidden 轴核内切分
  int64_t HSplitTailFactor = 0; // hidden 轴核内尾块
  int64_t HSplitCoreNum = 1; // hidden 轴核间切分
  int64_t mainCoreHLoopNum = 0; // 主核 H 循环次数
  int64_t tailCoreHLoopNum = 1; // 尾核 H 循环次数
  int64_t KSplitFactor = 1; // k 轴核内切分
  int64_t KSplitTailFactor = 0; // k 轴尾块切分
  int64_t KSplitLoopNum = 1; // k 轴循环次数
};
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                      \
  AttentionWorkerCombineTilingData tilingData;                                          \
  INIT_TILING_DATA(AttentionWorkerCombineTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                            \
  (tilingData).BS = tilingDataPointer->BS;                                              \
  (tilingData).K = tilingDataPointer->K;                                                \
  (tilingData).H = tilingDataPointer->H;                                                \
  (tilingData).needSchedule = tilingDataPointer->needSchedule;                          \
  (tilingData).BsSplitFactor = tilingDataPointer->BsSplitFactor;                        \
  (tilingData).BsSplitCoreNum = tilingDataPointer->BsSplitCoreNum;                      \
  (tilingData).mainCoreBsLoopNum = tilingDataPointer->mainCoreBsLoopNum;                \
  (tilingData).tailCoreBsLoopNum = tilingDataPointer->tailCoreBsLoopNum;                \
  (tilingData).HSplitFactor = tilingDataPointer->HSplitFactor;                          \
  (tilingData).HSplitTailFactor = tilingDataPointer->HSplitTailFactor;                  \
  (tilingData).HSplitCoreNum = tilingDataPointer->HSplitCoreNum;                        \
  (tilingData).mainCoreHLoopNum = tilingDataPointer->mainCoreHLoopNum;                  \
  (tilingData).tailCoreHLoopNum = tilingDataPointer->tailCoreHLoopNum;                  \
  (tilingData).KSplitFactor = tilingDataPointer->KSplitFactor;                          \
  (tilingData).KSplitTailFactor = tilingDataPointer->KSplitTailFactor;                  \
  (tilingData).KSplitLoopNum = tilingDataPointer->KSplitLoopNum;
#endif