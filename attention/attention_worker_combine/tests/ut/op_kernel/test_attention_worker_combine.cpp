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
 * \file test_attention_worker_combine.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_attention_worker_combine.h"
#include "../data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void attention_worker_combine(GM_ADDR schedule_context, GM_ADDR expert_scales, 
                                                            GM_ADDR layer_id, GM_ADDR y, GM_ADDR next_layer_id,
                                                            GM_ADDR workspace, GM_ADDR tiling);

class attention_worker_combine_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "attention_worker_combine_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "attention_worker_combine_test TearDown\n" << endl;
  }
};

TEST_F(attention_worker_combine_test, test_case_0000) {
  size_t scheduleContextSize = 1024 * sizeof(int8_t);
  size_t expertScalesSize = 32 * 9 * sizeof(float);
  size_t layerIdSize = 1 * sizeof(int32_t);
  size_t ySize = 32 * 7168 * sizeof(int8_t);
  size_t nextLayerIdSize = 1 * sizeof(int32_t);

  size_t tiling_data_size = sizeof(AttentionWorkerCombineTilingData);
  uint32_t blockDim = 32;

  uint8_t* scheduleContext = (uint8_t*)AscendC::GmAlloc(scheduleContextSize);
  uint8_t* expertScales = (uint8_t*)AscendC::GmAlloc(expertScalesSize);
  uint8_t* layerId = (uint8_t*)AscendC::GmAlloc(layerIdSize);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
  uint8_t* nextLayer = (uint8_t*)AscendC::GmAlloc(nextLayerIdSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

  char* path_ = get_current_dir_name();
  string path(path_);

  AttentionWorkerCombineTilingData* tilingDatafromBin = reinterpret_cast<AttentionWorkerCombineTilingData*>(tiling);

  tilingDatafromBin->usedCoreNum = 32;
  tilingDatafromBin->BS = 32;
  tilingDatafromBin->K = 8;
  tilingDatafromBin->H = 7168;
  tilingDatafromBin->needSchedule = 0;
  tilingDatafromBin->BsSplitFactor = 1;
  tilingDatafromBin->BsSplitCoreNum = 32;
  tilingDatafromBin->mainCoreBsLoopNum = 1;
  tilingDatafromBin->tailCoreBsLoopNum = 1;
  tilingDatafromBin->HSplitFactor = 7168;
  tilingDatafromBin->HSplitTailFactor = 0;
  tilingDatafromBin->HSplitCoreNum = 1;
  tilingDatafromBin->mainCoreHLoopNum = 0;
  tilingDatafromBin->tailCoreHLoopNum = 1;
  tilingDatafromBin->KSplitFactor = 1;
  tilingDatafromBin->KSplitTailFactor = 0;
  tilingDatafromBin->KSplitLoopNum = 1;

  ICPU_SET_TILING_KEY(10020);
  ICPU_RUN_KF(attention_worker_combine, blockDim, scheduleContext, expertScales, layerId, y, nextLayer, workspace,
              (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(scheduleContext);
  AscendC::GmFree(expertScales);
  AscendC::GmFree(layerId);
  AscendC::GmFree(y);
  AscendC::GmFree(nextLayer);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(attention_worker_combine_test, test_case_0001) {
  size_t scheduleContextSize = 1024 * sizeof(int8_t);
  size_t expertScalesSize = 32 * 9 * sizeof(float);
  size_t layerIdSize = 1 * sizeof(int32_t);
  size_t ySize = 32 * 20480 * sizeof(int8_t);
  size_t nextLayerIdSize = 1 * sizeof(int32_t);

  size_t tiling_data_size = sizeof(AttentionWorkerCombineTilingData);
  uint32_t blockDim = 32;

  uint8_t* scheduleContext = (uint8_t*)AscendC::GmAlloc(scheduleContextSize);
  uint8_t* expertScales = (uint8_t*)AscendC::GmAlloc(expertScalesSize);
  uint8_t* layerId = (uint8_t*)AscendC::GmAlloc(layerIdSize);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
  uint8_t* nextLayer = (uint8_t*)AscendC::GmAlloc(nextLayerIdSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

  char* path_ = get_current_dir_name();
  string path(path_);

  AttentionWorkerCombineTilingData* tilingDatafromBin = reinterpret_cast<AttentionWorkerCombineTilingData*>(tiling);

  tilingDatafromBin->usedCoreNum = 32;
  tilingDatafromBin->BS = 32;
  tilingDatafromBin->K = 8;
  tilingDatafromBin->H = 20480;
  tilingDatafromBin->needSchedule = 0;
  tilingDatafromBin->BsSplitFactor = 1;
  tilingDatafromBin->BsSplitCoreNum = 16;
  tilingDatafromBin->mainCoreBsLoopNum = 2;
  tilingDatafromBin->tailCoreBsLoopNum = 1;
  tilingDatafromBin->HSplitFactor = 11776;
  tilingDatafromBin->HSplitTailFactor = 8704;
  tilingDatafromBin->HSplitCoreNum = 2;
  tilingDatafromBin->mainCoreHLoopNum = 1;
  tilingDatafromBin->tailCoreHLoopNum = 1;
  tilingDatafromBin->KSplitFactor = 1;
  tilingDatafromBin->KSplitTailFactor = 0;
  tilingDatafromBin->KSplitLoopNum = 8;

  ICPU_SET_TILING_KEY(100021);
  ICPU_RUN_KF(attention_worker_combine, blockDim, scheduleContext, expertScales, layerId, y, nextLayer, workspace,
              (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(scheduleContext);
  AscendC::GmFree(expertScales);
  AscendC::GmFree(layerId);
  AscendC::GmFree(y);
  AscendC::GmFree(nextLayer);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(attention_worker_combine_test, test_case_0002) {
  size_t scheduleContextSize = 1024 * sizeof(int8_t);
  size_t expertScalesSize = 32 * 9 * sizeof(float);
  size_t layerIdSize = 1 * sizeof(int32_t);
  size_t ySize = 32 * 1024 * sizeof(int8_t);
  size_t nextLayerIdSize = 1 * sizeof(int32_t);

  size_t tiling_data_size = sizeof(AttentionWorkerCombineTilingData);
  uint32_t blockDim = 32;

  uint8_t* scheduleContext = (uint8_t*)AscendC::GmAlloc(scheduleContextSize);
  uint8_t* expertScales = (uint8_t*)AscendC::GmAlloc(expertScalesSize);
  uint8_t* layerId = (uint8_t*)AscendC::GmAlloc(layerIdSize);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
  uint8_t* nextLayer = (uint8_t*)AscendC::GmAlloc(nextLayerIdSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

  char* path_ = get_current_dir_name();
  string path(path_);

  AttentionWorkerCombineTilingData* tilingDatafromBin = reinterpret_cast<AttentionWorkerCombineTilingData*>(tiling);

  tilingDatafromBin->usedCoreNum = 32;
  tilingDatafromBin->BS = 32;
  tilingDatafromBin->K = 8;
  tilingDatafromBin->H = 1024;
  tilingDatafromBin->needSchedule = 0;
  tilingDatafromBin->BsSplitFactor = 1;
  tilingDatafromBin->BsSplitCoreNum = 32;
  tilingDatafromBin->mainCoreBsLoopNum = 1;
  tilingDatafromBin->tailCoreBsLoopNum = 1;
  tilingDatafromBin->HSplitFactor = 1024;
  tilingDatafromBin->HSplitTailFactor = 0;
  tilingDatafromBin->HSplitCoreNum = 1;
  tilingDatafromBin->mainCoreHLoopNum = 0;
  tilingDatafromBin->tailCoreHLoopNum = 1;
  tilingDatafromBin->KSplitFactor = 8;
  tilingDatafromBin->KSplitTailFactor = 8;
  tilingDatafromBin->KSplitLoopNum = 1;

  ICPU_SET_TILING_KEY(10000);
  ICPU_RUN_KF(attention_worker_combine, blockDim, scheduleContext, expertScales, layerId, y, nextLayer, workspace,
              (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(scheduleContext);
  AscendC::GmFree(expertScales);
  AscendC::GmFree(layerId);
  AscendC::GmFree(y);
  AscendC::GmFree(nextLayer);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}