/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file ffn_worker_batching_tiling.h
 * \brief
 */
#ifndef FFN_WORKER_BATCHING_TILING_H
#define FFN_WORKER_BATCHING_TILING_H
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FfnWorkerBatchingTilingData)
TILING_DATA_FIELD_DEF(int64_t, Y);
TILING_DATA_FIELD_DEF(int64_t, H);
TILING_DATA_FIELD_DEF(int64_t, tokenDtype);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, sortLoopMaxElement);
TILING_DATA_FIELD_DEF(int64_t, sortNumWorkSpace);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FfnWorkerBatching, FfnWorkerBatchingTilingData)

struct FfnWorkerBatchingCompileInfo {
};

class FfnWorkerBatchingTiling
{
public:
    explicit FfnWorkerBatchingTiling(gert::TilingContext* context) : context_(context){};
    ge::graphStatus RunFfnWorkerBatchingTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus CheckInputParam();
    ge::graphStatus SetBatchingWorkspaceSize(int64_t expertNum);
    ge::graphStatus GetAttrsInfo();
    void SetBatchingTilingKey(int64_t sortLoopMaxElement);

private:
    FfnWorkerBatchingTilingData tilingData_;
    gert::TilingContext *context_ = nullptr;

    int64_t A_ = 0;
    int64_t BS_ = 0;
    int64_t K_ = 0;
    int64_t Y_ = 0;
    int64_t needSchedule_ = 0;
    int64_t layerNum_ = 0;
    uint32_t ubBlockSize_ = 0;
};

} // namespace optiling
#endif