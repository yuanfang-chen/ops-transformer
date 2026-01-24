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
 * \file attention_worker_combine_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ATTENTION_WORKER_COMBINE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ATTENTION_WORKER_COMBINE_H_

#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"
#include "register/op_compile_info_base.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionWorkerCombineTilingData)
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum); // 使用的核数
  TILING_DATA_FIELD_DEF(int64_t, BS); // batchSize
  TILING_DATA_FIELD_DEF(int64_t, K); // TopK 选出的专家数
  TILING_DATA_FIELD_DEF(int64_t, H); // hiddenSize
  TILING_DATA_FIELD_DEF(int64_t, needSchedule); // 是否需要扫描 schedule_context
  TILING_DATA_FIELD_DEF(int64_t, BsSplitFactor); // bs轴核内切分
  TILING_DATA_FIELD_DEF(int64_t, BsSplitCoreNum); // bs轴核间切分
  TILING_DATA_FIELD_DEF(int64_t, mainCoreBsLoopNum); // 主核 bs 循环次数
  TILING_DATA_FIELD_DEF(int64_t, tailCoreBsLoopNum); // 尾核 bs 循环次数
  TILING_DATA_FIELD_DEF(int64_t, HSplitFactor); // hidden 轴核内切分
  TILING_DATA_FIELD_DEF(int64_t, HSplitTailFactor); // hidden 轴核内尾块
  TILING_DATA_FIELD_DEF(int64_t, HSplitCoreNum); // hidden 轴核间切分
  TILING_DATA_FIELD_DEF(int64_t, mainCoreHLoopNum); // 主核 H 循环次数
  TILING_DATA_FIELD_DEF(int64_t, tailCoreHLoopNum); // 尾核 H 循环次数
  TILING_DATA_FIELD_DEF(int64_t, KSplitFactor); // k 轴核内切分
  TILING_DATA_FIELD_DEF(int64_t, KSplitTailFactor); // k 轴尾块切分
  TILING_DATA_FIELD_DEF(int64_t, KSplitLoopNum); // k 轴循环次数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionWorkerCombine, AttentionWorkerCombineTilingData)

struct AttentionWorkerCombineCompileInfo {
  int64_t coreNum = 0;
  int64_t ubSize = 0;
};

class AttentionWorkerCombineTiling : public TilingBaseClass {
  public:
    explicit AttentionWorkerCombineTiling(gert::TilingContext* context_) : TilingBaseClass(context_) {
    }
    ~AttentionWorkerCombineTiling() override {
    }

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
  
  protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

  void DoOpTilingFullK(int64_t batchSize, int64_t bsCoreNum, int64_t hAlign, int64_t kIn);
  void SelectBestHCore(int64_t hAlign, int64_t hInSplitK, int64_t &bsCoreNum, int64_t &lastCoreNum, int64_t &lastBestHCore);

private:
  int64_t tokenDtype_ = 0;
  AttentionWorkerCombineTilingData tilingData_;
};

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_ATTENTION_WORKER_COMBINE_H_
