/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V3_TILING_H_
#define RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V3_TILING_H_

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyAdamwV3TilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);
TILING_DATA_FIELD_DEF(float, beta1Power);
TILING_DATA_FIELD_DEF(float, beta2Power);
TILING_DATA_FIELD_DEF(float, lr);
TILING_DATA_FIELD_DEF(float, weightDecay);
TILING_DATA_FIELD_DEF(float, beta1);
TILING_DATA_FIELD_DEF(float, beta2);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, maximizeFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyAdamwV3, ApplyAdamwV3TilingData)

struct ApplyAdamwV3CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class ApplyAdamwV3Tiling {
 public:
  explicit ApplyAdamwV3Tiling(gert::TilingContext *context) : tilingContext_(context) {};

  ge::graphStatus RunTiling();

 protected:
  ge::graphStatus SetTilingData();
  ge::graphStatus CheckIsScalar(size_t inputIdx);
  ge::graphStatus CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape);
  ge::graphStatus CheckSameDtype(size_t inputIdx, const ge::DataType& input0Dtype);
  ge::graphStatus CheckShapeAndType();
  ge::graphStatus ReadScalarInputs();
  ge::graphStatus CalcTilingParams();

 private:
  gert::TilingContext *tilingContext_;
  ApplyAdamwV3TilingData tilingData_;
  bool amsgradAttr_ = false;
  bool maximizeAttr_ = false;
  int64_t totalLength_ = 0;
  float beta1Power_ = 0.0f;
  float beta2Power_ = 0.0f;
  float lr_ = 0.0f;
  float weightDecay_ = 0.0f;
  float beta1_ = 0.0f;
  float beta2_ = 0.0f;
  float epsilon_ = 0.0f;
};

}  // namespace optiling

#endif  // RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V3_TILING_H_
