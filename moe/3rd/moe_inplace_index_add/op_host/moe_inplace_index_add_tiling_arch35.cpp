/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
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

/* !
 * \file moe_inplace_index_add_tiling_arch35.cpp
 * \brief
 */

#include "tiling_base/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling
{
ge::graphStatus MoeInplaceIndexAddTilingForAscendC(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

int64_t CeilAlign(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

bool GetTilingCoreNum(const gert::TilingParseContext* context, uint32_t& core_num) {
  auto platform_info = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, platform_info, false);

  core_num = platform_info->GetCoreNum();
  OP_LOGD(context->GetNodeName(), "get tiling core num is %u", core_num);
  return true;
}

}  // namespace optiling
