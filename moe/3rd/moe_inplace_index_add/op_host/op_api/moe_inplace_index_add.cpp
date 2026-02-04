/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file inplace_index_add.cpp
 * \brief
 */
#include "moe_inplace_index_add.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInplaceIndexAdd);
OP_TYPE_REGISTER(InplaceIndexAdd);
// AICORE算子kernel
const aclTensor *MoeInplaceIndexAddAiCore(const aclTensor *self, const int64_t dim, const aclTensor *index,
                                       const aclTensor *source, const aclTensor *alphaTensor,
                                       aclOpExecutor *executor) {
  L0_DFX(MoeInplaceIndexAddAiCore, self, dim, index, source, alphaTensor);
  auto indexAddOut = const_cast<aclTensor*>(self);
  auto ret=ACLNN_SUCCESS;
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
      ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeInplaceIndexAdd, OP_INPUT(self, index, source, alphaTensor),
                                        OP_OUTPUT(indexAddOut), OP_ATTR(dim));
  } else {
      ret = ADD_TO_LAUNCHER_LIST_AICORE(InplaceIndexAdd, OP_INPUT(self, index, source, alphaTensor),
                                        OP_OUTPUT(indexAddOut), OP_ATTR(dim));
  }
  OP_CHECK(ret == ACLNN_SUCCESS,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MoeInplaceIndexAddAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
           return nullptr);
  return indexAddOut;
}

}  // namespace l0op
