/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_gather_matmul_inner.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "common/op_host/op_api/matmul_util.h"
#include "hccl_util.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t id;
  const char *funcName;
  bool hasReg;
} NnopbaseDfxId;

extern "C" aclnnStatus aclnnInnerAllGatherMatmulInnerGetWorkspaceSize(const aclTensor *x1,
    const aclTensor *context, const char *group, int64_t commTurn, uint32_t rankSize, const aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerAllGatherMatmulInner(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream);

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);

aclnnStatus aclnnAllGatherMatmulInnerGetWorkspaceSize(const aclTensor *x1, const aclTensor *context,
    const char *group, int64_t commTurn, uint32_t rankSize,
    const aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor) {
  uint64_t timeStamp = NnopbaseMsprofSysTime();

  OP_LOGD("X1 is %s.", x1->ToString().GetString());

  aclnnStatus ret = aclnnInnerAllGatherMatmulInnerGetWorkspaceSize(x1, context, group, commTurn, rankSize, output, workspaceSize, executor);
  OP_LOGD("AllGatherMatmulInner, aclnnInnerGetWorkspaceSize ret = %d.", ret);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;

}

aclnnStatus aclnnAllGatherMatmulInner(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream) {
  if ((workspace == nullptr) || (workspaceSize == 0UL)) {
    OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
    return ACLNN_SUCCESS;
  }

  return aclnnInnerAllGatherMatmulInner(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif