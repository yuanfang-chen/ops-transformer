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
 * \file aclnn_mhc_post_case.cpp
 * \brief MhcPost Aclnn 测试用例实现.
 */

#include <utility>
#include "tests/utils/log.h"
#include "aclnn_mhc_post.h"
#include "aclnn_mhc_post_case.h"

using namespace ops::adv::tests::mhc_post;

bool MhcPostTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnMhcPostCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;

    aclnnStatus ret = aclnnMhcPostGetWorkspaceSize(
        aclnnParam->aclnnX.GetAclTensor(),
        aclnnParam->aclnnHRes.GetAclTensor(),
        aclnnParam->aclnnHOut.GetAclTensor(),
        aclnnParam->aclnnHPost.GetAclTensor(),
        aclnnParam->aclnnY.GetAclTensor(),
        workSpaceSize,
        opExecutor);

    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnMhcPostGetWorkspaceSize failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

bool MhcPostKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnMhcPostCase *>(curCase);
    auto *aclnnCtx = &cs->mAclnnCtx;

    aclnnStatus ret = aclnnMhcPost(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(),
                                    aclnnCtx->GetAclOpExecutor(), aclnnCtx->GetAclRtStream());

    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnMhcPost failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

AclnnMhcPostCase::AclnnMhcPostCase() : MhcPostCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnMhcPostParam())
{
}

AclnnMhcPostCase::AclnnMhcPostCase(const char *name, bool enable, const char *dbgInfo,
                                    ops::adv::tests::utils::OpInfo opInfo, AclnnMhcPostParam aclnnParam,
                                    int32_t tilingTemplatePriority)
    : MhcPostCase(name, enable, dbgInfo, std::move(opInfo), {}, tilingTemplatePriority),
      mAclnnParam(std::move(aclnnParam))
{
}

bool AclnnMhcPostCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnMhcPostCase::InitOpInfo()
{
    if (!MhcPostCase::InitOpInfo()) {
        return false;
    }

    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(MhcPostTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(MhcPostKernelRunCbf);
    rst = rst && mAclnnCtx.SetOutputs({&mAclnnParam.aclnnY});
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnMhcPostCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool AclnnMhcPostCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}