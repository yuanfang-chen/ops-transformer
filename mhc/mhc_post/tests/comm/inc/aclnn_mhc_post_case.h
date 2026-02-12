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
 * \file aclnn_mhc_post_case.h
 * \brief MhcPost Aclnn 测试用例定义.
 */

#ifndef UTEST_ACLNN_MHC_POST_CASE_H
#define UTEST_ACLNN_MHC_POST_CASE_H

#include "mhc_post_case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/aclnn_context.h"
#include "aclnn_mhc_post_param.h"

namespace ops::adv::tests::mhc_post {

class AclnnMhcPostCase : public ops::adv::tests::mhc_post::MhcPostCase {
public:
    using AclnnContext = ops::adv::tests::utils::AclnnContext;

public:
    /* 算子控制信息 */
    AclnnContext mAclnnCtx;

    /* 输入/输出 参数 */
    AclnnMhcPostParam mAclnnParam;

public:
    AclnnMhcPostCase();
    AclnnMhcPostCase(const char *name, bool enable, const char *dbgInfo, ops::adv::tests::utils::OpInfo opInfo,
                     AclnnMhcPostParam param, int32_t tilingTemplatePriority = 0);
    bool Run() override;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::mhc_post

#endif // UTEST_ACLNN_MHC_POST_CASE_H