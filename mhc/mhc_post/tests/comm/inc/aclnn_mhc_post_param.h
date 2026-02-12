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
 * \file aclnn_mhc_post_param.h
 * \brief MhcPost Aclnn 参数信息.
 */

#ifndef UTEST_ACLNN_MHC_POST_PARAM_H
#define UTEST_ACLNN_MHC_POST_PARAM_H

#include "mhc_post_case.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::mhc_post {

class AclnnMhcPostParam : public ops::adv::tests::utils::Param {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    /* 输入输出 */
    AclnnTensor aclnnX;
    AclnnTensor aclnnHRes;
    AclnnTensor aclnnHOut;
    AclnnTensor aclnnHPost;
    AclnnTensor aclnnY;

public:
    AclnnMhcPostParam() = default;

    bool Init();
};

/**
 * @brief AclnnMhcPostParam构造函数辅助
 */
inline AclnnMhcPostParam BuildAclnnMhcPostParam(const std::vector<ops::adv::tests::utils::Tensor> &inputs)
{
    AclnnMhcPostParam param;
    std::map<std::string, ops::adv::tests::utils::Tensor> tensors;
    for (const auto &tensor : inputs) {
        tensors[tensor.Name()] = tensor;
    }
    // 设置张量映射以便后续初始化
    param.mParamData["tensors"] = tensors;
    return param;
}

} // namespace ops::adv::tests::mhc_post

#endif // UTEST_ACLNN_MHC_POST_PARAM_H