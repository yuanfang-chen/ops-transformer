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
 * \file mhc_post_case.h
 * \brief MhcPost 基础测试用例定义.
 */

#ifndef UTEST_MHC_POST_CASE_H
#define UTEST_MHC_POST_CASE_H

#include <map>
#include <string>
#include <vector>
#include "tests/utils/case.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::mhc_post {

class MhcPostCase : public ops::adv::tests::utils::Case {
public:
    MhcPostCase();
    MhcPostCase(const char *name, bool enable, const char *dbgInfo, ops::adv::tests::utils::OpInfo opInfo,
                ops::adv::tests::utils::Param param = {}, int32_t tilingTemplatePriority = 0);

    virtual bool Run();

protected:
    virtual bool InitParam();
    virtual bool InitOpInfo();

    std::map<std::string, ops::adv::tests::utils::Tensor> mTensors;
};

/**
 * @brief 生成测试张量的辅助函数
 */
inline ops::adv::tests::utils::Tensor GenTensor(const std::string &name, const std::vector<int64_t> &shape,
                                                ge::DataType dtype)
{
    return ops::adv::tests::utils::Tensor(name, shape, dtype);
}

} // namespace ops::adv::tests::mhc_post

#endif // UTEST_MHC_POST_CASE_H