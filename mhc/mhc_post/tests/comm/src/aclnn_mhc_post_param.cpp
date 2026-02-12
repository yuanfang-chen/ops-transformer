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
 * \file aclnn_mhc_post_param.cpp
 * \brief MhcPost Aclnn 参数信息实现.
 */

#include "aclnn_mhc_post_param.h"
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/log.h"

using namespace ops::adv::tests::mhc_post;

bool AclnnMhcPostParam::Init()
{
    // 获取张量数据
    auto tensorsMap = std::any_cast<std::map<std::string, ops::adv::tests::utils::Tensor>>(
        mParamData["tensors"]);

    if (tensorsMap.find("x") != tensorsMap.end()) {
        aclnnX = ops::adv::tests::utils::AclnnTensor(tensorsMap["x"]);
    }
    if (tensorsMap.find("h_res") != tensorsMap.end()) {
        aclnnHRes = ops::adv::tests::utils::AclnnTensor(tensorsMap["h_res"]);
    }
    if (tensorsMap.find("h_out") != tensorsMap.end()) {
        aclnnHOut = ops::adv::tests::utils::AclnnTensor(tensorsMap["h_out"]);
    }
    if (tensorsMap.find("h_post") != tensorsMap.end()) {
        aclnnHPost = ops::adv::tests::utils::AclnnTensor(tensorsMap["h_post"]);
    }
    if (tensorsMap.find("y") != tensorsMap.end()) {
        aclnnY = ops::adv::tests::utils::AclnnTensor(tensorsMap["y"]);
    }

    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);

    // 释放设备数据并重新分配
    for (auto *t : {&aclnnX, &aclnnHRes, &aclnnHOut, &aclnnHPost, &aclnnY}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return false;
        }
        std::string filePath = std::string(cs->GetRootPath()) + t->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!t->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    return true;
}