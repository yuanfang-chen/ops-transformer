/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_all_gather_add.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_all_gather_add.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int64_t RANK_SIZE = 2;
constexpr int64_t COMM_TURN = 2;

struct ExampleContext {
    HcclComm hcclComm = nullptr;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char *argv[])
{
    // 1. 调用 acl 进行初始化
    int32_t deviceId = 0;
    ExampleContext exampleContext;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
    ret = aclrtCreateContext(&exampleContext.context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(exampleContext.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);
    ret = aclrtCreateStream(&exampleContext.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);

    int32_t devices[RANK_SIZE] = {0, 1};
    HcclComm comms[RANK_SIZE] = {nullptr};
    ret = HcclCommInitAll(RANK_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret);
        if (exampleContext.stream != nullptr) { aclrtDestroyStream(exampleContext.stream); }
        if (exampleContext.context != nullptr) { aclrtDestroyContext(exampleContext.context); }
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret);
    exampleContext.hcclComm = comms[deviceId];

    char groupName[128] = {0};
    ret = HcclGetCommName(exampleContext.hcclComm, groupName);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret);
        HcclCommDestroy(exampleContext.hcclComm);
        if (exampleContext.stream != nullptr) { aclrtDestroyStream(exampleContext.stream); }
        if (exampleContext.context != nullptr) { aclrtDestroyContext(exampleContext.context); }
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret);

    // 2. 构造输入输出
    std::vector<int64_t> aShape = {1024, 2048};
    std::vector<int64_t> bShape = {2048, 2048};
    std::vector<int64_t> outputShape = {2048, 2048};

    void *aDeviceAddr = nullptr;
    void *bDeviceAddr = nullptr;
    void *aGatheredDeviceAddr = nullptr;
    void *cDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    aclTensor *a = nullptr;
    aclTensor *b = nullptr;
    aclTensor *aGathered = nullptr;
    aclTensor *c = nullptr;

    std::vector<int16_t> aHostData(GetShapeSize(aShape), 1);
    std::vector<int16_t> bHostData(GetShapeSize(bShape), static_cast<int16_t>(10));
    std::vector<int16_t> aGatheredHostData(GetShapeSize(outputShape), 0);
    std::vector<int16_t> cHostData(GetShapeSize(outputShape), 0);

    ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_FLOAT16, &b);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(aGatheredHostData, outputShape, &aGatheredDeviceAddr, aclDataType::ACL_FLOAT16, &aGathered);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cHostData, outputShape, &cDeviceAddr, aclDataType::ACL_FLOAT16, &c);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const char *group = groupName;

    // 4. 调用 aclnnAllGatherAdd 第一段接口
    ret = aclnnAllGatherAddGetWorkspaceSize(a, b, group, RANK_SIZE, COMM_TURN, aGathered, c,
                                            &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAllGatherAddGetWorkspaceSize failed. ret = %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return ret);
    }

    // 5. 调用 aclnnAllGatherAdd 第二段接口
    ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, exampleContext.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherAdd failed. ret = %d\n", ret); return ret);

    // 6. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(exampleContext.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret = %d\n", ret); return ret);

    LOG_PRINT("[INFO] aclnnAllGatherAdd execute success\n");

    // 7. 释放资源
    aclDestroyTensor(a);
    aclDestroyTensor(b);
    aclDestroyTensor(aGathered);
    aclDestroyTensor(c);
    aclrtFree(aDeviceAddr);
    aclrtFree(bDeviceAddr);
    aclrtFree(aGatheredDeviceAddr);
    aclrtFree(cDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (exampleContext.stream != nullptr) {
        aclrtDestroyStream(exampleContext.stream);
    }
    if (exampleContext.hcclComm != nullptr) {
        HcclCommDestroy(exampleContext.hcclComm);
    }
    if (exampleContext.context != nullptr) {
        aclrtDestroyContext(exampleContext.context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
