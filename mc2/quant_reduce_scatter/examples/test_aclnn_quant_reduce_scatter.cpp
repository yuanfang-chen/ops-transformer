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
 * \file test_aclnn_quant_reduce_scatter.cpp
 * \brief
 */
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_quant_reduce_scatter.h"
 
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int DEV_NUM = 2;
constexpr int INTERNAL_LEN = 10;
int g_rankId = 0;
int streamWithTimeout = 10000;
int64_t g_hcclBufferSize = 200;
void GetOption(int argc, char **argv)
{
    while (true) {
        int optionIndex = 0;
        struct option longOptions[] = {
            {"rank_id", 1, 0, 'a'},
            {0, 0, 0, 0}
        };
        int c = getopt_long(argc, argv, "a:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'a':
                g_rankId = atoi(optarg);
                LOG_PRINT("[INFO] rankId = %d\n", g_rankId);
            default:
                break;
        }
    }
}

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret);
              return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadQtReduceScatter(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret);
              return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret);
              return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    std::vector<int64_t> xShape = {1024, 5120};
    std::vector<int64_t> scalesShape = {1024, 40};
    std::vector<int64_t> outputShape = {1024 / DEV_NUM, 5120};
    void *xDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *output = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    long long xShapeSize = GetShapeSize(xShape);
    long long scalesShapeSize = GetShapeSize(scalesShape);
    long long outputShapeSize = GetShapeSize(outputShape);

    std::vector<int8_t> xHostData(xShapeSize, 0);
    std::vector<int8_t> scalesHostData(scalesShapeSize, 0);
    std::vector<int16_t> outputHostData(outputShapeSize, 0);

    // 创建tensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnQuantReduceScatterGetWorkspaceSize(
        x, scales, hcomName, "sum", output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnQuantReduceScatterGetWorkspaceSize failed. ret = %d \n", ret);
                  return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                  return ret);
    }
    // 调用第二阶段接口
    ret = aclnnQuantReduceScatter(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantReduceScatter failed. ret = %d \n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnQuantReduceScatter execute successfully.\n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (output != nullptr) {
        aclDestroyTensor(output);
    }

    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (outputDeviceAddr != nullptr) {
        aclrtFree(outputDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret);
              return ret);
    
    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret);
              return ret);

    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret);
              return ret);

    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret);
              return ret);

    return 0;
}
int main(int argc, char *argv[])
{
    GetOption(argc, argv);
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret);
              return ret);
    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;
    ret = aclrtSetDevice(g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret);
              return ret);
    ret = aclrtCreateContext(&context, g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret);
              return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret);
              return ret);
    // 初始化集合通信域
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    
    config.hcclDeterministic = 1;
    config.hcclBufferSize = g_hcclBufferSize;
    strncpy(config.hcclCommName, "hccl_comm_test", COMM_NAME_MAX_LENGTH - 1);
    const char* rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] get rankTableFile failed.\n");
              return -1);
    ret = HcclCommInitClusterInfoConfig(rankTableFile, g_rankId, &config, &comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret = %d \n", ret);
              return ret);

    Args args;
    args.rankId = g_rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    ret = LaunchOneThreadQtReduceScatter(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LaunchOneThreadQtReduceScatter failed. ret = %d \n", ret);
              return ret);
    aclFinalize();
    return 0;
}
