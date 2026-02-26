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
 * \file test_aclnn_causal_conv1d_fn.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_fn.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 定义测试参数
    int64_t cuSeqLen = 16;      // 总序列长度
    int64_t dim = 64;           // 特征维度，需要是16的倍数
    int64_t kernelWidth = 4;    // 卷积核宽度 K <= 6
    int64_t batch = 2;          // batch数量
    int64_t cacheNum = 4;       // cache状态数量，需要 >= batch

    // 定义shape
    std::vector<int64_t> xShape = {cuSeqLen, dim};                              // [cu_seq_len, dim]
    std::vector<int64_t> weightShape = {kernelWidth, dim};                      // [K, dim]
    std::vector<int64_t> cacheStatesShape = {cacheNum, kernelWidth - 1, dim};   // [-1, K-1, dim]
    std::vector<int64_t> cacheIndicesShape = {batch};                           // [batch]
    std::vector<int64_t> seqStartIndexShape = {batch + 1};                      // [batch+1]
    std::vector<int64_t> hasInitialStateShape = {batch};                        // [batch]
    std::vector<int64_t> yShape = {cuSeqLen, dim};                              // 与x相同

    // 定义device侧内存地址
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* cacheStatesDeviceAddr = nullptr;
    void* cacheIndicesDeviceAddr = nullptr;
    void* seqStartIndexDeviceAddr = nullptr;
    void* hasInitialStateDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;

    // 定义aclTensor指针
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* cacheStates = nullptr;
    aclTensor* cacheIndices = nullptr;
    aclTensor* seqStartIndex = nullptr;
    aclTensor* hasInitialState = nullptr;
    aclTensor* y = nullptr;

    // 准备host侧数据
    std::vector<int16_t> hostX(GetShapeSize(xShape), 1);                                    // x数据，使用int16表示fp16
    std::vector<int16_t> hostWeight(GetShapeSize(weightShape), 1);                          // weight数据
    std::vector<int16_t> hostCacheStates(GetShapeSize(cacheStatesShape), 0);               // cacheStates初始化为0
    std::vector<int64_t> hostCacheIndices = {0, 1};                                         // 每个序列对应的cache索引
    std::vector<int64_t> hostSeqStartIndex = {0, 8, 16};                                    // 序列起始位置: 第一个序列[0,8), 第二个序列[8,16)
    std::vector<int8_t> hostHasInitialState = {1, 0};                                       // 第一个序列使用缓存，第二个不使用
    std::vector<int16_t> hostY(GetShapeSize(yShape), 0);                                    // y输出初始化为0

    // 创建x aclTensor
    ret = CreateAclTensor(hostX, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    ret = CreateAclTensor(hostWeight, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建cacheStates aclTensor
    ret = CreateAclTensor(hostCacheStates, cacheStatesShape, &cacheStatesDeviceAddr, aclDataType::ACL_FLOAT16, &cacheStates, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建cacheIndices aclTensor
    ret = CreateAclTensor(hostCacheIndices, cacheIndicesShape, &cacheIndicesDeviceAddr, aclDataType::ACL_INT64, &cacheIndices, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建seqStartIndex aclTensor
    ret = CreateAclTensor(hostSeqStartIndex, seqStartIndexShape, &seqStartIndexDeviceAddr, aclDataType::ACL_INT64, &seqStartIndex, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建hasInitialState aclTensor
    ret = CreateAclTensor(hostHasInitialState, hasInitialStateShape, &hasInitialStateDeviceAddr, aclDataType::ACL_BOOL, &hasInitialState, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建y aclTensor (输出)
    ret = CreateAclTensor(hostY, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnCausalConv1dFn第一段接口
    ret = aclnnCausalConv1dFnGetWorkspaceSize(x, weight, cacheStates, cacheIndices, seqStartIndex,
                                               hasInitialState, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dFnGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnCausalConv1dFn第二段接口
    ret = aclnnCausalConv1dFn(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dFn failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto ySize = GetShapeSize(yShape);
    std::vector<int16_t> resultY(ySize, 0);
    ret = aclrtMemcpy(resultY.data(), resultY.size() * sizeof(resultY[0]), yDeviceAddr,
                      ySize * sizeof(resultY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印部分结果
    LOG_PRINT("y output (first 10 elements):\n");
    for (int64_t i = 0; i < std::min(ySize, (int64_t)10); i++) {
        LOG_PRINT("y[%ld] = %d\n", i, resultY[i]);
    }

    // 获取更新后的cacheStates
    auto cacheStatesSize = GetShapeSize(cacheStatesShape);
    std::vector<int16_t> resultCacheStates(cacheStatesSize, 0);
    ret = aclrtMemcpy(resultCacheStates.data(), resultCacheStates.size() * sizeof(resultCacheStates[0]),
                      cacheStatesDeviceAddr, cacheStatesSize * sizeof(resultCacheStates[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy cacheStates result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\ncacheStates output (first 10 elements):\n");
    for (int64_t i = 0; i < std::min(cacheStatesSize, (int64_t)10); i++) {
        LOG_PRINT("cacheStates[%ld] = %d\n", i, resultCacheStates[i]);
    }

    LOG_PRINT("\nCausalConv1dFn execution success!\n");

    // 6. 释放aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(cacheStates);
    aclDestroyTensor(cacheIndices);
    aclDestroyTensor(seqStartIndex);
    aclDestroyTensor(hasInitialState);
    aclDestroyTensor(y);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(cacheStatesDeviceAddr);
    aclrtFree(cacheIndicesDeviceAddr);
    aclrtFree(seqStartIndexDeviceAddr);
    aclrtFree(hasInitialStateDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}