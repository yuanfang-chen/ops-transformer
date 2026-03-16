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
 * \file test_aclnn_flash_attention_varlen.cpp
 * \brief FlashAttention算子调用示例（变长序列场景）
 *
 * 场景说明：
 *   - TND变长序列场景：通过cu_seqlens_q/kv指定各batch的实际序列长度
 *   - 训练正向：returnSoftmaxLse=1，输出softmax_lse
 *   - 布局：layoutQ="TND"，layoutKv="TND"，layoutOut="TND"
 *   - 标准MHA：N_q=N_kv=4，D=128
 *   - 因果掩码（maskMode=1）
 *   - 数据类型：FP16
 *
 * cu_seqlens说明（与旧接口aclIntArray的差异）：
 *   旧接口使用aclIntArray传递序列长度，本接口改为INT32 Tensor：
 *     cu_seqlens_q shape=(B+1,)，cu_seqlens_q[i+1]-cu_seqlens_q[i]为第i个sample的q序列长度。
 *   例：B=2，每个sequence各128个token => cu_seqlens = [0, 128, 256]
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)      \
    do {                             \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(const std::vector<int64_t> &shape, void **deviceAddr, const char *name)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);
    LOG_PRINT("=== %s (first 8 elements) ===\n", name);
    for (int64_t i = 0; i < std::min(size, int64_t(8)); i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 将host侧数据拷贝到device侧
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/context/stream初始化
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // TND变长序列：B=2个sequence，每个128个token，共T=256个token
    // N_q=N_kv=4（标准MHA），D=128
    const int64_t B    = 2;    // batch size（sequence数量）
    const int64_t T_q  = 256;  // query总token数（B * seqLen_per_sample = 2 * 128）
    const int64_t T_kv = 256;  // kv总token数
    const int64_t N_q  = 4;
    const int64_t N_kv = 4;    // N_q == N_kv（标准MHA）
    const int64_t D    = 128;
    const int64_t seqLen = T_q / B;  // 每个sample的序列长度 = 128

    // TND layout: (T, N, D) — T为总token数
    std::vector<int64_t> qShape             = {T_q,  N_q,  D};
    std::vector<int64_t> kShape             = {T_kv, N_kv, D};
    std::vector<int64_t> vShape             = {T_kv, N_kv, D};
    std::vector<int64_t> attentionOutShape  = {T_q,  N_q,  D};
    // softmaxLse shape for TND layout: (T_q, N_q)
    std::vector<int64_t> softmaxLseShape    = {T_q, N_q};
    // cu_seqlens shape: (B+1,) = (3,) — INT32 tensor
    // cu_seqlens[i+1] - cu_seqlens[i] = 第i个sequence的token数
    std::vector<int64_t> cuSeqlensShape     = {B + 1};

    void *qDeviceAddr            = nullptr;
    void *kDeviceAddr            = nullptr;
    void *vDeviceAddr            = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr   = nullptr;
    void *cuSeqlensQDeviceAddr   = nullptr;
    void *cuSeqlensKvDeviceAddr  = nullptr;

    aclTensor *qTensor            = nullptr;
    aclTensor *kTensor            = nullptr;
    aclTensor *vTensor            = nullptr;
    aclTensor *attentionOutTensor = nullptr;
    aclTensor *softmaxLseTensor   = nullptr;
    aclTensor *cuSeqlensQTensor   = nullptr;
    aclTensor *cuSeqlensKvTensor  = nullptr;

    // FP16 输入数据（0x3C00 = 1.0 in FP16）
    int64_t qSize   = GetShapeSize(qShape);
    int64_t kvSize  = GetShapeSize(kShape);
    int64_t outSize = GetShapeSize(attentionOutShape);
    int64_t lseSize = GetShapeSize(softmaxLseShape);

    std::vector<uint16_t> qHostData(qSize,   0x3C00);
    std::vector<uint16_t> kHostData(kvSize,  0x3C00);
    std::vector<uint16_t> vHostData(kvSize,  0x3C00);
    std::vector<uint16_t> attentionOutHostData(outSize, 0);
    std::vector<float>    softmaxLseHostData(lseSize,   0.0f);  // FLOAT32

    // cu_seqlens: INT32 tensor，shape=(3,)=[0, 128, 256]
    // 表示：第0个sequence的token在[0, 128)，第1个在[128, 256)
    std::vector<int32_t> cuSeqlensQData  = {0, static_cast<int32_t>(seqLen), static_cast<int32_t>(T_q)};
    std::vector<int32_t> cuSeqlensKvData = {0, static_cast<int32_t>(seqLen), static_cast<int32_t>(T_kv)};

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr,
                          aclDataType::ACL_FLOAT16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // softmaxLse输出：FLOAT32
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr,
                          aclDataType::ACL_FLOAT, &softmaxLseTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // cu_seqlens使用INT32 Tensor（非旧接口的aclIntArray）
    ret = CreateAclTensor(cuSeqlensQData, cuSeqlensShape, &cuSeqlensQDeviceAddr,
                          aclDataType::ACL_INT32, &cuSeqlensQTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqlensKvData, cuSeqlensShape, &cuSeqlensKvDeviceAddr,
                          aclDataType::ACL_INT32, &cuSeqlensKvTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 属性参数
    float   softmaxMode      = 0.0f;  // 使用默认缩放 1/sqrt(D)
    int64_t maskMode         = 1;     // 1: 因果掩码（下三角）
    int64_t winLeft          = 0;
    int64_t winRight         = 0;
    const char *layoutQ      = "TND";
    const char *layoutKv     = "TND";
    const char *layoutOut    = "TND";
    int64_t returnSoftmaxLse = 1;     // 训练场景：输出softmax_lse
    int64_t deterministic    = 0;

    // 3. 调用aclnnFlashAttention两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 调用第一段接口计算workspace大小
    ret = aclnnFlashAttentionGetWorkspaceSize(
        qTensor, kTensor, vTensor,
        nullptr,             // blockTableOptional: 无分页KV缓存
        cuSeqlensQTensor,    // cuSeqlensQOptional: INT32 tensor (B+1,)=[0,128,256]
        cuSeqlensKvTensor,   // cuSeqlensKvOptional: INT32 tensor (B+1,)=[0,128,256]
        nullptr,             // sequsedQOptional: 与cuSeqlens互斥，不同时使用
        nullptr,             // sequsedKvOptional
        nullptr,             // sinksOptional
        nullptr,             // metadataOptional
        softmaxMode, maskMode, winLeft, winRight,
        layoutQ, layoutKv, layoutOut,
        returnSoftmaxLse, deterministic,
        attentionOutTensor,
        softmaxLseTensor,    // 训练场景输出softmax_lse
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口执行计算
    ret = aclnnFlashAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnFlashAttention failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧结果拷贝至host侧
    // 打印attentionOut（FP16，TND格式）
    {
        std::vector<uint16_t> attnResult(outSize, 0);
        ret = aclrtMemcpy(attnResult.data(), attnResult.size() * sizeof(uint16_t),
                          attentionOutDeviceAddr, outSize * sizeof(uint16_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy attentionOut failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("=== attentionOut (TND, first 8 elements, FP16 hex) ===\n");
        for (int64_t i = 0; i < std::min(outSize, int64_t(8)); i++) {
            LOG_PRINT("  [%ld] = 0x%04x\n", i, attnResult[i]);
        }
    }

    // 打印softmaxLse（FLOAT32，shape=(T_q, N_q)=(256, 4)）
    PrintOutResult(softmaxLseShape, &softmaxLseDeviceAddr, "softmaxLse (TND)");

    // 6. 释放aclTensor资源
    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(attentionOutTensor);
    aclDestroyTensor(softmaxLseTensor);
    aclDestroyTensor(cuSeqlensQTensor);
    aclDestroyTensor(cuSeqlensKvTensor);

    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    aclrtFree(cuSeqlensQDeviceAddr);
    aclrtFree(cuSeqlensKvDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
