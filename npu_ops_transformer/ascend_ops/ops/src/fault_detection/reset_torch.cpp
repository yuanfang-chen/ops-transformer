/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/torch.h>
#include <torch/all.h>
#include <torch/library.h>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <future>
#include <atomic>
#include <memory>
#include <chrono>
#include <sstream>
#include <ATen/Operators.h>
#include <ATen/ATen.h>
#include "acl/acl.h"
#include "op_log.h"
#include "op_api_common.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "stdio.h"

namespace ascend_ops {

namespace Reset {

const int WORKSPACE_COUNT = 3;
const int DIE_PER_SERVER = 16;
const int MAX_SERVER_NUM = 8;
const int PRINT_THRESHOLD = 8 * DIE_PER_SERVER;
const int64_t TEST_TIMEOUT = 150;
const int64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
using aclnn_get_workspace_func = int32_t (*)(const aclTensor *, const char *, const int, const int, uint64_t *, aclOpExecutor**);
using aclnn_get_func = int32_t (*)(void *, const uint64_t, aclOpExecutor *, aclrtStream);
using aclnn_reset_get_workspace_func = int32_t (*)(const aclTensor *, const char *, const int, const int, uint64_t *, aclOpExecutor**);
using aclnn_collect_get_workspace_func = int32_t (*)(const char *, const int, const aclTensor *, uint64_t *, aclOpExecutor**);
using aclnn_qr_get_workspace_func = int32_t (*)(const aclTensor *, bool, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor**);

static int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, message, ##__VA_ARGS__);  \
  } while (0)

class StreamCtxSingle {
private:
    aclrtContext collectCtx;
    aclrtStream collectStream;
    void* total_workspace_addr = nullptr;
    uint64_t aicpu_workspace_size = 0;
    void* aicpu_workspace_addr = nullptr;
    int64_t qr_workspace_size = 0;
    int server_num;
    int device_id;
    bool isInited = false;
    at::Tensor arange_value_npu;
    aclTensor* ctxSelfTensor = nullptr;
    aclTensor* ctxQTensor = nullptr;
    aclTensor* ctxRTensor = nullptr;
    template <typename T>
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
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
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    return 0;
    }

    StreamCtxSingle(int dev_id, int server_number) : device_id(dev_id), server_num(server_number) {
        aclrtCreateContext(&collectCtx, device_id);
        aclrtSetCurrentContext(collectCtx);
        aclrtCreateStream(&collectStream);
        int64_t total_workspace_size = WORK_SPACE_SIZE * (server_num + WORKSPACE_COUNT);
        auto ret = aclrtMalloc(&total_workspace_addr, total_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR");
    }
    StreamCtxSingle(const StreamCtxSingle&) = delete;
    StreamCtxSingle& operator = (const StreamCtxSingle&) = delete;
public:
    ~StreamCtxSingle() {
        if (total_workspace_addr != nullptr) {
            aclrtFree(total_workspace_addr);
            total_workspace_addr = nullptr;
        }
    }
    static StreamCtxSingle& getInstance(int dev_id, int server_number) {
        static StreamCtxSingle single_instance(dev_id, server_number);
        return single_instance;
    }
    int32_t GetCtx(int32_t server_num, aclrtContext *ctx) {
        if (server_num >= MAX_SERVER_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get Ctx server num bigger than max server num 8.\n");
            return -1;
        }
        *ctx = detectCtx[server_num];
        return 0;
    }
    int32_t GetStream(int32_t server_num, aclrtStream *stream) {
        if (server_num >= MAX_SERVER_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Get Ctx server num bigger than max server num 8.\n");
            return -1;
        }
        *stream = detectStream[server_num];
        return 0;
    }
    aclrtContext GetCollectCtx() {
        return collectCtx;
    }
    aclrtStream GetCollectStream() {
        return collectStream;
    }
    void* GetTestWorkSpaceAddr(uint64_t server_num) {
        if (total_workspace_addr != nullptr) {
            return (void*)((uint8_t*)total_workspace_addr + WORK_SPACE_SIZE * WORKSPACE_COUNT + server_num * WORK_SPACE_SIZE);
        } else {
            return nullptr;
        }
    }
    void* GetAicpuWorkSpaceAddr(uint64_t workspaceSize) {
        std::cout << "aicpu workspaceSize" << aicpu_workspace_addr << "aicpu_workspace_addr" << workspaceSize << std::endl;
        if (aicpu_workspace_size == 0) {
            std::cout << "aicpu workspaceSize" << workspaceSize << std::endl;
            auto ret = aclrtMalloc(&aicpu_workspace_addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aicpu_workspace_size = workspaceSize;
        }
        if (workspaceSize == aicpu_workspace_size) {
            return aicpu_workspace_addr;
        } else {
            return nullptr;
        }
    }
    void* GetWorkSpaceAddr(int count) {
        if (total_workspace_addr != nullptr) {
            return (void*)((uint8_t*)total_workspace_addr + count * WORK_SPACE_SIZE);
        } else {  
            return nullptr;
        }
    }
    at::Tensor& GetArrangeTensor() {
        if (!isInited) {
            int detection_world_size = server_num * DIE_PER_SERVER;
            at::Tensor arange_value = torch::arange(0, detection_world_size, 1, torch::kInt32);
            arange_value_npu = at::empty({detection_world_size}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
                .memory_format(c10::MemoryFormat::Contiguous));
            aclrtMemcpy(arange_value_npu.data_ptr(), detection_world_size * sizeof(int32_t),
                arange_value.data_ptr(), detection_world_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            isInited = true;
        }
        return arange_value_npu;
    }

    void GetAicpuTestTensor(aclTensor** self, aclTensor** q, aclTensor** r) {
        if (ctxSelfTensor == nullptr) {
            std::vector<int64_t> selfShape = {2, 2};
            std::vector<int64_t> qShape = {2, 2};
            std::vector<int64_t> rShape = {2, 2};
            void* selfDeviceAddr = nullptr;
            void* qDeviceAddr = nullptr;
            void* rDeviceAddr = nullptr;
            std::vector<float> selfHostData = {1, 2, 3, 4};
            std::vector<float> qHostData = {0, 0, 0, 0};
            std::vector<float> rHostData = {0, 0, 0, 0};
            // 创建self aclTensor
            CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &ctxSelfTensor);
            // 创建q,r aclTensor
            CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &ctxQTensor);
            CreateAclTensor(rHostData, rShape, &rDeviceAddr, aclDataType::ACL_FLOAT, &ctxRTensor);
            std::cout << "CreateTensor" << std::endl;
        }
        *self = ctxSelfTensor;
        *q = ctxQTensor;
        *r = ctxRTensor;
    }
};

static void CleanAndReset(StreamCtxSingle& streamCtx, c10::string_view group_ep, int64_t ep_world_size, c10::string_view group_barrier,
                          int64_t barrier_world_size, at::Tensor &output_elastic_rank, at::Tensor &output_elastic_rank_npu, int64_t detection_world_size)
{
    aclrtStream collect_stream = streamCtx.GetCollectStream();
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    aclrtMemcpy(output_elastic_rank_npu.data_ptr(), detection_world_size * sizeof(int32_t),
        output_elastic_rank.data_ptr(), detection_world_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    static const auto resetGetWorkspaceSizeFuncAddr =
    reinterpret_cast<aclnn_reset_get_workspace_func>(GetOpApiFuncAddr("aclnnMoeDistributeBufferResetGetWorkspaceSize"));
    static const auto resetOpApiFuncAddr = reinterpret_cast<aclnn_get_func>(GetOpApiFuncAddr("aclnnMoeDistributeBufferReset"));
    TORCH_CHECK(
        resetGetWorkspaceSizeFuncAddr != nullptr && resetOpApiFuncAddr != nullptr,
        "aclnnMoeDistributeBufferResetGetWorkspaceSize or aclnnMoeDistributeBufferReset symbol not found.");
    const int64_t stride = 1;
    const int64_t storage_dims = ep_world_size;
    const int64_t view_dims = ep_world_size;
    aclTensor *elastic_rank_tensor = aclCreateTensor(&view_dims, 1, ACL_INT32, &stride, 0,
            ACL_FORMAT_ND, &storage_dims, 1, output_elastic_rank_npu.data_ptr());
    uint64_t workspace_size = 0;
    void *workspace_addr = nullptr;
    aclOpExecutor *executor = nullptr;
    resetGetWorkspaceSizeFuncAddr(elastic_rank_tensor, const_cast<char*>(group_ep.data()),
        ep_world_size, 0, &workspace_size, &executor);
    if (workspace_size > 0) {
        workspace_addr = streamCtx.GetWorkSpaceAddr(1);
    }
    resetOpApiFuncAddr(workspace_addr, workspace_size, executor, collect_stream);

    uint64_t barrier_workspace_size = 0;
    void *barrier_workspace_addr = nullptr;
    aclOpExecutor *barrier_executor = nullptr;
    resetGetWorkspaceSizeFuncAddr(elastic_rank_tensor, const_cast<char*>(group_barrier.data()),
        barrier_world_size, 0, &barrier_workspace_size, &barrier_executor);
    if (barrier_workspace_size > 0) {
        barrier_workspace_addr = streamCtx.GetWorkSpaceAddr(2);
    }
    resetOpApiFuncAddr(barrier_workspace_addr, barrier_workspace_size, barrier_executor, collect_stream);
    aclrtSynchronizeStreamWithTimeout(collect_stream, 10000);
}

void DetecTionThreadFun(at::Tensor &elastic_rank_table, at::Tensor &elastic_rank_table_cpu,
    at::Tensor &output_elastic_rank, at::Tensor &output_elastic_rank_npu,
    c10::string_view group_ep, int64_t ep_world_size,
    int64_t ep_rank_id, c10::string_view group_detection, int64_t detection_world_size, int64_t detection_rank_id,
    c10::string_view group_barrier, int64_t barrier_world_size, int64_t barrier_rank_id, int64_t test_timeout) 
{
    auto start_time = std::chrono::high_resolution_clock::now();
    OP_DFX_LOGI("Start fault detection.");
    int32_t serverNum = detection_world_size / DIE_PER_SERVER;
    int32_t lastServerDieNum = detection_world_size % DIE_PER_SERVER;
    StreamCtxSingle& streamCtx = StreamCtxSingle::getInstance(static_cast<int>(elastic_rank_table.device().index()), serverNum);
    
    AicpuTest(streamCtx, detection_world_size);
    auto aicpu_test_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(aicpu_test_time - start_time);
    OP_DFX_LOGI("Finish testing aicpu at %dms.", duration);

    AssignTest(streamCtx, serverNum, detection_world_size, group_detection, test_timeout);
    auto elastic_test_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(elastic_test_time - aicpu_test_time);
    OP_DFX_LOGI("Finish detection sending at %dms with seperation of %dms", duration, test_timeout);

    AssignCollect(streamCtx, group_detection, detection_world_size, elastic_rank_table, elastic_rank_table_cpu);
    auto info_collect_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(info_collect_time - elastic_test_time);
    OP_DFX_LOGI("Finish detection receiving at %dms", duration);
    PrintCollectResult(elastic_rank_table_cpu, detection_world_size);
    bool needClean =
        CalValidRankListInfo(detection_rank_id, elastic_rank_table_cpu, detection_world_size, output_elastic_rank);
    if (needClean) {
        CleanAndReset(streamCtx, group_ep, ep_world_size, group_barrier, barrier_world_size, output_elastic_rank, output_elastic_rank_npu, detection_world_size);
    }
    
    auto clean_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(clean_time - info_collect_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(clean_time - start_time);
    OP_DFX_LOGI("Finish detection cleaning at %dms. Total time cost is %dms", duration, total_duration);
}

at::Tensor detection_npu(const at::Tensor &expand_x, c10::string_view group_ep, int64_t ep_world_size,
                               int64_t ep_rank_id,
                               c10::string_view group_detection,
                               int64_t detection_world_size,
                               int64_t detection_rank_id,
                               c10::string_view group_barrier,
                               int64_t barrier_world_size,
                               int64_t barrier_rank_id,
                               int64_t test_timeout) {
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size) && (test_timeout >= 0),
              "ep_rank_id should be in [0, ep_world_size), but got",
              " ep_world_size: ", ep_world_size, ", ep_rank_id: ", ep_rank_id);
    TORCH_CHECK(
        (detection_rank_id >= 0) && (detection_rank_id < detection_world_size),
        "detection_rank_id should be in [0, detection_world_size), but got",
        " detection_world_size: ", detection_world_size,
        ", detection_rank_id: ", detection_rank_id);
    TORCH_CHECK(
        (barrier_rank_id >= 0) && (barrier_rank_id < barrier_world_size),
        "detection_rank_id should be in [0, detection_world_size), but got",
        " detection_world_size: ", barrier_world_size,
        ", detection_rank_id: ", barrier_rank_id);
    test_timeout = test_timeout == 0 ? TEST_TIMEOUT : test_timeout;
    at::Tensor output_elastic_rank = torch::full({detection_world_size}, 0, torch::kInt);
    at::Tensor output_elastic_rank_npu = at::empty({detection_world_size}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
        .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor elastic_rank_table = at::empty({detection_world_size * detection_world_size}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
        .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor elastic_rank_cpu = torch::full({detection_world_size * detection_world_size}, 0, torch::kInt);
    SimpleSingleThread& threadPool = SimpleSingleThread::getInstance();
    threadPool.executeAndWait(std::bind(DetecTionThreadFun, elastic_rank_table, elastic_rank_cpu, output_elastic_rank, output_elastic_rank_npu,
        group_ep, ep_world_size, ep_rank_id, group_detection, detection_world_size, detection_rank_id,
        group_barrier, barrier_world_size, barrier_rank_id, test_timeout));
    return output_elastic_rank;
}

// Register Ascend implementations for isfinite
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("detection", detection_npu);
}

} // namespace FaultDetection
} // namespace ascend_ops