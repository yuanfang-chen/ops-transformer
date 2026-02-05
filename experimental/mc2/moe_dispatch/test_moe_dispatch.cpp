/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <algorithm>
#include <sys/wait.h>
#include <unistd.h>

#include "aclnn/opdev/fp16_t.h"
#include "aclnn/opdev/bfloat16.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem.h"
#include "hccl/hccl.h"
#include "moe_dispatch.h"
#include "kernel/moe_distribute_dispatch_shmem_tiling.h"


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


int n_ranks = 2;
const char *ipport = "tcp://127.0.0.1:50001";
int f_rank = 0;
int f_npu = 0;
const char *data_type = "int";
constexpr int64_t BLOCK_NUM = 64;
constexpr int64_t gworkSpaceSize = 256*1024*1024;
constexpr int64_t BS = 8;
constexpr int64_t H = 7168;
constexpr int64_t K = 1;
constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

const uint32_t MACHINE_NUM = 1;
const char* rank_table_file = std::getenv("RANK_TABLE_FILE");
const char* first_rank_id = std::getenv("FIRST_RANK_ID");
const char* env_dev_num = std::getenv("ENV_DEV_NUM");

const uint32_t EP_WORLD_SIZE = (!first_rank_id) ? 2 : 16;
const uint32_t TP_WORLD_SIZE = (!first_rank_id) ? 1 : 0;
const uint32_t DEV_NUM = (!first_rank_id) ? EP_WORLD_SIZE * TP_WORLD_SIZE : EP_WORLD_SIZE;


aclshmemx_uniqueid_t default_flag_uid;

inline int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port, aclshmemx_uniqueid_t default_flag_uid,
                       aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }
    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    return ACLSHMEM_SUCCESS;
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
int AllocDeviceRes(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    return 0;
}
const uint32_t MACHINE_NUM = 1;
const char* rank_table_file = std::getenv("RANK_TABLE_FILE");
const char* first_rank_id = std::getenv("FIRST_RANK_ID");
const char* env_dev_num = std::getenv("ENV_DEV_NUM");

const uint32_t EP_WORLD_SIZE = (!first_rank_id) ? 2 : 16;
const uint32_t TP_WORLD_SIZE = (!first_rank_id) ? 1 : 0;
const uint32_t DEV_NUM = (!first_rank_id) ? EP_WORLD_SIZE * TP_WORLD_SIZE : EP_WORLD_SIZE;


int InitTilingData(MoeDistributeDispatchShmemTilingData* TilingAddr) {
    
    TilingAddr->MoeDistributeDispatchShmemInfo.bs = BS;
    TilingAddr->MoeDistributeDispatchShmemInfo.h = H;
    TilingAddr->MoeDistributeDispatchShmemInfo.k = K;
    TilingAddr->MoeDistributeDispatchShmemInfo.aivNum = BLOCK_NUM;
    return 0;
}

template <class T>
int test_shmem_dispatch(int rank_id)
{
   aclrtStream stream = nullptr;
   aclrtCreateStream(&stream);
    // 初始化
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, DEV_NUM, local_mem_size, ipport, default_flag_uid, &attributes);
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    // Prepare FFTS address
    uint64_t fftsAddr = util_get_ffts_config();
    int aiv_num = BLOCK_NUM;
    void* ptr - aclshmem_calloc(1, aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

//   输入 8*7168
//   专家表 8*1
//   输出 expand_x 
//        expand_idx

    // 设置场景
    // int64_t BS = 8;
    // int64_t H = 7168;
    // int64_t K = 1;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    if (!rank_table_file && !first_rank_id) {
        sharedExpertNum = 1;
        sharedExpertRankNum = 1;
    } 
    int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
    int64_t quantMode = 0;
    int64_t globalBS = BS * EP_WORLD_SIZE;
    int64_t expertTokenNumsType = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        A = globalBS / sharedExpertRankNum;
    } else { 
        // Moe专家卡
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        A = globalBS * (localExpertNum < K ? localExpertNum : K);
    }
    std::string commAlg = "";

    /* 根据当前场景，构造device侧输入输出变量*/
    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr; // 输入
    void *expertIdsDeviceAddr = nullptr; // 专家表输入
    void *scalesDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandXDeviceAddr = nullptr; //输出
    void *dynamicScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    void *tpRecvCountsDeviceAddr = nullptr;
    void *expandScalesDeviceAddr = nullptr;
    void *tilingAddr = nullptr;
    void *workspaceAddr = nullptr;

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> expandXShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A, H};
    std::vector<int64_t> dynamicScalesShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A};
    std::vector<int64_t> expandIdxShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * localExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1};
    std::vector<int64_t> expandScalesShape{A};

    long long xShapeSize = GetShapeSize(xShape);
    long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
    long long scalesShapeSize = GetShapeSize(scalesShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

    // 构造host侧变量
    std::vector<op::fp16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        // 每个token发给moe专家{0, 1, ... k - 1}
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> scalesHostData(scalesShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    // 构造device侧变量
    ret = AllocDeviceRes(xHostData, xShape, &xDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(scalesHostData, scalesShape, &scalesDeviceAddr);  
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(expandXHostData, expandXShape, &expandXDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr);         
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr); 
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AllocDeviceRes(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr);             
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建TilingData
    MoeDistributeDispatchShmemTilingData *tilingData = new MoeDistributeDispatchShmemTilingData();
    ret = InitTilingData(tilingData);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto tilingSize = sizeof(struct MoeDistributeDispatchShmemTilingData);
    ret = aclrtMalloc(&tilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc TilingData failed. ret: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(tilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy TilingData failed. ret: %d\n", ret);
              return ret);
    // Memcpy WorkSpaceSize
    int64_t workspaceSize = gworkSpaceSize;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                  return ret);
    }           
    // shmem buffer
    int aiv_num = BLOCK_NUM;
    void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));
    // kernel调用入口
    moe_dispatch_demo<T>(BLOCK_NUM, stream, fftsAddr, ptr, (uint8_t *)xDeviceAddr, (uint8_t *)expertIdsDeviceAddr, 
        (uint8_t *)scalesDeviceAddr, nullptr, (uint8_t *)expertScalesDeviceAddr, nullptr, nullptr, (uint8_t *)expandXDeviceAddr,
        (uint8_t *)dynamicScalesDeviceAddr ,(uint8_t *)expandIdxDeviceAddr, (uint8_t *)expertTokenNumsDeviceAddr, (uint8_t *)epRecvCountsDeviceAddr,
        (uint8_t *)tpRecvCountsDeviceAddr, (uint8_t *)expandScalesDeviceAddr, (uint8_t*)workspaceAddr, (uint8_t*)tilingAddr);

    status = aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        std::cout << "Case: " << test_cases[i] << " Finised !!" << std::endl;
    }
    if (ptr != nullptr) {
        aclshmem_free(ptr);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (dynamicScalesDeviceAddr != nullptr) {
        aclrtFree(dynamicScalesDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (expertTokenNumsDeviceAddr != nullptr) {
        aclrtFree(expertTokenNumsDeviceAddr);
    }
    if (epRecvCountsDeviceAddr != nullptr) {
        aclrtFree(epRecvCountsDeviceAddr);
    }
    if (expandScalesDeviceAddr != nullptr) {
        aclrtFree(expandScalesDeviceAddr);
    }
    if (tpRecvCountsDeviceAddr != nullptr) {
        aclrtFree(tpRecvCountsDeviceAddr);
    }
    status = aclrtDestroyStream(stream);
    return status;
}


int main(int argc, char *argv[])
{
    int ret;
    std::vector<pid_t> pids(DEV_NUM);
    for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        pid_t pid = fork();
        if(pid == 0) {
            // 子进程
            ret = aclInit(nullptr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            test_shmem_dispatch<int32_t>(rankId);
            aclrtResetDevice(rankId);
            exit(0);
        } else if (pid > 0) {
            // 父进程
            pids[rankId] = pid;
        } else {
            std::cerr << "Fork failed" << std::endl;
            exit(1);
        }
    }
    int status;
    for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        waitpid(pids[rankId], &status, 0);
        if (WIFEXITED(status)) {
            LOG_PRINT("[INFO] Rank %d (PID %d) exited with status %d\n", 
                      rankId, pids[rankId], WEXITSTATUS(status));
        } else {
            LOG_PRINT("[ERROR] Rank %d (PID %d) exited abnormally\n", rankId, pids[rankId]);
        }
    }
    return 0;
}