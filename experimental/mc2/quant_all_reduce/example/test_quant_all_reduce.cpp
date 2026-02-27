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
 * \file test_aclnn_quant_all_reduce.cpp
 * \brief <<<>>>测试样例
 */
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <fstream>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "./kernel/quant_all_reduce_tiling_data.h"
#include "./quant_all_reduce.h"

using namespace std;
using namespace AscendC;

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

constexpr int DEV_NUM = 2; // 设备数量
constexpr int64_t BLOCK_NUM = 64;
int rankId = 0;
int streamWithTimeout = 10000;
   
std::string input_tensor_type = "int8_t"; 
std::string scales_type = "float32_t";    
std::string output_type = "float16_t";    
std::string case_name = "int8_t-float32_t-float16_t-ID001";  
int ranksize = 2;                     
int mxfp = 0; 
int run_type = 1;  

// QuantAllReduceTilingInfo
int gbs = 16;
int ghiddenSize = 8192;
int gscaleHiddenSize = 8192;
int gmc2ContextSize = 576;
int64_t ghcclBufferSize = 200;
int64_t gworkSpaceSize = 256*1024*1024;

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

void GetOption(int argc, char **argv)
{
    while (true) {
        int optionIndex = 0;
        struct option longOptions[] = {
            {"rank_id",      1, 0, 'a'},
            {"bs",           1, 0, 'b'},
            {"hidden_size",  1, 0, 'c'},
            {"run_type",     1, 0, 'd'},  
            {"input_tensor_type", 1, 0, 'e'}, 
            {"scales_type",  1, 0, 'f'}, 
            {"output_type",  1, 0, 'g'},  
            {"ranksize",     1, 0, 'h'},  
            {"mxfp",         1, 0, 'i'},  
            {"case_name",    1, 0, 'j'},
            {0, 0, 0, 0}                  
        };
        int c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:", longOptions, &optionIndex);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'a':
                rankId = atoi(optarg);
                LOG_PRINT("[INFO] rankId = %d\n", rankId);
                break;
            case 'b':
                gbs = atoi(optarg);
                LOG_PRINT("[INFO] bs = %d\n", gbs);
                break;
            case 'c':
                ghiddenSize = atoi(optarg);
                LOG_PRINT("[INFO] hidden_size = %d\n", ghiddenSize);
                break;
            case 'd':
                run_type = atoi(optarg);  
                LOG_PRINT("[INFO] run_type = %d\n", run_type);
                break;
            case 'e':
                input_tensor_type = optarg;
                LOG_PRINT("[INFO] input_tensor_type = %s\n", input_tensor_type.c_str());
                break;
            case 'f':
                scales_type = optarg;
                LOG_PRINT("[INFO] scales_type = %s\n", scales_type.c_str());
                break;
            case 'g':
                output_type = optarg;
                LOG_PRINT("[INFO] output_type = %s\n", output_type.c_str());
                break;
            case 'h':
                ranksize = atoi(optarg);
                LOG_PRINT("[INFO] ranksize = %d\n", ranksize);
                break;
            case 'i':
                mxfp = atoi(optarg);
                LOG_PRINT("[INFO] mxfp = %d\n", mxfp);
                break;
            case 'j':
                case_name = optarg;
                LOG_PRINT("[INFO] case_name = %s\n", case_name.c_str());
                break;
            default:
                LOG_PRINT("[WARN] Unknown option: %c\n", c);
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

int ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        printf("Failed to get file\n");
        return -1;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        printf("%s is not a file, please enter a file.\n", filePath.c_str());
        return -1;
    }
    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("Open file failed. path = %s.\n", filePath.c_str());
        return -1;
    }
    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        printf("File size is 0\n");
        file.close();
        return -1;
    }
    if (size > bufferSize) {
        printf("File size is larger than buffer size.\n");
        file.close();
        return -1;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return 0;
}

int WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        printf("Write file failed. Buffer is nullptr.\n");
        return -1;
    }
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        printf("Open file failed. path = %s\n", filePath.c_str());
        return -1;
    }
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }
    // write data
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << strerror(errno) << std::endl;
    }
    flock(fd, LOCK_UN);
    close(fd);
    return 0;
}

template<typename T>
int CreatDeviceInput(const std::vector<int64_t> &shape, void **deviceAddr, std::string dataName) //gaiming
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret);
              return ret);
    uint8_t *input_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), size);
    std::string inputFile = "./golden/quantallreduce_" + std::string(case_name) + "_" + std::to_string(gbs) + "_" +
                            std::to_string(ghiddenSize) + "/input_" + dataName + "_" + std::to_string(rankId) + ".bin";
    ret = ReadFile(inputFile, input_host, size); 
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] ReadFile failed. ret: %d\n", ret);
              return ret);                       
    ret = aclrtMemcpy(*deviceAddr, size, input_host, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret);
              return ret);
    ret = aclrtFreeHost(input_host);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtFreeHost failed. ret: %d\n", ret);
              return ret);
    return 0;
}

int InitTilingData(QuantAllReduceTilingData* QuantAllReduceTilingAddr, std::string hcomName) {
    Mc2InitTiling mc2InitTiling; 
    Mc2CcTiling mc2CcTiling;
    QuantAllReduceTilingAddr->mc2InitTiling = mc2InitTiling;
    QuantAllReduceTilingAddr->mc2CcTiling = mc2CcTiling;
    QuantAllReduceTilingAddr->quantAllReduceTilingInfo.bs = gbs;
    QuantAllReduceTilingAddr->quantAllReduceTilingInfo.hiddenSize = ghiddenSize;
    QuantAllReduceTilingAddr->quantAllReduceTilingInfo.scaleHiddenSize = gscaleHiddenSize;
    QuantAllReduceTilingAddr->quantAllReduceTilingInfo.aivNum = BLOCK_NUM;
    QuantAllReduceTilingAddr->quantAllReduceTilingInfo.totalWinSize = ghcclBufferSize;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);
    int ret = mc2CcTilingConfig.GetTiling(QuantAllReduceTilingAddr->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2InitTiling failed. ret = %d\n", ret);
              return -1);
    ret = mc2CcTilingConfig.GetTiling(QuantAllReduceTilingAddr->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2CcTiling failed. ret = %d\n", ret);
              return -1);
    return 0;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadQuantAllReduce(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret);
              return ret);
    char hcomName[128] = {0}; 
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret);
              return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    // 创建TilingData
    QuantAllReduceTilingData *tilingData = new QuantAllReduceTilingData();
    ret = InitTilingData(tilingData, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建通信域Context
    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(hcomName, &commHandle);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcomGetCommHandleByGroup failed. ret = %d\n", ret);
              return -1);    
    void *mc2Context = nullptr;
    ret = HcclAllocComResourceByTiling(commHandle, args.stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclAllocComResourceByTiling failed. ret = %d\n", ret);
              return -1);
    if (mc2Context == nullptr) {
        LOG_PRINT("[ERROR] mc2Context is nullptr\n");
        return -1;
    }    
    // 创建input
    std::vector<int64_t> xShape = {gbs, ghiddenSize}; // (bs, H)
    std::vector<int64_t> scalesShape;
    if (mxfp == 0) {
        scalesShape = {gbs, ghiddenSize/128}; // (bs, H/128)
    } else {
        scalesShape = {gbs, ghiddenSize/64, 2};
    }
    std::vector<int64_t> outputShape = {gbs, ghiddenSize}; // (bs, H)
    // Device地址
    void *xDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *mc2ContextAddr = nullptr;
    void *tilingAddr = nullptr;
    // Memcpy TilingData
    auto tilingSize = sizeof(struct QuantAllReduceTilingData);
    ret = aclrtMalloc(&tilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc TilingData failed. ret: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(tilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy TilingData failed. ret: %d\n", ret);
              return ret);
    // Memcpy Mc2Context
    int64_t mc2ContextSize = gmc2ContextSize;
    ret = aclrtMalloc(&mc2ContextAddr, mc2ContextSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc Mc2Context failed. ret: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(mc2ContextAddr, mc2ContextSize, mc2Context, mc2ContextSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy Mc2Context failed. ret: %d\n", ret);
              return ret); 
    // Memcpy WorkSpaceSize
    int64_t workspaceSize = gworkSpaceSize;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                  return ret);
    }           
    // Memcpy Input
    ret = CreatDeviceInput<int8_t>(xShape, &xDeviceAddr, "x");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (scales_type == "float32_t") {
        ret = CreatDeviceInput<int32_t>(scalesShape, &scalesDeviceAddr, "scale");
    } else {
        ret = CreatDeviceInput<int8_t>(scalesShape, &scalesDeviceAddr, "scale");
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 分配Output内存
    void *outputDeviceAddr = nullptr;
    long long outputShapeSize = GetShapeSize(outputShape) * sizeof(int16_t);
    if (output_type == "float32_t") {
        outputShapeSize = GetShapeSize(outputShape) * sizeof(int32_t);
    }
    ret = aclrtMalloc(&outputDeviceAddr, outputShapeSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret);
              return ret);
    // <<<>>>调用核函数
    quant_all_reduce_demo(run_type, BLOCK_NUM, args.stream, (uint8_t*)xDeviceAddr, (uint8_t*)scalesDeviceAddr, (uint8_t*)outputDeviceAddr,
                            (uint8_t*)workspaceAddr, (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnQuantAllReduce execute successfully.\n", args.rankId);
    // 写出Output
    int16_t *output_host;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&output_host), outputShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMallocHost failed. ret: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(output_host, outputShapeSize, outputDeviceAddr, outputShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::string outputFile = "./golden/quantallreduce_" + std::string(case_name) + "_" + std::to_string(gbs) + "_" +
                            std::to_string(ghiddenSize) + "/output_npu_" + std::to_string(rankId) + ".bin";
    ret = WriteFile(outputFile, output_host, outputShapeSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] WriteFile failed. ret: %d\n", ret);
              return ret);
    ret = aclrtFreeHost(output_host);    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtFreeHost failed. ret: %d\n", ret);
              return ret);
    // 释放device资源，需要根据具体API的接口定义修改
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (outputDeviceAddr != nullptr) {
        aclrtFree(outputDeviceAddr);
    }
    if (mc2ContextAddr != nullptr) {
        aclrtFree(mc2ContextAddr);
    }
    if (tilingAddr != nullptr) {
        aclrtFree(tilingAddr);
    }    
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclDeterministic = 1;
    config.hcclBufferSize = ghcclBufferSize;
    ret = strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "hccl_comm_test");
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] hcclCommName strcpy failed. ret = %d \n", ret);
              return ret);    
    const char* rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] get rankTableFile failed.\n");
              return -1);
    ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &config, &comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret = %d \n", ret);
              return ret);

    Args args;
    args.rankId = rankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;
    ret = LaunchOneThreadQuantAllReduce(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LaunchOneThreadQuantAllReduce failed. ret = %d \n", ret);
              return ret);
    aclFinalize();
    return 0;
}