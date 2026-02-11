/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_dispatch_log.h
 * \brief
 */

#ifndef MOE_DISPATCH_LOG_H
#define MOE_DISPATCH_LOG_H
namespace MoeDispatchLog {
using namespace AscendC;


__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg, const GM_ADDR addr) {
    uint32_t aivid = GetBlockIdx();
    const __gm__ void * addrtemp = static_cast<const __gm__ void*>(addr);
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s: %p\n", msg ,addrtemp);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg) {
    uint32_t aivid = GetBlockIdx();
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s\n", msg);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg,uint64_t num) {
    uint32_t aivid = GetBlockIdx();
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s:%ld \n", msg, num);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg,uint32_t num) {
    uint32_t aivid = GetBlockIdx();
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s:%d \n", msg, num);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg,int32_t num) {
    uint32_t aivid = GetBlockIdx();
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s:%d \n", msg, num);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg,int64_t num) {
    uint32_t aivid = GetBlockIdx();
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s:%ld \n", msg, num);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg,float num) {
    uint32_t aivid = GetBlockIdx();
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s:%f \n", msg, num);
}

__aicore__ inline void LogInfo(uint32_t line, const __gm__ char *msg,uint32_t num,int32_t rankid) {
    uint32_t aivid = GetBlockIdx();
    printf("[rankid: %d]",rankid);
    printf("[aivid: %d]",aivid);
    printf("[PRINT]");
    printf("[line:%d]", line);
    printf("%s:%d \n", msg, num);
}


template <typename T>
__aicore__ inline void LogInfo(uint32_t line, LocalTensor<T>& tensor,TPipe *tpipe_ = nullptr,uint32_t print_len = 9) {
    // if(tpipe_ == nullptr){
    //     LogInfo(line, "[error: ] local tensor tpipe_ is nullptr");
    //     return;
    // }
    //PipeBarrier<PIPE_ALL>();
    // uint32_t len = tensor.GetSize();
    // uint32_t aivid = GetBlockIdx();
    // printf("[aivid: %d]",aivid);
    // printf("[PRINT]");
    // printf("[line:%d] [start print tensor:%d]\n", line,len);
    // TBuf<> buf;
    // tpipe_->InitBuffer(buf, len * sizeof(float));
    // LocalTensor<float> float_tensor = buf.Get<float>();
    // if constexpr (!std::is_same_v<T,float>) {
    //     Cast(float_tensor, tensor,AscendC::RoundMode::CAST_NONE,len);
    // }
    // for(uint32_t i = 0; i < len; i++) {
    //     if constexpr (!std::is_same_v<T,float>) {
    //        printf("%f ", float_tensor(i)); 
    //     }
    //     else{
    //         printf("%f ",tensor(i));
    //     }
    //     if((i+1) % print_len ==0){
    //         printf("\n");
    //     }
    // }
    // printf("\ncomplete print half\n");  
}

template <>
__aicore__ inline void LogInfo<float>(uint32_t line, LocalTensor<float>& tensor,TPipe *tpipe_,uint32_t print_len) {
    // if(tpipe_ == nullptr){
    //     LogInfo(line, "[error: ] local tensor tpipe_ is nullptr");
    //     return;
    // }
    // uint32_t len = tensor.GetSize();
    // uint32_t aivid = GetBlockIdx();
    // printf("[aivid: %d]",aivid);
    // printf("[PRINT]");
    // printf("[line:%d] [start print tensor:%d]\n", line,len);
    // for(uint32_t i = 0; i < len; i++) {
    //    printf("%f ", tensor(i));
    //     if((i+1) % print_len ==0){
    //         printf("\n");
    //     }
    // }
    // printf("\ncomplete print float\n");  
}

template <>
__aicore__ inline void LogInfo<uint32_t>(uint32_t line, LocalTensor<uint32_t>& tensor,TPipe *tpipe_,uint32_t print_len) {
    // PipeBarrier<PIPE_ALL>();
    // uint32_t aivid = GetBlockIdx();
    // printf("[aivid: %d]",aivid);
    // printf("[PRINT]");
    // printf("[line:%d]\n", line);
    // uint32_t len = tensor.GetSize();
    // printf("[PRINT]");
    // printf("[line:%d] [start print tensor:%d]\n", line,len);
    // for(uint32_t i = 0; i < len; i++) {
    //    printf("%d ", tensor(i));
    //     if((i+1) % print_len ==0){
    //         printf("\n");
    //     }
    // }
    // printf("\ncomplete print uint32_t\n");  
}

template <>
__aicore__ inline void LogInfo<int32_t>(uint32_t line, LocalTensor<int32_t>& tensor,TPipe *tpipe_,uint32_t print_len) {
    // PipeBarrier<PIPE_ALL>();
    // uint32_t aivid = GetBlockIdx();
    // printf("[aivid: %d]",aivid);
    // printf("[PRINT]");
    // printf("[line:%d]\n", line);
    // uint32_t len = tensor.GetSize();
    // printf("[PRINT]");
    // printf("[line:%d] [start print tensor:%d]\n", line,len);
    // for(uint32_t i = 0; i < len; i++) {
    //    printf("%d ", tensor(i));
    //     if((i+1) % print_len ==0){
    //         printf("\n");
    //     }
    // }
    // printf("\ncomplete print int32_t\n");  
}

template <>
__aicore__ inline void LogInfo<uint64_t>(uint32_t line, LocalTensor<uint64_t>& tensor,TPipe *tpipe_,uint32_t print_len) {
    // PipeBarrier<PIPE_ALL>();
    // uint32_t aivid = GetBlockIdx();
    // printf("[aivid: %d]",aivid);
    // printf("[PRINT]");
    // printf("[line:%d]\n", line);
    // uint32_t len = tensor.GetSize();
    // printf("[PRINT]");
    // printf("[line:%d] [start print tensor:%d]\n", line,len);
    // for(uint32_t i = 0; i < len; i++) {
    //    printf("%d ", tensor(i));
    //     if((i+1) % print_len ==0){
    //         printf("\n");
    //     }
    // }
    // printf("\ncomplete print uint64_t\n");  
}

}


#endif // MOE_DISPATCH_LOG_H
