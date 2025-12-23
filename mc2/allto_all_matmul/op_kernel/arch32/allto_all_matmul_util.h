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
 * \file allto_all_matmul_util.h
 * \brief
 */
#ifndef ALL_TO_ALL_MATMUL_UTIL
#define ALL_TO_ALL_MATMUL_UTIL

#include "allto_all_matmul_tiling.h"
#include <cstdint>

using namespace AscendC;

constexpr static uint32_t BUFFER_NUM = 2U;                   // 多buf
constexpr static int32_t MAX_BLOCK_COUNT = 2;
constexpr static int32_t FLAG_ZERO_IDX = 0;
constexpr static int32_t FLAG_ONE_IDX = 1;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;
constexpr static int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);
constexpr static uint32_t UB_OFFSET = 97440;  // 根据类型变动这个值   / sizeof(int16_t)

template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = SIZE / sizeof(T);

    static __aicore__ inline size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    static __aicore__ inline bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    static __aicore__ inline size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    static __aicore__ inline size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

template <typename T>
using Block32B = BaseBlock<T, 32>;  // 按照32对齐

template <typename T>
using Block256B = BaseBlock<T, 256>;  // 按照256对齐

template <typename T>
using Block512B = BaseBlock<T, 512>;  // 按照512对齐

class CommBase {
public:
     __aicore__ explicit CommBase(){};

    template <typename T>
    __aicore__ inline void SetArgs(int32_t rank, int32_t rank_size, AlltoAllMatmulTilingData info)
    {
        block_id = GetBlockIdx();
        core_num = GetBlockNum();
        aiv_idx = GetSubBlockIdx();
        core_idx = block_id / GetTaskRation();
        this->rank = rank;
        this->rank_size = rank_size;

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        winContext_ = (__gm__ HcclCombineOpParam *)contextGM0;

        for (int i = 0; i < rank_size; i++) {
            buff[i] = (GM_ADDR)winContext_->windowsIn[i];
        }

        SetTiling(info);

        num_per_rank_m = m0 * p_value;
        mid_output_k_size = k * rank_size;
        peer_mem_k_size = Block512B<T>::AlignUp(mid_output_k_size);
        k_align = k;

        total_data_size = 1LL * m * k_align; // 矩阵A大小
        data_size_per_rank = total_data_size / rank_size; // 搬运到每个rank的数据量
        num_per_rank_move = num_per_rank_m * k_align; // 一次通信搬运到每个rank的数据量
        num_per_move = num_per_rank_move * rank_size; // 一次通信搬运的数据量
        peer_mem_block_size = num_per_rank_m * peer_mem_k_size; // pingpong缓冲区大小
        data_per_core = num_per_move / first_step_core_num; // 每个core搬运的数据量
        core_num_per_rank = first_step_core_num / rank_size; // 每个rank的core数量
        cal_count = DivCeil(total_data_size, num_per_move); // 总共需要计算的次数
        core_count = first_step_core_num + second_step_core_num; // 总共的core数量

        total_data_size_block_align = cal_count * peer_mem_block_size;
        used_peer_mem_size = total_data_size_block_align < FLAG_OFFSET ? total_data_size_block_align : FLAG_OFFSET;
        peer_mem_block_count = MAX_BLOCK_COUNT;

        if ASCEND_IS_AIV {
            TPipe pipe;
            pipe.InitBuffer(uBuf_, USED_UB_SIZE);
            pipe.InitBuffer(uBufSync_, 128);
            pipe.Destroy();
        }
    }

    __aicore__ inline void SetTiling(AlltoAllMatmulTilingData& info)
    {
        m = info.allToAllMatmulInfo.M;
        k = info.allToAllMatmulInfo.K;
        n = info.allToAllMatmulInfo.N;
        m0 = info.cocTiling.m0;
        k0 = info.cocTiling.k0;
        n0 = info.cocTiling.n0;
        first_step_core_num = info.cocTiling.first_step_core_num;
        second_step_core_num = info.cocTiling.second_step_core_num;
        swizzl_count = info.cocTiling.swizzlCount;
        swizzl_direct =info.cocTiling.swizzlDirect;
        p_value = info.cocTiling.pValue;

        max_ub_ping_pong_size = info.cocTiling.ubMoveNum / 2;
    }

    __aicore__ inline void AlignJudge(bool trans_a, bool trans_b, int32_t m, int32_t k, int32_t n, int32_t m_align,
                                    int32_t k_align, int32_t n_align, int32_t &aligned_a, int32_t &aligned_b)
    {
        if (!trans_a) {
            aligned_a = k != k_align;
        } else {
            aligned_a = (m != m_align && m != 1);
        }

        if (!trans_b) {
            aligned_b = (n != n_align);
        } else {
            aligned_b = (k != k_align);
        }
    }

    template <typename T>
    __aicore__ inline void CopyUbufToGm(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst, uint16_t lenBurst,
                                        uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyParams dataCopyParams(nBurst,     // blockCount
                                    lenBurst,   // blockLen
                                    srcStride,  // srcStride
                                    dstStride   // dstStride
        );
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopy(gmTensor, ubTensor, dataCopyParams);
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbuf(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                        uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyParams dataCopyParams(nBurst,     // blockCount
                                    lenBurst,   // blockLen
                                    srcStride,  // srcStride
                                    dstStride   // dstStride
        );
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopy(ubTensor, gmTensor, dataCopyParams);
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbufAlignB16(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                                uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,     // blockCount
                                        lenBurst,   // blockLen
                                        srcStride,  // srcStride
                                        dstStride,  // dstStride
                                        0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToGmAlignB16(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst, uint32_t lenBurst,
                                                uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,     // blockCount
                                        lenBurst,   // blockLen
                                        srcStride,  // srcStride
                                        dstStride,  // dstStride
                                        0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }

    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = uBufSync_.AllocTensor<int32_t>();
        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) == flag) {
                break;
            }
        }
        uBufSync_.FreeTensor<int32_t>(ubTensor);
    }

    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = uBufSync_.AllocTensor<int32_t>();
        ubTensor(0) = flag;
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    }

    template <typename T>
    __aicore__ inline void MoveResultFromSrcToPeerMem(__gm__ T *gm_src, __gm__ T *gm_dst, int32_t len)
    {
        LocalTensor<T> ubTensor = uBuf_.AllocTensor<T>();
        LocalTensor<T> copyTensor0 = ubTensor;
        LocalTensor<T> copyTensor1 = ubTensor[ub_offset];
        int32_t len_burst = k * sizeof(T) / 32;
        int32_t src_stride = (k_align - k) * sizeof(T) / 32;
        int32_t dst_stride = (peer_mem_k_size - k) * sizeof(T) / 32;
        int32_t ub_ping_pong_size = max_ub_ping_pong_size / k_align * k_align;
        int32_t ping_pong_move_count = (len + ub_ping_pong_size - 1) / ub_ping_pong_size;
        int32_t actual_move_size = ub_ping_pong_size;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
            if (move_idx == ping_pong_move_count - 1) {
                actual_move_size = len - move_idx * ub_ping_pong_size;
            }
            auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            CopyGmToUbuf(copyTensor, gm_src, 1, actual_move_size * sizeof(T) / 32, 0, 0);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            CopyUbufToGm(gm_dst, copyTensor, actual_move_size / k_align, len_burst, src_stride, dst_stride);
            gm_dst += ub_ping_pong_size * rank_size;
            gm_src += ub_ping_pong_size;
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        uBuf_.FreeTensor<T>(ubTensor);
    }

    template <typename T>
    __aicore__ inline void MoveResultFromPeerMemToOutput(__gm__ T *gm_src, __gm__ T *gm_dst, int32_t len)
    {
        LocalTensor<T> ubTensor = uBuf_.AllocTensor<T>();
        LocalTensor<T> copyTensor0 = ubTensor;
        LocalTensor<T> copyTensor1 = ubTensor[ub_offset];
        int32_t len_burst = mid_output_k_size * sizeof(T) / 32;
        int32_t src_stride = (peer_mem_k_size - mid_output_k_size) * sizeof(T) / 32;
        int32_t total_m = len / peer_mem_k_size;
        int32_t move_m = max_ub_ping_pong_size > peer_mem_k_size ? max_ub_ping_pong_size / peer_mem_k_size : 1;
        int32_t src_k = peer_mem_k_size;
        src_stride = move_m > 1 ? src_stride : 0;
        int32_t actual_move_size = move_m;
        int32_t ping_pong_move_count = (total_m + move_m - 1) / move_m;
        int32_t last_move_size = total_m - (ping_pong_move_count - 1) * move_m;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
            if (move_idx == ping_pong_move_count - 1) {
                actual_move_size = last_move_size;
            }
            auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            CopyGmToUbuf(copyTensor, gm_src, 1, actual_move_size * src_k * sizeof(T) / 32, 0, 0);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            CopyUbufToGm(gm_dst, copyTensor, actual_move_size, len_burst, src_stride, 0);
            gm_dst += move_m * mid_output_k_size;
            gm_src += move_m * peer_mem_k_size;
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (core_idx == 0 && aiv_idx == 0){
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, 0);
            }
        }
    }

    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (core_idx == 0 && aiv_idx == 0) {
            SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flag_idx, flag_data);
        }
        if (core_idx < rank_size && aiv_idx == 0) {
            CheckBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + flag_idx, flag_data);
        }
    }

public:
    int32_t block_id;
    int32_t core_num;
    int32_t aiv_idx;
    int32_t core_idx;
    int32_t rank;
    int32_t rank_size;
    GM_ADDR buff[8];
    __gm__ HcclCombineOpParam *winContext_{nullptr};
    TBuf<AscendC::TPosition::VECCALC> uBuf_;
    TBuf<AscendC::TPosition::VECCALC> uBufSync_;

    int64_t total_data_size;
    int64_t data_size_per_rank;
    int32_t data_per_core;
    int32_t used_peer_mem_size;
    int32_t used_peer_mem_count;
    int32_t peer_mem_block_size;
    int32_t peer_mem_block_count;
    int32_t num_per_rank_m;
    int32_t num_per_rank_move;
    int32_t num_per_move;
    int32_t cal_count;
    int32_t core_count;
    int32_t total_data_size_block_align;
    int32_t core_num_per_rank;
    int32_t peer_mem_k_size;
    int32_t mid_output_k_size;
    int32_t first_step_core_num;
    int32_t second_step_core_num; 
    int32_t ub_offset;

    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t swizzl_count;
    int32_t swizzl_direct;
    int32_t p_value;
    int32_t max_ub_ping_pong_size;

    uint32_t m;
    uint32_t k;
    uint32_t n;

    int32_t k_align;
};

__aicore__ inline void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
{
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(flag_idx + pipe_depth);
    AscendC::CrossCoreWaitFlag(flag_idx + pipe_depth);
}

__aicore__ inline void SetAicSync(uint64_t flag_idx)
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag_idx);
}
#endif