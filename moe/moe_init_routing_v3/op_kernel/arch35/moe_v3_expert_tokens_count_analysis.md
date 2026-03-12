# MoE Expert Tokens Count 代码分析与SIMD重构

## 一、原始文件代码

**文件路径**: `D:\huawei\opt\ops-transformer\moe\moe_init_routing_v3\op_kernel\arch35\moe_v3_expert_tokens_count.h`

```cpp
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
 * \file moe_v3_expert_tokens_count.h
 * \brief
 */
#ifndef MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE
#define MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t KEY_VALUE_MODE = 2LL;
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2LL;

class ExpertTokensCount {
public:
    __aicore__ inline ExpertTokensCount(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();

private:
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    const MoeV3Arch35ExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertTokensNumType_ = 0;
    int64_t expertCountElements_ = 0;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t *sortedExpertIdGmAddr,
    __local_mem__ int32_t *expertFirstIndexLocalAddr)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertCountOutSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t *sortedExpertIdGmAddr,
    __local_mem__ int32_t *expertFirstIndexLocalAddr, __local_mem__ int32_t *expertCountOutLocalAddr)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        if (i == elementNum - 1 || currExpertId != sortedExpertIdGmAddr[i + 1]) {
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}

__aicore__ inline void ExpertTokensCount::Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                               const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertNum_ = tilingData->expertNum;
    expertTokensNumType_ = tilingData->expertTokensNumType;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    } else {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    }

    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) ? (actualExpertNum_ + 1) * KEY_VALUE_MODE_DIM_NUM :
                                                                       expertNum_ * KEY_VALUE_MODE_DIM_NUM;
    } else {
        expertCountElements_ = actualExpertNum_;
    }

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCount, expertCountElements_);
    expertCountTempGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, actualExpertNum_);
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        actualExpertNum_);

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreElements_);
    if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
        InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
    pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
    pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}

__aicore__ inline void ExpertTokensCount::Process()
{
    if (blockIdx_ < needCoreNum_) {
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
        Duplicate(expertCountOutLocal, 0, actualExpertNum_);

        __gm__ int32_t *sortedExpertIdxGmAddr = (__gm__ int32_t *)sortedExpertIdxGm_.GetPhyAddr();
        __local_mem__ int32_t *expertCountOutLocalAddr = (__local_mem__ int32_t *)expertCountOutLocal.GetPhyAddr();

        Simt::VF_CALL<ComputeExpertFirstIndexSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                   expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
        Simt::VF_CALL<ComputeExpertCountOutSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                 expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr,
                                                 expertCountOutLocalAddr);

        expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
        CopyOut();
    }

    SyncAll();
    /* copy expert tokens count result from worksapce to output GM. */
    if (blockIdx_ == 0) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
    SyncAll();
}

__aicore__ inline void ExpertTokensCount::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();
    expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyIn()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
    expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
    event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        int64_t expertOffset = 0;
        Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertCountElements_ * KEY_VALUE_MODE));
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            if (expertCount != 0) {
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM, i + expertStart_);
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM + 1, expertCount);
                expertOffset++;
                actualExpertTotalNum_ += expertCount;
            }
        }
        expertCountElements_ = Min(expertCountElements_, static_cast<int64_t>((expertOffset + 1) * KEY_VALUE_MODE));
    } else {
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            expertCountOutLocal.SetValue(i, expertCount);
            actualExpertTotalNum_ += expertCount;
        }
    }
    expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
    expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    copyParams.blockLen = sizeof(int32_t);
    DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
    expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
    expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE
```

---

## 二、逐行代码解释

### 文件头和命名空间 (1-26行)

```cpp
// 1-9行：版权声明，华为CANN开源软件许可证
// 11-14行：文件说明，防止重复包含的宏定义

// 24-25行：定义常量
constexpr int64_t KEY_VALUE_MODE = 2LL;           // 键值对模式标识
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2LL;   // 键值对模式的维度数(每个专家输出2个值)
```

### ExpertTokensCount类定义 (27-70行)

```cpp
// 27-70行：类声明
class ExpertTokensCount {
public:
    __aicore__ inline ExpertTokensCount(){};  // 构造函数
    __aicore__ inline void Init(...);          // 初始化函数
    __aicore__ inline void Process();          // 主处理函数

private:
    __aicore__ inline void CopyOut();          // 拷贝结果到全局内存

    __aicore__ inline void expertCountCopyIn();   // 读入专家计数中间结果
    __aicore__ inline void expertCountCompute();  // 计算最终专家计数
    __aicore__ inline void expertCountCopyOut(); // 输出最终结果

private:
    // 全局内存张量 (GM = Global Memory)
    GlobalTensor<int32_t> sortedExpertIdxGm_;      // 排序后的专家索引
    GlobalTensor<int32_t> expertCountTempGm_;      // 专家计数临时存储
    GlobalTensor<int64_t> expertTokensCountGm_;    // 最终输出的专家token计数
    GlobalTensor<int32_t> expertTotalCountGm_;     // 专家总计数
    GlobalTensor<int32_t> expandedRowIdxGm_;        // 扩展的行索引
    TPipe *pipe_;                                   // 管道指针

    // 队列 (VECIN = 向量输入, VECOUT = 向量输出)
    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;       // 排序专家索引输入队列
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;   // 专家计数输出到临时队列
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;       // 专家计数临时输入队列
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;      // 专家索引计数输出队列
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;       // 专家总计数队列

    // Tiling数据和计算参数
    const MoeV3Arch35ExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t blockIdx_;                // 当前核的索引
    int64_t needCoreNum_;              // 需要的核数量
    int64_t perCoreElements_;          // 每个核处理的元素数
    int64_t curCoreElements_ = 0;      // 当前核处理的元素数
    int64_t expertStart_ = 0;          // 专家起始索引
    int64_t expertEnd_ = 0;            // 专家结束索引
    int64_t actualExpertNum_ = 0;      // 实际专家数量
    int64_t coreLoopsNum_ = 0;         // 核循环次数
    int64_t perCorePerLoopElements_ = 0;   // 每核每循环元素数
    int64_t perCoreLastLoopElements_ = 0;  // 每核最后循环元素数
    int64_t actualExpertTotalNum_ = 0;    // 实际专家总数
    int64_t expertNum_ = 0;            // 专家总数
    int64_t expertTokensNumType_ = 0;  // 专家token数量类型
    int64_t expertCountElements_ = 0;  // 专家计数元素数
};
```

### SIMT向量函数 (72-106行)

```cpp
// 72-88行：计算每个专家首次出现的索引
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum,              // 元素总数
    int32_t expertStart,             // 专家起始索引
    int32_t expertEnd,               // 专家结束索引
    __gm__ int32_t *sortedExpertIdGmAddr,      // 排序后的专家ID数组(全局内存)
    __local_mem__ int32_t *expertFirstIndexLocalAddr)  // 输出:每个专家首次出现的索引(本地内存)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());  // 获取线程索引
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());  // 获取线程总数
    
    // 多线程并行遍历，每个线程处理 stride=threadNum 的元素
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];  // 当前专家ID
        if (currExpertId >= expertEnd) {
            break;  // 超出当前核负责的专家范围，退出
        }
        // 获取前一个专家ID(第一个元素前一个视为-1)
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        
        // 如果当前专家ID与前一个不同，说明是新的专家开始位置
        if (currExpertId != prevExpertId) {
            // 记录该专家首次出现的索引
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

// 90-106行：计算每个专家处理的token数量
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertCountOutSimt(
    int32_t elementNum,
    int32_t expertStart,
    int32_t expertEnd,
    __gm__ int32_t *sortedExpertIdGmAddr,       // 排序后的专家ID数组
    __local_mem__ int32_t *expertFirstIndexLocalAddr,  // 每个专家首次出现的索引
    __local_mem__ int32_t *expertCountOutLocalAddr)    // 输出:每个专家的token计数
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        // 如果是最后一个元素，或者当前专家ID与下一个不同
        // 说明这是该专家的最后一个token
        if (i == elementNum - 1 || currExpertId != sortedExpertIdGmAddr[i + 1]) {
            // 计算该专家的token数量 = 最后索引 + 1 - 首次出现索引
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}
```

### Init函数 (108-162行)

```cpp
__aicore__ inline void ExpertTokensCount::Init(
    GM_ADDR expandedRowIdx,        // 扩展行索引地址
    GM_ADDR expertTokensCount,     // 输出:专家token计数地址
    GM_ADDR workspace,            // 工作空间地址
    const MoeInitRoutingV3Arch35TilingData *tilingData,  // Tiling数据
    TPipe *tPipe)                 // 管道指针
{
    // 111-120行：初始化基本参数
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();           // 获取当前核索引
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertNum_ = tilingData->expertNum;
    expertTokensNumType_ = tilingData->expertTokensNumType;

    // 122-132行：根据是否是最后一个核，设置不同的处理参数
    if (blockIdx_ == needCoreNum_ - 1) {
        // 最后一个核可能处理不同数量的元素
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    } else {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    }

    // 134-139行：根据模式计算输出元素数
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        // 键值对模式: 输出[expert_id, count]对，所以维度x2
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) 
            ? (actualExpertNum_ + 1) * KEY_VALUE_MODE_DIM_NUM 
            : expertNum_ * KEY_VALUE_MODE_DIM_NUM;
    } else {
        expertCountElements_ = actualExpertNum_;
    }

    // 141-148行：设置全局内存缓冲区
    // 排序后的专家索引(每个核处理自己的部分)
    sortedExpertIdxGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
    // 最终输出
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCount, expertCountElements_);
    // 临时计数存储(所有核共享，用于累加)
    expertCountTempGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, 
        actualExpertNum_);
    // 专家总计数
    expertTotalCountGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace +
            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
            Align(actualExpertNum_, sizeof(int32_t)),
        actualExpertNum_);

    // 150-154行：初始化扩展行索引
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreElements_);
    if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
        InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);  // 初始化为-1
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);   // 等待内存同步
    }

    // 156-161行：初始化队列缓冲区
    int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
    pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
    pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}
```

### Process函数 (164-191行)

```cpp
__aicore__ inline void ExpertTokensCount::Process()
{
    // 166-180行：每个核并行计算自己负责的专家计数
    if (blockIdx_ < needCoreNum_) {
        // 分配本地张量并初始化为0
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
        Duplicate(expertCountOutLocal, 0, actualExpertNum_);

        // 获取物理地址用于SIMT调用
        __gm__ int32_t *sortedExpertIdxGmAddr = (__gm__ int32_t *)sortedExpertIdxGm_.GetPhyAddr();
        __local_mem__ int32_t *expertCountOutLocalAddr = (__local_mem__ int32_t *)expertCountOutLocal.GetPhyAddr();

        // 调用SIMT向量函数并行计算
        // 第一步: 计算每个专家首次出现的索引
        Simt::VF_CALL<ComputeExpertFirstIndexSimt>(
            Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, 
            curCoreElements_, expertStart_, expertEnd_, 
            sortedExpertIdxGmAddr, expertCountOutLocalAddr);
        
        // 第二步: 计算每个专家的token数量
        Simt::VF_CALL<ComputeExpertCountOutSimt>(
            Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, 
            curCoreElements_, expertStart_, expertEnd_, 
            sortedExpertIdxGmAddr, expertCountOutLocalAddr, expertCountOutLocalAddr);

        expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
        CopyOut();  // 将结果写回全局内存(原子加)
    }

    SyncAll();  // 同步所有核
    
    // 184-189行：由核0汇总所有核的结果
    if (blockIdx_ == 0) {
        expertCountCopyIn();    // 读入所有核累加后的临时结果
        expertCountCompute();   // 计算最终输出格式
        expertCountCopyOut();   // 输出最终结果
    }
    SyncAll();
}
```

### CopyOut函数 (193-203行)

```cpp
__aicore__ inline void ExpertTokensCount::CopyOut()
{
    // 从队列取出本地张量
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

    // 设置数据拷贝参数
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1),                    // burst数量
        static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),  // 数据长度
        0, 0, 0
    };
    
    SetAtomicAdd<int32_t>();  // 设置原子加操作(多核并行写入需要原子操作)
    DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);  // 拷贝到全局内存
    SetAtomicNone();          // 关闭原子操作
    
    expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);  // 释放张量
}
```

### expertCountCopyIn函数 (205-214行)

```cpp
__aicore__ inline void ExpertTokensCount::expertCountCopyIn()
{
    // 分配本地张量
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

    // 从全局内存拷贝到本地内存
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 
        0, 0, 0
    };
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
    
    expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}
```

### expertCountCompute函数 (216-255行)

```cpp
__aicore__ inline void ExpertTokensCount::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
    
    // 设置事件同步(MTE2->S: 内存到标量单元)
    event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        // 键值对模式: 输出 [expert_id, count] 对
        int64_t expertOffset = 0;
        // 初始化输出为0
        Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertCountElements_ * KEY_VALUE_MODE));
        
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        
        // 遍历每个专家，只输出有token的专家
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            if (expertCount != 0) {
                // 写入 [expert_id, count]
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM, i + expertStart_);
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM + 1, expertCount);
                expertOffset++;
                actualExpertTotalNum_ += expertCount;  // 累加总数
            }
        }
        // 更新实际输出的元素数
        expertCountElements_ = Min(expertCountElements_, static_cast<int64_t>((expertOffset + 1) * KEY_VALUE_MODE));
    } else {
        // 默认模式: 直接输出每个专家的计数
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            expertCountOutLocal.SetValue(i, expertCount);
            actualExpertTotalNum_ += expertCount;
        }
    }
    
    // 保存总计数
    expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
    
    // 设置事件同步(S->MTE3: 标量单元到内存)
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    
    expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
    expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}
```

### expertCountCopyOut函数 (257-268行)

```cpp
__aicore__ inline void ExpertTokensCount::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
    
    // 拷贝专家计数结果
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 
        0, 0, 0
    };
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    
    // 拷贝总计数
    copyParams.blockLen = sizeof(int32_t);
    DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
    
    expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
    expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
}
```

---

## 三、总结

**整体流程**：
1. **Init**: 初始化内存缓冲区、队列和参数
2. **Process**: 
   - 多核并行执行SIMT函数，统计各自负责的专家token数
   - 使用原子加将各核结果累加到全局临时存储
   - 核0汇总并输出最终结果

**核心算法**：
- 输入是已排序的专家ID数组（相同专家ID连续排列）
- 通过检测专家ID变化点，计算每个专家处理的token数量
- 支持两种输出格式：直接计数 或 `[expert_id, count]` 键值对

---

## 四、SIMT与SIMD的区别

| 特性 | SIMD (Single Instruction Multiple Data) | SIMT (Single Instruction Multiple Threads) |
|------|----------------------------------------|-------------------------------------------|
| **执行模型** | 单指令同时操作多个数据元素 | 单指令由多个线程并行执行 |
| **硬件单元** | CPU向量单元（如AVX、NEON） | GPU流式多处理器 |
| **线程独立性** | 无线程概念，纯数据并行 | 每个线程有独立寄存器状态 |
| **分支处理** | 分支导致部分计算单元空闲 | Warp内分支导致线程发散（divergence） |
| **编程模型** | 向量化指令/intrinsics | CUDA/昇腾核函数 |

**在昇腾架构中的应用**：

```cpp
// SIMT使用示例：
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) 
inline void ComputeExpertFirstIndexSimt(...)
{
    auto threadIdx = Simt::GetThreadIdx();  // 每个线程有独立ID
    auto threadNum = Simt::GetThreadNum();   // 总线程数
    
    // 每个线程处理不同的数据索引
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        // 线程独立执行...
    }
}
```

**简单理解**：
- **SIMD**：一条指令 = 多个数据并行操作（CPU向量化）
- **SIMT**：一条指令 = 多个线程同时执行，每个线程处理自己的数据（GPU风格）

---

## 五、SIMD重构版本代码

**文件路径**: `D:\huawei\opt\ops-transformer\moe\moe_init_routing_v3\op_kernel\arch35\moe_v3_expert_tokens_count_simd.h`

```cpp
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
 * \file moe_v3_expert_tokens_count_simd.h
 * \brief SIMD version of expert tokens count
 */
#ifndef MOE_V3_EXPERT_TOKENS_COUNT_SIMD_H_REGBASE
#define MOE_V3_EXPERT_TOKENS_COUNT_SIMD_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t KEY_VALUE_MODE = 2LL;
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2LL;
constexpr int32_t EXPERT_INDEX_NOT_FOUND = -1;

class ExpertTokensCountSimd {
public:
    __aicore__ inline ExpertTokensCountSimd(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut();
    __aicore__ inline void ComputeExpertFirstIndex();
    __aicore__ inline void ComputeExpertCountOut();
    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();

private:
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    const MoeV3Arch35ExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertTokensNumType_ = 0;
    int64_t expertCountElements_ = 0;
};

__aicore__ inline void ExpertTokensCountSimd::ComputeExpertFirstIndex()
{
    LocalTensor<int32_t> expertFirstIndexLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> sortedExpertIdxLocal = sortedExpertIdxInQueue_.AllocTensor<int32_t>();
    
    int64_t expertRange = expertEnd_ - expertStart_;
    Duplicate(expertFirstIndexLocal, EXPERT_INDEX_NOT_FOUND, expertRange);
    
    int64_t totalElements = curCoreElements_;
    int64_t elementsPerLoop = sortedExpertIdxInQueue_.GetTensorSize() / sizeof(int32_t);
    int64_t loopCount = (totalElements + elementsPerLoop - 1) / elementsPerLoop;
    
    int32_t prevExpertId = -1;
    
    for (int64_t loop = 0; loop < loopCount; loop++) {
        int64_t curLoopElements = (loop == loopCount - 1) 
            ? (totalElements - loop * elementsPerLoop) 
            : elementsPerLoop;
        
        int64_t offset = loop * elementsPerLoop;
        
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), 
            static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams padParams{false, 0, 0, 0};
        DataCopyPad(sortedExpertIdxLocal, sortedExpertIdxGm_[offset], copyParams, padParams);
        
        pipe_->Barrier(PIPE_MTE2, PIPE_V);
        
        for (int64_t i = 0; i < curLoopElements; i++) {
            int32_t currExpertId = sortedExpertIdxLocal.GetValue(i);
            
            if (currExpertId >= expertEnd_) {
                break;
            }
            
            if (currExpertId != prevExpertId) {
                expertFirstIndexLocal.SetValue(currExpertId - expertStart_, 
                    static_cast<int32_t>(offset + i));
                prevExpertId = currExpertId;
            }
        }
    }
    
    sortedExpertIdxInQueue_.FreeTensor(sortedExpertIdxLocal);
    expertCountOutToTempQueue_.EnQue<int32_t>(expertFirstIndexLocal);
}

__aicore__ inline void ExpertTokensCountSimd::ComputeExpertCountOut()
{
    LocalTensor<int32_t> expertFirstIndexLocal = expertCountOutToTempQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> sortedExpertIdxLocal = sortedExpertIdxInQueue_.AllocTensor<int32_t>();
    
    int64_t expertRange = expertEnd_ - expertStart_;
    Duplicate(expertCountOutLocal, static_cast<int32_t>(0), expertRange);
    
    int64_t totalElements = curCoreElements_;
    int64_t elementsPerLoop = sortedExpertIdxInQueue_.GetTensorSize() / sizeof(int32_t);
    int64_t loopCount = (totalElements + elementsPerLoop - 1) / elementsPerLoop;
    
    int32_t prevFirstIndex = EXPERT_INDEX_NOT_FOUND;
    int32_t prevExpertId = -1;
    
    for (int64_t loop = 0; loop < loopCount; loop++) {
        int64_t curLoopElements = (loop == loopCount - 1) 
            ? (totalElements - loop * elementsPerLoop) 
            : elementsPerLoop;
        
        int64_t offset = loop * elementsPerLoop;
        
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), 
            static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams padParams{false, 0, 0, 0};
        DataCopyPad(sortedExpertIdxLocal, sortedExpertIdxGm_[offset], copyParams, padParams);
        
        pipe_->Barrier(PIPE_MTE2, PIPE_V);
        
        for (int64_t i = 0; i < curLoopElements; i++) {
            int32_t currExpertId = sortedExpertIdxLocal.GetValue(i);
            int64_t globalIdx = offset + i;
            
            if (currExpertId >= expertEnd_) {
                break;
            }
            
            if (prevExpertId != -1 && prevExpertId != currExpertId) {
                if (prevFirstIndex != EXPERT_INDEX_NOT_FOUND) {
                    int32_t count = static_cast<int32_t>(globalIdx - prevFirstIndex);
                    expertCountOutLocal.SetValue(prevExpertId - expertStart_, count);
                }
                prevFirstIndex = expertFirstIndexLocal.GetValue(currExpertId - expertStart_);
            }
            
            if (prevExpertId == -1) {
                prevFirstIndex = expertFirstIndexLocal.GetValue(currExpertId - expertStart_);
            }
            
            if (globalIdx == totalElements - 1 || 
                (i < curLoopElements - 1 && currExpertId != sortedExpertIdxLocal.GetValue(i + 1)) ||
                (i == curLoopElements - 1 && loop == loopCount - 1)) {
                
                int32_t nextExpertId = (globalIdx == totalElements - 1) 
                    ? -1 
                    : sortedExpertIdxLocal.GetValue(i + 1);
                
                if (currExpertId != nextExpertId) {
                    int32_t firstIdx = expertFirstIndexLocal.GetValue(currExpertId - expertStart_);
                    if (firstIdx != EXPERT_INDEX_NOT_FOUND) {
                        int32_t count = static_cast<int32_t>(globalIdx + 1 - firstIdx);
                        expertCountOutLocal.SetValue(currExpertId - expertStart_, count);
                    }
                }
            }
            
            prevExpertId = currExpertId;
        }
    }
    
    sortedExpertIdxInQueue_.FreeTensor(sortedExpertIdxLocal);
    expertCountOutToTempQueue_.FreeTensor(expertFirstIndexLocal);
    expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCountSimd::Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, 
                                                    GM_ADDR workspace,
                                                    const MoeInitRoutingV3Arch35TilingData *tilingData, 
                                                    TPipe *tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertNum_ = tilingData->expertNum;
    expertTokensNumType_ = tilingData->expertTokensNumType;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    } else {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    }

    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) 
            ? (actualExpertNum_ + 1) * KEY_VALUE_MODE_DIM_NUM 
            : expertNum_ * KEY_VALUE_MODE_DIM_NUM;
    } else {
        expertCountElements_ = actualExpertNum_;
    }

    sortedExpertIdxGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCount, expertCountElements_);
    expertCountTempGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, 
        actualExpertNum_);
    expertTotalCountGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace +
            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
            Align(actualExpertNum_, sizeof(int32_t)),
        actualExpertNum_);

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreElements_);
    if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
        InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
    pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
    pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}

__aicore__ inline void ExpertTokensCountSimd::Process()
{
    if (blockIdx_ < needCoreNum_) {
        ComputeExpertFirstIndex();
        ComputeExpertCountOut();
        CopyOut();
    }

    SyncAll();
    
    if (blockIdx_ == 0) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
    SyncAll();
}

__aicore__ inline void ExpertTokensCountSimd::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), 
        static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();
    expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCountSimd::expertCountCopyIn()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
    expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCountSimd::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
    
    event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        int64_t expertOffset = 0;
        Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertCountElements_ * KEY_VALUE_MODE));
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            if (expertCount != 0) {
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM, i + expertStart_);
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM + 1, expertCount);
                expertOffset++;
                actualExpertTotalNum_ += expertCount;
            }
        }
        expertCountElements_ = Min(expertCountElements_, 
            static_cast<int64_t>((expertOffset + 1) * KEY_VALUE_MODE));
    } else {
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            expertCountOutLocal.SetValue(i, expertCount);
            actualExpertTotalNum_ += expertCount;
        }
    }
    
    expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
    
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    
    expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
    expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCountSimd::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
    
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    
    copyParams.blockLen = sizeof(int32_t);
    DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
    
    expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
    expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_EXPERT_TOKENS_COUNT_SIMD_H_REGBASE
```

---

## 六、SIMT版本与SIMD版本对比

| 方面 | SIMT版本 | SIMD版本 |
|------|----------|----------|
| **函数定义** | `__simt_vf__` 标记，独立函数 | 类成员函数 |
| **线程模型** | `Simt::GetThreadIdx()`, 多线程并行 | 单线程循环 + 向量数据搬运 |
| **调用方式** | `Simt::VF_CALL<函数名>()` | 直接调用成员函数 |
| **内存访问** | 直接访问全局内存 `__gm__` | 通过 `LocalTensor` 和 `DataCopyPad` |
| **同步机制** | SIMT隐式同步 | `pipe_->Barrier()` 显式同步 |

### 核心变化说明

**SIMT版本**：
```cpp
__simt_vf__ inline void ComputeExpertFirstIndexSimt(...) {
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        // 多线程并行处理
    }
}
```

**SIMD版本**：
```cpp
__aicore__ inline void ComputeExpertFirstIndex() {
    // 分块加载数据到本地内存
    DataCopyPad(sortedExpertIdxLocal, sortedExpertIdxGm_[offset], ...);
    pipe_->Barrier(PIPE_MTE2, PIPE_V);
    // 标量循环处理
    for (int64_t i = 0; i < curLoopElements; i++) { ... }
}
```

---

## 七、关键修改点

1. **移除SIMT相关修饰符**：
   - 删除 `__simt_vf__` 和 `LAUNCH_BOUND(SIMT_THREAD_NUM)`
   - 删除 `__gm__` 和 `__local_mem__` 地址修饰符

2. **线程模型转换**：
   - SIMT：多线程并行，每个线程处理不同索引
   - SIMD：单线程分块处理，通过`DataCopyPad`批量搬运数据

3. **数据访问方式**：
   - SIMT：直接访问全局内存地址
   - SIMD：通过`LocalTensor`和队列管理内存

4. **同步机制**：
   - SIMT：线程自动同步
   - SIMD：使用`pipe_->Barrier()`显式同步流水线

5. **新增常量**：
   - `EXPERT_INDEX_NOT_FOUND = -1`：用于标识未找到的专家索引