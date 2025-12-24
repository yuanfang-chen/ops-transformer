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
 * \file mte_comm.h
 * \brief quant_all_reduce 与quant_reduce_scatter MTE通信相关 kernel代码公共部分
 */

#ifndef MTE_COMMON_H
#define MTE_COMMON_H

namespace QuantMTECommImpl {

using namespace AscendC;
// 后缀_BYTES 表示单位为字节大小B, _NUM 表示单位为个
constexpr static uint32_t UB_ALIGN_BYTES = 32U;     // UB按32B对齐
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;         // float格式下32B对齐需要 32/4 =8个
constexpr uint32_t BUFFER_NUM = 2U;                 // 用于double buffer
constexpr static uint32_t SCALE_BLCOK_BYTES = 32U;  // 一个scale数据块固定32B，搬运要求32B对齐
constexpr static uint32_t X_BLOCK_BYTES = 1024U;    // 当前一个x数据块固定1024B = 128 * 8(PT) = 32 * 32(MX)，为穿刺取值

#define TemplateTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateType XType, ScalesType, OutputType

template<TemplateTypeClass>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContext();
    __aicore__ inline void InitParams();
    __aicore__ inline void InitGMTensor(GM_ADDR x, GM_ADDR scales, GM_ADDR output, uint64_t alignedXSize);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void SetBlockSize(uint32_t elementsPerBlock);
     template <bool isReduceScatter = false>
    __aicore__ inline void CopyDataToWin(uint64_t xSliceSizeNums = 0, uint64_t scaleSliceNums = 0);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float>& localResultTensor);

    __gm__ HcclA5OpResParam *hcclContext_;
    uint32_t round_{0};
    uint32_t tailBlockNums_{0};
    uint32_t assignedBlockNums_{0};

    uint64_t scaleNumsPerBlcok_{0};
    uint64_t xOffset_{0};  
    uint64_t scaleOffset_{0};
private:
    uint32_t aivId_{0};  
    uint32_t xNumPerBlock_{0};

    __aicore__ inline void CopyDataBlock(uint64_t curXOffset, uint64_t curScaleOffset);

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;
    GlobalTensor<XType> localWinXGMTensor_;
    GlobalTensor<ScalesType> localWinScaleGMTensor_;
    GlobalTensor<OutputType> outputTensor_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<ScalesType> scaleTmpTensor_;
    LocalTensor<float> stateResetTensor_;
    LocalTensor<OutputType> xOutTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_, scaleQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateResetBuf_;
};

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitHcclContext()
{
    hcclContext_ = (__gm__ HcclA5OpResParam*)GetHcclContext<HCCL_GROUP_ID_0>();
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitParams()
{
    aivId_ = GetBlockIdx(); // 获取当前核Id
    scaleNumsPerBlcok_ = SCALE_BLCOK_BYTES / sizeof(ScalesType); // 一块scale固定32B, 计算包含多少个数据
    assignedBlockNums_ = aivId_ < tailBlockNums_ ? round_ + 1 : round_; // 当前核分配到的数据块数量，顺序分核，序号小的核多搬一轮
    uint64_t blockIdx = aivId_ * round_ + (aivId_ < tailBlockNums_ ? aivId_ : tailBlockNums_); // 计算当前核分派到的首个数据块序列号
    xOffset_ = blockIdx * xNumPerBlock_; // 块数 * 一块有多少数据，得到要搬第几个x数据
    scaleOffset_ = blockIdx * scaleNumsPerBlcok_; // 计算要搬第几个 scale
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitGMTensor(GM_ADDR x, GM_ADDR scales, GM_ADDR output, uint64_t alignedXSize)
{
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
    outputTensor_.SetGlobalBuffer((__gm__ OutputType*)output);

    // 通过rankId获取本地winIn区地址对应卡的数据区域
    // 获取本卡地址写数据
    GM_ADDR localDataSpaceGm = (GM_ADDR)(hcclContext_->windowsIn[hcclContext_->rankId]);
    localWinXGMTensor_.SetGlobalBuffer((__gm__ XType*)localDataSpaceGm);
    localWinScaleGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)(localDataSpaceGm + alignedXSize)); // GM上sclae数据跟在x后
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, X_BLOCK_BYTES);    
    tPipe->InitBuffer(scaleQueue_, BUFFER_NUM, UB_ALIGN_BYTES);     
    tPipe->InitBuffer(xOutQueue_, BUFFER_NUM, xNumPerBlock_ * sizeof(OutputType)); // 用于输出的OutPutTensor
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES); // 状态位每一个按32B对齐
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES); // 每次读 rankDim 个状态位
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES); // 用于清理状态区

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM)); // 用于状态区清零
}

/**
 * @brief 设置数据块大小（每个DataBlock包含的数据x元素数量）
 * 
 * @param elementsPerBlock 每个数据块包含的x元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::SetBlockSize(uint32_t elementsPerBlock)
{
    xNumPerBlock_ = elementsPerBlock;
}

/**
 * @brief 复制数据块从GM到Win区
 * 
 * 复制流程：
 * 1. 量化数据（x）复制：GM → UB（队列分配缓冲区） → Win区
 * 2. 缩放因子（scale）复制：GM → UB（队列分配缓冲区） → Win区
 * 
 * @param curXOffset 量化数据x在全局内存(GM)中的偏移量
 * @param curScaleOffset 缩放因子scale在全局内存(GM)中的偏移量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::CopyDataBlock(uint64_t curXOffset, uint64_t curScaleOffset)
{
    // 先拷贝data数据， 再拷贝scales
    /* x 从 GM -> UB -> Win */
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor_, xGMTensor_[curXOffset], xNumPerBlock_);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();
    DataCopy(localWinXGMTensor_[curXOffset], xTmpTensor_, xNumPerBlock_);
    xQueue_.FreeTensor<XType>(xTmpTensor_);

    /* scale 从 GM -> UB -> Win */
    scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
    DataCopy(scaleTmpTensor_, scalesGMTensor_[curScaleOffset], scaleNumsPerBlcok_);
    scaleQueue_.EnQue(scaleTmpTensor_);
    scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
    DataCopy(localWinScaleGMTensor_[curScaleOffset], scaleTmpTensor_, scaleNumsPerBlcok_);
    scaleQueue_.FreeTensor<ScalesType>(scaleTmpTensor_);
}

/**
 * @brief 将数据复制到Win区（支持ReduceScatter和AllReduce两种通信模式）
 * 
 * 该函数根据通信模式将数据从全局内存GM全部复制到本卡Win区，支持两种分布式通信模式：
 * 1. ReduceScatter模式：由于数据按Rank（设备）分片，需按rankDim遍历以搬运所有数据
 * 2. AllReduce模式：数据全局复制，每个核直接将数据复制到Win区
 * 
 * @tparam isReduceScatter 通信模式标识，true表示ReduceScatter模式，false表示AllReduce模式
 * 
 * @param[in] xSliceSizeNums 每个Rank的数据分片大小（元素数量），ReduceScatter模式使用
 * @param[in] scaleSliceNums 每个Rank的缩放因子分片大小（元素数量），ReduceScatter模式使用
 */
template <TemplateTypeClass>
template <bool isReduceScatter>
__aicore__ inline void MTECommunication<TemplateType>::CopyDataToWin(uint64_t xSliceSizeNums, uint64_t scaleSliceNums)
{
    // 遍历每个核需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xNumPerBlock_; // 计算现在搬第几个x
        uint64_t curScaleOffset = scaleOffset_ + curBlock * scaleNumsPerBlcok_; // 计算现在搬第几个scale
        
        if constexpr (isReduceScatter) {
            // ReduceScatter过程，数据按卡均分，需要对卡进行遍历
            for(uint64_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
                // all2all过程，加上卡偏移
                uint64_t curRankXOffset = curXOffset + curRank * xSliceSizeNums;
                uint64_t curRankScaleOffset = curScaleOffset + curRank * scaleSliceNums;

                // 搬运当前数据块到Win区
                CopyDataBlock(curRankXOffset, curRankScaleOffset);
            }
        } else {
            // AllReduce过程，allgather直接搬运
            CopyDataBlock(curXOffset, curScaleOffset);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

/**
 * @brief 向Win区状态区写入本核完成数据搬运状态
 * 
 * 该函数负责将当前AI Core（核）的数据搬运完成状态写入到所有Rank的状态Win区中，
 * 通知其他设备当前核的数据准备已完成。这是集群通信软同步机制的一部分，
 * 每个核需要向所有Rank（包括本机和其他设备）的状态窗口写入标识，
 * 确保所有设备都能感知到当前核的完成状态。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankDim; // Win区大小为 aivNum * rankDim, 此处计算核偏移
    // 遍历每一张卡，给每一张卡都要写入状态
    for (uint32_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        // 写入状态到对端，每个核写一个状态，表示自己的数据块已经写完
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_UB_ALIGN_NUM); // 先重置statusTensor数据，后面累加需要Tensor内全部数据，防止脏数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor(0) = (float)1.0;  // 用1标识
        GM_ADDR remoteWinStateGM = (GM_ADDR)hcclContext_->windowsOut[curRank]; // 获取当前要写对端卡的状态区地址
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        // 不同卡上的核的状态写到相邻位置，读时可以一次读rankDim个状态, 状态区大小设计为 aivNum * ranDim
        uint64_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_UB_ALIGN_NUM; // 当前核偏移 + 卡偏移， 按32B对齐
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM); // 按32B对齐拷贝
    }
}

/**
 * @brief 读取Win区状态区并等待同步状态
 * 
 * 该函数从本Rank的状态Win区读取同步标识，并在本地进行忙等待，直到所有参与通信的Rank
 * 都已完成状态设置。这种软同步机制确保所有Rank上当前核（AI Core）所需的数据都已
 * 准备就绪，从而避免跨设备数据不一致性问题。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::ReadStatus()
{
    GM_ADDR stateGM = (GM_ADDR)hcclContext_->windowsOut[hcclContext_->rankId]; // 获取本卡的状态区用于读取
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM; // 获取当前核所需读取状态位的头地址，状态按32B对齐
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0; // 用于计算状态和
    uint32_t statusCnt = hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM; // 一次读rankDim个，按32B对齐
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = hcclContext_->rankDim - (float)0.5;
    float maxTarget = hcclContext_->rankDim + (float)0.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt); // 相关状态区重新置零
}

/**
 * @brief 将计算得到的UB上的结果Tensor的数据复制到GM上的OutPutTensor，并进行类型转换（如果需要）。
 * 
 * @param outputOffset 输出OutputTensor的GM偏移量，用于指定目标位置。
 * @param sourceTensor 本地UB上计算结果Tensor，包含计算完成的数据。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float>& localResultTensor)
{
    // 将计算好的数据拷贝到输出tensor，如果是非float数据类型需要先转换成目标数据类型
    xOutTensor_ = xOutQueue_.AllocTensor<OutputType>();
    if constexpr (AscendC::IsSameType<OutputType, float>::value) {
        DataCopy(xOutTensor_, localResultTensor, xNumPerBlock_);
    } else {
        Cast(xOutTensor_, localResultTensor, RoundMode::CAST_RINT, xNumPerBlock_);
    }
    xOutQueue_.EnQue(xOutTensor_);
    xOutTensor_ = xOutQueue_.DeQue<OutputType>();
    DataCopy(outputTensor_[outOffsetGM], xOutTensor_, xNumPerBlock_);
    xOutQueue_.FreeTensor(xOutTensor_);
}
} // QuantMTECommImpl
#endif  // MTE_COMMON_H