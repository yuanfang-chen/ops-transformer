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
 * \file quant_grouped_matmul.h
 * \brief
 */

#ifndef QUANT_GROUPED_MATMUL_H
#define QUANT_GROUPED_MATMUL_H

namespace ATAVKernelTemplate {

template <typename QGMMKernel, typename TilingDataType>
class GmmExpertOp
{
public:
    __aicore__ inline void Init();
    __aicore__ inline GmmExpertOp(AscendC::TPipe* tPipe): tPipe_(tPipe) {};
    __aicore__ inline void Process(bool hasNext);
    __aicore__ inline void Update(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, ExtraDataType* extraData, TilingDataType* tilingData);
    __aicore__ inline void End();
private:
    QGMMKernel gmmOp_;
    BaseGmAddrs baseAddrs_;
    AscendC::TPipe* tPipe_;
    TilingDataType* tilingData_;    
};

template <typename QGMMKernel, typename TilingDataType>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::Init() 
{
    tPipe_->Reset();
    gmmOp_.Init(baseAddrs_.xGM, baseAddrs_.weightGM, nullptr, baseAddrs_.WeightScaleGM, 0, baseAddrs_.xScaleGM, baseAddrs_.yGM, nullptr,
 	            &gmmQuantParams_, &mmTilingData_, gmmArrayAddr_, tPipe_);  // &gmmQuantParams_, &mmTilingData_, gmmArrayAddr_ 待分析
}

template <typename QGMMKernel, typename TilingDataType>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::Process(bool isFirst){
    
    if (!isFirst) {
        // 待完善baseAddrs_.xGM等地址偏移的计算

        Init();
    }
    gmmOp_.Process();
}

template <typename QGMMKernel, typename TilingDataType>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::Update(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR yGM, TilingDataType* tilingData) 
{    
    // 待完善baseAddrs_.xGM等地址偏移的计算

    if (tilingData != nullptr) {
        tilingData_ = tilingData;
        Init();
    }
}

template <typename QGMMKernel, typename TilingDataType>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::End()
{
}

};

#endif //QUANT_GROUPED_MATMUL_H