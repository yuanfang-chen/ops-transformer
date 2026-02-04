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
 * \file matmul_factory.h
 * \brief
 */

#ifndef MC2_MATMUL_FACTORY_H
#define MC2_MATMUL_FACTORY_H 

namespace MC2KernelTemplate {
// 基本输入输出和偏移
struct MC2MMBaseGmAddrs {
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR cGM;
    GM_ADDR biasGM;
    uint64_t a_offset;
    uint64_t b_offset;
    uint64_t c_offset;
};

// matmul计算节点的数据上下文
template <typename AdditionalGmAddrDataType, typename TilingDataType>
struct MC2MMContext {
    // 基本输入输出和偏移
    MC2MMBaseGmAddrs baseData;
    // 不同场景下额外的输入输出和偏移
    AdditionalGmAddrDataType additionalData;
    // mamtul的tiling
    TilingDataType* tilingDataPtr;
};

// matmul计算节点的模板类
template <typename MMContextType, template <typename> class MMControlType, typename MMType>
class MC2MMFactory {
protected:
    // 数据上下文
    MMContextType<MMType> MMcontext_;
    // 不同场景对应的控制逻辑实现
    MMControlType MMControl_;
    // 不同场景使用的matmul实现
    MMType MMImpl_;
protected:
    __aicore__ inline void UpdataBaseGm(MC2MMBaseGmAddrs* data);
public:
    __aicore__ inline MC2MMFactory(){};
    // 初始化方法
    __aicore__ inline void Init(AscendC::TPipe* tPipe);
    // 获取数据上下文引用
    __aicore__ inline MMContextType* GetMMContextPtr();
    // 执行一次计算的方法
    __aicore__ inline void Process(bool isFirst);
    // 结束方法
    __aicore__ inline void End();
};

template <typename MMContextType, typename MMControlType, typename MMType>
__aicore__ inline void MC2MMFactory<MMContextType, MMControlType, MMType>::UpdataBaseGm(MC2MMBaseGmAddrs* data)
{
    data->aGM = data->aGM + data->a_offset;
    data->bGM = data->bGM + data->b_offset;
    data->cGM = data->cGM + data->c_offset;
}

template <typename MMContextType, typename MMControlType, typename MMType>
__aicore__ inline void MC2MMFactory<MMContextType, MMControlType, MMType>::Init(AscendC::TPipe *tPipe)
{
    MMControl_.Init(&MMcontext_.baseData, &MMcontext_.additionalData, MMcontext_.tilingDataPtr, &MMImpl_, tPipe);
}

template <typename MMContextType, typename MMControlType, typename MMType>
__aicore__ inline MMContextType* MC2MMFactory<MMContextType, MMControlType, MMType>::GetMMContextPtr()
{
    return &MMcontext_;
}

template <typename MMContextType, typename MMControlType, typename MMType>
__aicore__ inline void MC2MMFactory<MMContextType, MMControlType, MMType>::Process(bool isFirst)
{
    if (!isFirst) {
        UpdataBaseGm(&MMcontext_.baseData);
        MMControl_.UpdateAdditionalData();
    }
    MMControl_.InitMM();
    MMImpl_.Process();
}

template <typename MMContextType, typename MMControlType, typename MMType>
__aicore__ inline void MC2MMFactory<MMContextType, MMControlType, MMType>::End()
{
    MMControl_.EndMM();
}

}; // namespace MC2KernelTemplate

#endif