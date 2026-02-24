/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_BASE_QUANT_MATMUL_H
#define MC2_BASE_QUANT_MATMUL_H

namespace MC2KernelTemplate {

template <typename Derived, typename MMKernel, typename ExtraDataType, typename TilingDataType>
class BaseQuantMatmul {
public:
    __aicore__ inline BaseQuantMatmul(AscendC::TPipe *tPipe);

    __aicore__ inline void Process(bool isFirst);

    __aicore__ inline void Update(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, ExtraDataType *extraData,
                                  TilingDataType *tilingData);

    __aicore__ inline void End();

protected:
    MMKernel mmOp_;
    BaseGmAddrs baseAddrs_;
    ExtraDataType extraData_;
    AscendC::TPipe *tPipe_;
    TilingDataType *tilingData_;
};

template <typename Derived, typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline BaseQuantMatmul<Derived, MMKernel, ExtraDataType, TilingDataType>::BaseQuantMatmul(
    AscendC::TPipe *tPipe)
    : tPipe_(tPipe)
{
}

template <typename Derived, typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void BaseQuantMatmul<Derived, MMKernel, ExtraDataType, TilingDataType>::Process(bool isFirst)
{
    if (!isFirst) {
        // 基类处理通用的矩阵地址偏移
        baseAddrs_.aGM = (GM_ADDR)((uint64_t)baseAddrs_.aGM + extraData_.a_offset);
        baseAddrs_.bGM = (GM_ADDR)((uint64_t)baseAddrs_.bGM + extraData_.b_offset);
        baseAddrs_.cGM = (GM_ADDR)((uint64_t)baseAddrs_.cGM + extraData_.c_offset);

        // 调用钩子处理变化的量化参数偏移 (scale, offset 等)
        static_cast<Derived *>(this)->ShiftQuantParams();
    }
    static_cast<Derived *>(this)->Init();
    mmOp_.Process();
}

template <typename Derived, typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void BaseQuantMatmul<Derived, MMKernel, ExtraDataType, TilingDataType>::Update(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, ExtraDataType *extraData, TilingDataType *tilingData)
{
    baseAddrs_.aGM = aGM;
    baseAddrs_.bGM = bGM;
    baseAddrs_.cGM = cGM;
    baseAddrs_.biasGM = biasGM;
    if (extraData != nullptr) {
        // 通用 offset 拷贝
        extraData_.a_offset = extraData->a_offset;
        extraData_.b_offset = extraData->b_offset;
        extraData_.c_offset = extraData->c_offset;

        // 调用钩子拷贝具体的量化数据
        static_cast<Derived *>(this)->UpdateQuantParams(extraData);
    }
    if (tilingData != nullptr) {
        tilingData_ = tilingData;
    }
}

template <typename Derived, typename MMKernel, typename ExtraDataType, typename TilingDataType>
__aicore__ inline void BaseQuantMatmul<Derived, MMKernel, ExtraDataType, TilingDataType>::End()
{
}

}; // namespace MC2KernelTemplate
#endif