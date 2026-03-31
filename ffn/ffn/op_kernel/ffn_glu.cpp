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
 * \file ffn_glu.cpp
 * \brief
 */
#ifndef ASCENDC_FFN_GLU_CPP
#define ASCENDC_FFN_GLU_CPP

#include "ffn_glu.h"


namespace FFN {
template <typename T>
__aicore__ inline void FFNGlu<T>::Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                       __gm__ uint8_t *bias1, __gm__ uint8_t *bias2, __gm__ uint8_t *y,
                                       __gm__ uint8_t *workSpace, const FFNTilingData *__restrict tiling, TPipe *tPipe)
{
    curBlockIdx = GetBlockIdx();
    tilingData = tiling;
    pipe = tPipe;
    initTilingData();

    xGm.SetGlobalBuffer((__gm__ T *)x);
    weight1Gm.SetGlobalBuffer((__gm__ T *)weight1);
    if (bias1 != nullptr) {
        hasBias1 = true;
        bias1Gm.SetGlobalBuffer((__gm__ T *)bias1);
    }
    weight2Gm.SetGlobalBuffer((__gm__ T *)weight2);
    if (bias2 != nullptr) {
        hasBias2 = true;
        bias2Gm.SetGlobalBuffer((__gm__ T *)bias2);
    }
    yGm.SetGlobalBuffer((__gm__ T *)y);

    // the data of mm1 once is baseM * baseN, and it will be multiple times
    uint32_t singleOffset = singleTimeM * singleTimeN * sizeof(T);
    uint32_t singleCoreOffset = singleOffset * 4; // 4: 2(left,right) * db(2)
    mm1ResLeft[0].SetGlobalBuffer((__gm__ T *)(workSpace + curBlockIdx * singleCoreOffset), singleOffset);
    mm1ResRight[0].SetGlobalBuffer((__gm__ T *)(workSpace + curBlockIdx * singleCoreOffset + singleOffset),
                                   singleOffset);
    // 2: the offset of ping(left + right)
    mm1ResLeft[1].SetGlobalBuffer((__gm__ T *)(workSpace + curBlockIdx * singleCoreOffset + singleOffset * 2),
                                  singleOffset);
    // 3: the offset of other 3 mm1 res(ping all + left pong)
    mm1ResRight[1].SetGlobalBuffer((__gm__ T *)(workSpace + curBlockIdx * singleCoreOffset + singleOffset * 3),
                                   singleOffset);
    // use real usedCoreNum
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ T *)(workSpace + m1Loops * n1Loops * singleCoreOffset));

    // vector without db, not a performance bottleneck, and calc more data
    pipe->InitBuffer(vecInQueueLeft, 1, baseM1 * baseN1 * sizeof(T));
    pipe->InitBuffer(vecInQueueRight, 1, baseM1 * baseN1 * sizeof(T));
    pipe->InitBuffer(vecOutQueue, 1, baseM1 * baseN1 * sizeof(T));
}

template <typename T>
__aicore__ inline void FFNGlu<T>::initTilingData()
{
    m1 = tilingData->ffnBaseParams.totalTokens;
    k1 = tilingData->ffnBaseParams.k1;
    n1 = tilingData->ffnBaseParams.n1;
    k2 = n1 / 2; // 2: k2 is half of n1
    n2 = tilingData->ffnBaseParams.n2;
    coreNum = tilingData->ffnBaseParams.coreNum;
    gluActiveType = tilingData->ffnBaseParams.activeType - numActiveTypes;
    dataTypeSize = 2; // 2: fp16 dtypesize

    // vector base
    baseM1 = tilingData->ffnSingleCoreParams.baseM1;
    baseN1 = tilingData->ffnSingleCoreParams.baseN1;

    // MM1
    singleM1 = tilingData->mm1TilingData.singleCoreM;
    singleN1 = tilingData->mm1TilingData.singleCoreN;
    m1Loops = Ceil(m1, static_cast<uint32_t>(singleM1));
    n1Loops = Ceil(k2, static_cast<uint32_t>(singleN1));
    singleM1Tail = m1 - (m1Loops - 1) * singleM1;
    singleN1Tail = k2 - (n1Loops - 1) * singleN1;
    singleTimeM = tilingData->mm1TilingData.baseM;
    singleTimeN = tilingData->mm1TilingData.baseN;

    // MM2
    singleM2 = tilingData->mm2TilingData.singleCoreM;
    singleN2 = tilingData->mm2TilingData.singleCoreN;
    m2Loops = Ceil(m1, singleM2);
    n2Loops = Ceil(n2, singleN2);
    singleM2Tail = m1 - (m2Loops - 1) * singleM2;
    singleN2Tail = n2 - (n2Loops - 1) * singleN2;

    InitActivationFunction();
}

template <typename T>
__aicore__ void GeGluPointer(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal0, LocalTensor<T> srcLocal1,
                             uint32_t dataSize)
{
    return GeGLU(dstLocal, srcLocal0, srcLocal1, dataSize);
}

template <typename T>
__aicore__ void SwiGluPointer(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal0, LocalTensor<T> srcLocal1,
                              uint32_t dataSize)
{
    return SwiGLU(dstLocal, srcLocal0, srcLocal1, BETA_, dataSize);
}

template <typename T>
__aicore__ void ReGluPointer(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal0, LocalTensor<T> srcLocal1,
                             uint32_t dataSize)
{
    return ReGlu(dstLocal, srcLocal0, srcLocal1, dataSize);
}

template <typename T>
__aicore__ inline void FFNGlu<T>::InitActivationFunction()
{
    GluActiveType2Func<T> active2funMap[numGluActiveTypes] = {{GluActiveType::GEGLU, GeGluPointer},
                                                              {GluActiveType::SWIGLU, SwiGluPointer},
                                                              {GluActiveType::REGLU, ReGluPointer}};
    if (gluActiveType < numGluActiveTypes) {
        activeFunc = active2funMap[gluActiveType].gluFuncPointer;
    }
}

template <typename T>
__aicore__ inline void FFNGlu<T>::Process()
{
    // mm1 glu: parallel calc
    MM1GluSplit();
    // glu write first, then mm2 read
    SyncAll<true>();
    MM2Split();
}

template <typename T>
__aicore__ void FFNGlu<T>::MM1GluSplit()
{
    // real used cores may not be equal to aivNum
    if (curBlockIdx >= m1Loops * n1Loops) {
        return;
    }
    CalcMM1GluParams();

    uint32_t taskId = 0;
    mm1.SetOrgShape(m1, n1, k1);
    uint64_t xCurOffset = xCoreOffset;
    uint32_t lastAicCalc[2] = {0, 0};
    for (uint32_t mInnerIdx = 0; mInnerIdx < mInnerLoops; mInnerIdx++) {
        uint32_t curAicM = mInnerIdx == mInnerLoops - 1 ? aicMtail : singleTimeM;
        mm1.SetTensorA(xGm[xCurOffset]);
        xCurOffset += singleTimeM * k1;
        uint64_t w1LeftOffset = w1CoreOffset;
        for (uint32_t nInnerIdx = 0; nInnerIdx < nInnerLoops; nInnerIdx++) {
            uint32_t curAicN = nInnerIdx == nInnerLoops - 1 ? aicNtail : singleTimeN;
            if (taskId > 0) {
                mm1.WaitIterateAll();
            }
            MM1Compute(curAicM, curAicN, w1LeftOffset, taskId & 0x1);
            w1LeftOffset += singleTimeN;
            // only aic when task is 0
            if (taskId > 0) {
                uint64_t lastMm1Offset = lastCoreMm1Offset;
                // parallel compute: aic_{i} and aiv_{i-1}，aiv compute offset should be aic_{i-1}
                if (nInnerIdx == 0) {
                    lastMm1Offset += (mInnerIdx - 1) * singleTimeM * k2 + (nInnerLoops - 1) * singleTimeN;
                } else {
                    lastMm1Offset += mInnerIdx * singleTimeM * k2 + (nInnerIdx - 1) * singleTimeN;
                }
                GluSplit(lastAicCalc[0], lastAicCalc[1], lastMm1Offset, 1 - taskId & 0x1);
            }
            lastAicCalc[0] = curAicM;
            lastAicCalc[1] = curAicN;
            taskId++;
        }
    }
    mm1.WaitIterateAll();
    mm1.End();

    // last aiv compute
    uint64_t lastMm1Offset = lastCoreMm1Offset + (mInnerLoops - 1) * singleTimeM * k2 + (nInnerLoops - 1) * singleTimeN;
    GluSplit(aicMtail, aicNtail, lastMm1Offset, 1 - taskId & 0x1);
}

template <typename T>
__aicore__ void FFNGlu<T>::CalcMM1GluParams()
{
    uint32_t mCoreIndx = curBlockIdx / n1Loops;
    uint32_t nCoreIndx = curBlockIdx % n1Loops;
    xCoreOffset = static_cast<uint64_t>(mCoreIndx) * singleM1 * k1;
    w1CoreOffset = static_cast<uint64_t>(nCoreIndx) * singleN1;
    lastCoreMm1Offset = static_cast<uint64_t>(mCoreIndx) * singleM1 * k2 + nCoreIndx * singleN1;
    curSingleM = mCoreIndx == m1Loops - 1 ? singleM1Tail : singleM1;
    curSingleN = nCoreIndx == n1Loops - 1 ? singleN1Tail : singleN1;
    // the data of mm1 once is baseM * baseN
    mInnerLoops = Ceil(curSingleM, singleTimeM);
    nInnerLoops = Ceil(curSingleN, singleTimeN);
    aicMtail = curSingleM - (mInnerLoops - 1) * singleTimeM;
    aicNtail = curSingleN - (nInnerLoops - 1) * singleTimeN;
}

template <typename T>
__aicore__ void FFNGlu<T>::MM1Compute(uint32_t curAicM, uint32_t curAicN, uint64_t w1Offset, uint32_t pingPongId)
{
    // calc left(0 ~ n1/2)
    mm1.SetTensorB(weight1Gm[w1Offset]);
    mm1.SetTail(curAicM, curAicN, k1);
    if (hasBias1) {
        mm1.SetBias((bias1Gm[w1Offset]));
    }
    mm1.template IterateAll<true>(mm1ResLeft[pingPongId], false, true, false);

    // calc right(n1/2 ~ n1)
    uint64_t w1RightOffset = k2 + w1Offset;
    mm1.SetTensorB(weight1Gm[w1RightOffset]);
    if (hasBias1) {
        mm1.SetBias((bias1Gm[w1RightOffset]));
    }
    mm1.template IterateAll<false>(mm1ResRight[pingPongId], false, true, true);
}

template <typename T>
__aicore__ void FFNGlu<T>::GluSplit(uint32_t curAicM, uint32_t curAicN, uint64_t lastMm1Offset, uint32_t pingPongId)
{
    uint32_t mAicAivLoops = Ceil(curAicM, baseM1);
    uint32_t nAicAivLoops = Ceil(curAicN, baseN1);
    uint32_t aivMTail = curAicM - (mAicAivLoops - 1) * baseM1;
    uint32_t aivNTail = curAicN - (nAicAivLoops - 1) * baseN1;

    for (uint32_t mMuti = 0; mMuti < mAicAivLoops; mMuti++) {
        uint32_t curBaseM1 = mMuti == mAicAivLoops - 1 ? aivMTail : baseM1;
        for (uint32_t nMuti = 0; nMuti < nAicAivLoops; nMuti++) {
            uint32_t curBaseN1 = nMuti == nAicAivLoops - 1 ? aivNTail : baseN1;
            DataCopyParams gm2UbParams;
            DataCopyPadParams padParams;
            gm2UbParams.blockLen = curBaseN1 * dataTypeSize;
            gm2UbParams.blockCount = curBaseM1;
            gm2UbParams.srcStride = (curAicN - curBaseN1) * dataTypeSize;
            gm2UbParams.dstStride = 0;

            LocalTensor<T> aIn = vecInQueueLeft.AllocTensor<T>();
            LocalTensor<T> bIn = vecInQueueRight.AllocTensor<T>();

            uint32_t curAicAivOffset = mMuti * baseM1 * curAicN + nMuti * baseN1;
            // 32B aligned in ub
            DataCopyPad(aIn, mm1ResLeft[pingPongId][curAicAivOffset], gm2UbParams, padParams);
            DataCopyPad(bIn, mm1ResRight[pingPongId][curAicAivOffset], gm2UbParams, padParams);

            vecInQueueLeft.EnQue(aIn);
            vecInQueueRight.EnQue(bIn);
            uint32_t computeBaseN1 = AlignUp<GetNumInUbBlock<T>()>(curBaseN1);
            GluCompute(curBaseM1 * computeBaseN1);

            uint64_t activeOffset = lastMm1Offset + mMuti * baseM1 * k2 + nMuti * baseN1;
            LocalTensor<T> gluOut = vecOutQueue.DeQue<T>();

            DataCopyExtParams ub2GmParams;
            ub2GmParams.blockLen = curBaseN1 * dataTypeSize;
            ub2GmParams.blockCount = curBaseM1;
            ub2GmParams.srcStride = 0;
            ub2GmParams.dstStride = (k2 - curBaseN1) * dataTypeSize;
            DataCopyPad(mm2WorkspaceGm[activeOffset], gluOut, ub2GmParams);

            vecOutQueue.FreeTensor(gluOut);
        }
    }
}

template <typename T>
__aicore__ void FFNGlu<T>::GluCompute(uint32_t computeSize)
{
    LocalTensor<T> aIn = vecInQueueLeft.DeQue<T>();
    LocalTensor<T> bIn = vecInQueueRight.DeQue<T>();
    LocalTensor<T> gluOut = vecOutQueue.AllocTensor<T>();

    activeFunc(gluOut, aIn, bIn, computeSize);

    vecInQueueLeft.FreeTensor(aIn);
    vecInQueueRight.FreeTensor(bIn);
    vecOutQueue.EnQue<T>(gluOut);
}

template <typename T>
__aicore__ inline void FFNGlu<T>::MM2Split()
{
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t blockIdx = curBlockIdx / GetTaskRation();

    if (blockIdx < m2Loops * n2Loops && subBlockIdx == 0) {
        uint32_t m2Idx = blockIdx / n2Loops;
        uint32_t n2Idx = blockIdx % n2Loops;
        uint32_t curSingleM2 = m2Idx != m2Loops - 1 ? singleM2 : singleM2Tail;
        uint32_t curSingleN2 = n2Idx != n2Loops - 1 ? singleN2 : singleN2Tail;
        mm2.SetSingleShape(curSingleM2, curSingleN2, k2);
        mm2.SetTensorA(mm2WorkspaceGm[static_cast<uint64_t>(m2Idx) * singleM2 * k2]);
        mm2.SetTensorB(weight2Gm[n2Idx * singleN2]);
        if (hasBias2) {
            mm2.SetBias(bias2Gm[n2Idx * singleN2]);
        }
        uint32_t outOffset = static_cast<uint64_t>(m2Idx) * singleM2 * n2 + n2Idx * singleN2;
        mm2.template IterateAll<true>(yGm[outOffset], false);
        mm2.End();
    }
}
} // namespace FFN

#endif // ASCENDC_FFN_GLU_CPP



D:\code\ops-nn\index\where
请参照上述where 目录下的文件夹和文件信息，创建一个新的名为SpatialTransformer算子的目录结构及文件
SpatialTransformer 算子的kernel aicpu实现源码在
D:\code\cann\canndev\ops\built-in\aicpu\impl\kernels\normalized\image\spatial_transformer.cc
proto文件在
D:\code\cann\canndev\ops\built-in\op_proto\inc\image_ops.h中
test下的ut文件在
D:\code\cann\canndev\ops\built-in\tests\ut\aicpu_test\testcase\spatial_transformer\src\spatial_transformer_utest.cpp


根据D:\code\cann\canndev\ops\built-in\aicpu\op_info_cfg\aicpu_kernel\aicpu_kernel.ini中where 到D:\code\ops-nn\index\where\op_kernel_aicpu\where.json中信息的映射规则，
根据ini文件中SpatialTransformer信息，在D:\code\ops-nn\index\spatial_transformer\op_kernel_aicpu\ 目录下生成spatial_transformer.json文件


根据/mnt/d/code/ops-nn/index/where/examples/test_geir_where.cpp中的代码实现，在/mnt/d/code/ops-nn/index/spatial_transformer下新增examples目录，并
生成test_geir_spatial_transformer.cpp源文件，SpatialTransformer的算子原型在/mnt/d/code/ops-nn/index/spatial_transformer/op_graph目录下

 bash build.sh --pkg --ops=crop_and_resize


/mnt/d/code/ops-cv

apt-get install dos2unix


source /home/runtime/ascend-toolkit/latest/bin/setenv.bash

cd /mnt/d/code/ops-nn
source /home/runtime/ascend-toolkit/latest/bin/setenv.bash
nohup bash build.sh --pkg --ops=where > build.log 2>&1

tensorflow==2.20.0

问题一：
op_kenrel 下面的漏指定规则生成json文件
问题二：
op_kenel 下面Cmake中CMAKE_CURRENT_SOURCE_DIR 变量错误
问题三：
aicpu_xx.cc  未改为aicpu_xx.cpp
问题四：
aicpu_xx.cc 中包含的头文件 未改为aicpu_xx_aicpu.h
问题五：
aicpu_xx_aicpu.cpp文件中KERNEL_LOG_ERROR未找到定义  ---原先实现未包含log.h
问题六：
aicpu_xx_aicpu.cpp 文件中原先未包含kernel_util.h 文件 ---强行模仿了标杆文件中的头文件，删减了之前有的部分文件
问题七：
搬运过来到头文件中的类成员变量被改动----，原先int64_t 被改成int32_t 很隐蔽很危险，初始化的动作也给干掉了
问题八：
tests 目录下op_kernel_aicpu 算子kernel 源文件中，部分实现未迁移过来
问题九：
test kernel下面的头文件aicpu_read_file.h文件不再使用

