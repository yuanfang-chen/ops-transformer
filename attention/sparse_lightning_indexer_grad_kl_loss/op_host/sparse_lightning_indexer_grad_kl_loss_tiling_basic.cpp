/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_basic.cpp
 * \brief
 */

#include "sparse_lightning_indexer_grad_kl_loss_tiling_basic.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus SparseLightningIndexerGradKLLossTilingGeneral::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    // 输入属性参数验证
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    // idx++跳过pretoken和nexttoken
    idx++;
    idx++;
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    // 输入shape验证
    auto queryShape = context_->GetInputShape(QUERY_INPUT_INDEX);
    auto keyShape = context_->GetInputShape(KEY_INPUT_INDEX);
    auto queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX);
    auto keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX);
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX);
    auto sparseIndicesShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    auto softmaxMaxShape = context_->GetInputShape(SOFTMAX_MAX_INPUT_INDEX);
    auto softmaxSumShape = context_->GetInputShape(SOFTMAX_SUM_INPUT_INDEX);
    // 输出shape验证
    auto dQueryIndexShape = context_->GetOutputShape(D_QUERY_INDEX_OUTPUT_INDEX);
    auto dkeyIndexShape = context_->GetOutputShape(D_KEY_INDEX_OUTPUT_INDEX);
    auto dWeightsShape = context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX);
    auto lossShape = context_->GetOutputShape(LOSS_OUTPUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseIndicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxMaxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxSumShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dQueryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dkeyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, lossShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    
    return ge::GRAPH_SUCCESS;
}

bool SparseLightningIndexerGradKLLossTilingGeneral::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    // idx++跳过pretoken和nexttoken
    idx++;
    idx++;
    scaleValue = *scaleValuePtr;
    inputLayout = inputLayoutPtr;
    sparseMode = *sparseModePtr;
    deterministic = (context_->GetDeterministic() == 1);

    OP_CHECK_IF(sparseMode != SPARSE_MODE_SIZE_3,
                OP_LOGE(opName, " The value of sparse_mode is [%d], but currently only supports mode [3].", sparseMode),
                return false);
    OP_LOGD(context_, "attrs: scaleValue[%f] input_layout[%s] sparse_mode[%ld].",
            scaleValue, inputLayout, sparseMode);
    return true;
}

void SparseLightningIndexerGradKLLossTilingGeneral::GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return;
    }
    int64_t seqLen = actualSeqLenShape.GetDim(0);
    try{
        res.reserve(seqLen);
    } catch (...) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "Init actual_seq_len failed, array is too long.");
        return;
    }
    res.emplace_back(value[0]);
    actualLen++;
    for (int64_t i = 1; i < seqLen; ++i) {
        auto qLen = value[i] - value[i - 1]; // value[i]代表偏移位置的索引
        qLen = qLen < 0 ? 0 : qLen;
        res.emplace_back(qLen);
        actualLen++;
    }
}

// 对每一个输入参数的变量类型进行对比校验
bool SparseLightningIndexerGradKLLossTilingGeneral::AnalyzeDtype()
{
    // 对8个必须输入的参数进行参数类型判断
    // 输入空指针校验
    auto queryDesc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    auto keyDesc = context_->GetInputDesc(KEY_INPUT_INDEX);
    auto queryIndexDesc = context_->GetInputDesc(QUERY_INDEX_INPUT_INDEX);
    auto keyIndexDesc = context_->GetInputDesc(KEY_INDEX_INPUT_INDEX);
    auto weightsDesc = context_->GetInputDesc(WEIGHT_INPUT_INDEX);
    auto sparseIndicesDesc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    auto softmaxMaxDesc = context_->GetInputDesc(SOFTMAX_MAX_INPUT_INDEX);
    auto softmaxSumDesc = context_->GetInputDesc(SOFTMAX_SUM_INPUT_INDEX); 
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryIndexDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyIndexDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseIndicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxMaxDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxSumDesc);

    // 以下5个为16类型保持一致
    auto queryDtype = queryDesc->GetDataType();
    auto keyDtype = keyDesc->GetDataType();
    auto queryIndexDtype = queryIndexDesc->GetDataType();
    auto keyIndexDtype = keyIndexDesc->GetDataType();
    auto weightsDtype = weightsDesc->GetDataType();

    // 以下3个为32类型
    auto sparseIndicesDtype = sparseIndicesDesc->GetDataType();
    auto softmaxMaxDtype = softmaxMaxDesc->GetDataType();
    auto softmaxSumDtype = softmaxSumDesc->GetDataType();

    bool same16 = false; // 判断输入为fp16或者部分16的是否一致
    bool same32 = false; // 判断输入为int32或者fp32的类型是否正确
    if (queryDtype == ge::DT_FLOAT16 && keyDtype == ge::DT_FLOAT16 && queryIndexDtype == ge::DT_FLOAT16 && 
                            keyIndexDtype == ge::DT_FLOAT16 && weightsDtype == ge::DT_FLOAT16) {
        same16 = true;
    } else if (queryDtype == ge::DT_BF16 && keyDtype == ge::DT_BF16 && queryIndexDtype == ge::DT_BF16 && 
                            keyIndexDtype == ge::DT_BF16 && weightsDtype == ge::DT_BF16) {
        same16 = true;
    } else {
        OP_LOGW(context_, "InputDtype is not same.: queryDtype[%s], keyDtype[%s], queryIndexDtype[%s], keyIndexDtype[%s], weightsDtype[%s]",
            ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(queryIndexDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyIndexDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str());
        same16 = false;
    } 
    if (sparseIndicesDtype == ge::DT_INT32 && softmaxMaxDtype == ge::DT_FLOAT && softmaxSumDtype == ge::DT_FLOAT) {
        same32 = true;
    } else {
        OP_LOGW(context_, "InputDtype is fault: sparseIndicesDtype must be int32, but[%s]; softmaxMaxDtype must be float32, but[%s]; softmaxSumDtype must be float32, but[%s].",
            ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(softmaxMaxDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(softmaxSumDtype).c_str());
        same32 = false;
    }
    // 所有类型不满足返回false
    if(same16 == false || same32 == false) {
        return false;
    }
    OP_LOGW(context_, "InputDtype: queryDtype[%s], keyDtype[%s], queryIndexDtype[%s], keyIndexDtype[%s], weightsDtype[%s],",
        ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(queryIndexDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(keyIndexDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str());
    OP_LOGW(context_, "sparseIndicesDtype[%s], softmaxMaxDtype[%s], softmaxSumDtype[%s].",
        ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(softmaxMaxDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(softmaxSumDtype).c_str());
    return true;
}

ge::graphStatus SparseLightningIndexerGradKLLossTilingGeneral::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SparseLightningIndexerGradKLLossCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        socVersion = compileInfoPtr->socVersion;
        npuArch = compileInfoPtr->npuArch;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        socVersion = ascendcPlatform.GetSocVersion();
        npuArch = ascendcPlatform.GetCurNpuArch();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

int64_t SparseLightningIndexerGradKLLossTilingGeneral::GetS2RealSize(int32_t sparseMode, int32_t s1Size, 
                                                                        int32_t s2Size, int32_t s1Idx) 
{
    int64_t s2RealSize = 0;
    // sparsemode = 3 的情形
    if (sparseMode == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = static_cast<int64_t>((s2Size - s1Size) + s1Idx + 1);
        if (s2RealSize <= 0) {
            s2RealSize = static_cast<int64_t>(s2Size);
        }
    }
    s2RealSize = AlignUp(s2RealSize, 512L);
    return std::min(s2RealSize, static_cast<int64_t>(kSize));
}

bool SparseLightningIndexerGradKLLossTilingGeneral::InitSparseValidArray(std::vector<int64_t> &sparseValidArray, uint32_t sparseMode)
{
    OP_CHECK_IF(sparseValidArray.size() == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."),
                return false);

    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        int64_t totalSize = 0;
        int64_t accumS1 = 0;
        for (int32_t i = 0; i < bSize; i++) {
            int64_t s1Size = actualSeqLenData[i];
            int64_t s2Size = actualSeqLenKData[i];
            for (int64_t j = 0; j < s1Size; j++) {
                int64_t s2RealSize = GetS2RealSize(sparseMode, s1Size, s2Size, j);
                sparseValidArray[accumS1] = s2RealSize;
                accumS1++;
            }
        }        
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t accum = 0;
        for (int32_t i = 0; i < bSize; i++) {
            for (int64_t j = 0; j < s1Size; j++) {
                int64_t s2RealSize = GetS2RealSize(sparseMode, s1Size, s2Size, j);
                sparseValidArray[accum] = s2RealSize;
                accum++;
            }
        } 
    }

    return true;
}

bool SparseLightningIndexerGradKLLossTilingGeneral::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
                                                                      const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
{
    for (int64_t idx = 0; idx < validAicNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAicNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] =
                std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
        } else {
            break;
        }
    }
    return true;
}

bool SparseLightningIndexerGradKLLossTilingGeneral::BalanceLoad(const std::vector<int64_t> &sparseValidArray, std::vector<int64_t> &localValue,
                                                                std::vector<int64_t> &sparseStartIdx, int64_t validAicNum, int64_t totalSize)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAicNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAicNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

bool SparseLightningIndexerGradKLLossTilingGeneral::Balance4DLoad(std::vector<int64_t> &tmpSparseValue, const std::vector<int64_t> sparseValidArray, 
                                                                const int64_t balanceNum)
{
    int64_t tmpIndex = 0;
    tmpSparseValue[tmpIndex] = 0;
    int64_t sumTmpArray = 0;
    int64_t sumTmpArrayLast = 0; // 记录上次的总和值
    for (int64_t idx = 0; idx < sparseValidArray.size(); ++idx) {
        // 第一次分到最后一块后就没必须计算，剩余的直接分给最后一个核即可
        if (tmpIndex == static_cast<int64_t>(tmpSparseValue.size()) - 1) {
            break; 
        }

        sumTmpArrayLast = sumTmpArray;
        sumTmpArray += sparseValidArray[idx];
        if (sumTmpArray == balanceNum) {
            tmpIndex = (tmpIndex + 1 < tmpSparseValue.size()) ? tmpIndex + 1 : tmpSparseValue.size() - 1;
            tmpSparseValue[tmpIndex] = idx + 1; //刚好等于均值时核的末尾为idx
            sumTmpArray = 0; // 重新计算总和
        } else if (sumTmpArray > balanceNum) {
            tmpIndex = (tmpIndex + 1 < tmpSparseValue.size()) ? tmpIndex + 1 : tmpSparseValue.size() - 1;
            // 第一次总和大于均值，判断前面一次和当前哪个更接近均值
            if (balanceNum - sumTmpArrayLast >= sumTmpArray - balanceNum) {
                // 当前更接近，取当前值
                tmpSparseValue[tmpIndex] = idx + 1; //刚好等于均值时核的末尾为idx                
            }else {
                // 上一次更接近， 取上一次值
                tmpSparseValue[tmpIndex] = idx; //刚好等于均值时核的末尾为idx                
                idx--; // 记录的上一个值，重新计算时也需要从上一个值开始计算                 
            }
            sumTmpArray = 0; // 重新计算总和
        }
    }
    return true;
}

// 负载均衡
bool SparseLightningIndexerGradKLLossTilingGeneral::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize, 
                                                                      int64_t *sparseStartIdx, int64_t splitFactorSize, int64_t maxCoreNum)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    OP_CHECK_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "totalSize should be larger than 0."),
                   return false);
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        // initLoad: 使用均分策略, 保证后续不会比均分差
        std::vector<int64_t> localSparseStartIdx(aicNum, totalSize);
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAicNum, 0);
        //得到了每个核需要计算的S2数量，在T方向平均分配，但是有于有Sparse的存在，localValue中的数值，每个核之间应该差距很大
        InitLoadValue(sparseValidArray, validAicNum, totalSize, localSparseStartIdx, localValue); 

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAicNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(aicNum, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有T上的S2的总数量
        int64_t avgVal = CeilDivision(sparseArraySum, validAicNum); // 平均每个核需要处理的S2的总数量

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < static_cast<int64_t>(aicNum); ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                            (tmpsparseStartIdx[idx] - 1) : (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                    singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAicNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAicNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, tmpLocalValue, tmpsparseStartIdx, validAicNum, totalSize)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有BS上的S2的总数量
        int64_t balanceNum = CeilDivision(sparseArraySum, validAicNum);
        std::vector<int64_t> tmpSparseValue(validAicNum, 0);
        Balance4DLoad(tmpSparseValue, sparseValidArray, balanceNum);
        for (int64_t idx = 0; idx < static_cast<int64_t>(validAicNum); ++idx) {
            sparseStartIdx[idx] = tmpSparseValue[idx];
        }
    }

    for (int64_t idx = 1; idx < maxCoreNum; ++idx) {
        if (sparseStartIdx[idx] == 0) { 
            sparseStartIdx[idx] = static_cast<int64_t>(sparseValidArray.size()); // 赋值为s1最大值
        }
    }    
    return true;
}

// 计算S1的总个数
int64_t SparseLightningIndexerGradKLLossTilingGeneral::CalcTotalSize() {
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return realT1Size;
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) { 
        return bSize * s1Size;
    }
    return 0; //什么也不走就返回0
}

int64_t SparseLightningIndexerGradKLLossTilingGeneral::InitOutputSplit()
{
    auto &dKeyIndexShape = context_->GetOutputShape(1)->GetStorageShape();
    int64_t totalsize = 0;
    uint32_t singlecoresize = 0;
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(2); // TND场景下D为第二个维度
        int64_t totalT2Size = dKeyIndexShape.GetDim(0); // T为第一个维度
        totalsize = totalT2Size * static_cast<int64_t>(dsizeDkeyindex);
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(3); // BSND场景下D为第三个维度
        int64_t totals2Size = dKeyIndexShape.GetDim(1); // s为第二个维度
        totalsize = static_cast<int64_t>(bSize) * totals2Size * dsizeDkeyindex;
    }
    return totalsize;
}

} // namespace optiling
