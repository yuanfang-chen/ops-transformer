/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include "graph/types.h"
#include "aclnn_mla_prolog_v3_weight_nz.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"

// 新增：引入ACL TensorDesc相关头文件（根据实际环境调整）
#include "acl/acl.h"
#include "acl/acl_tensor.h"

using namespace op;

// 修正：封装aclTensor Shape打印函数（通过TensorDesc获取维度）
static std::string GetAclTensorShapeStr(const aclTensor *tensor) {
    if (tensor == nullptr) {
        return "nullptr";
    }
    // 修正：先获取TensorDesc，再通过Desc获取维度
    const aclTensorDesc *desc = aclGetTensorDesc(tensor);
    if (desc == nullptr) {
        return "invalid tensor desc";
    }
    int numDims = aclGetTensorDescNumDims(desc);
    if (numDims < 0) {
        return "invalid dim num: " + std::to_string(numDims);
    }
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < numDims; ++i) {
        int64_t dim = aclGetTensorDescDim(desc, i);
        oss << dim;
        if (i != numDims - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

// 修正：打印aclTensor信息（名称+Shape+数据类型）
static void LogAclTensorInfo(const std::string &tensorName, const aclTensor *tensor) {
    if (tensor == nullptr) {
        OP_LOGI("Tensor [%s] is nullptr", tensorName.c_str());
        return;
    }
    // 修正：先获取TensorDesc，再获取数据类型
    const aclTensorDesc *desc = aclGetTensorDesc(tensor);
    if (desc == nullptr) {
        OP_LOGI("Tensor [%s] - Shape: invalid desc, DataType: invalid desc", tensorName.c_str());
        return;
    }
    // 获取数据类型字符串（增强日志可读性）
    std::string dtypeStr = "UNKNOWN";
    aclDataType dtype = aclGetTensorDescType(desc);
    switch (dtype) {
        case ACL_FLOAT: dtypeStr = "ACL_FLOAT"; break;
        case ACL_BF16: dtypeStr = "ACL_BF16"; break;
        case ACL_INT8: dtypeStr = "ACL_INT8"; break;
        case ACL_FLOAT8_E4M3FN: dtypeStr = "ACL_FLOAT8_E4M3FN"; break;
        case ACL_FLOAT8_E8M0: dtypeStr = "ACL_FLOAT8_E8M0"; break;
        default: dtypeStr = "ACL_" + std::to_string(static_cast<int>(dtype));
    }
    OP_LOGI("Tensor [%s] - Shape: %s, DataType: %s", 
             tensorName.c_str(), 
             GetAclTensorShapeStr(tensor).c_str(),
             dtypeStr.c_str());
}

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerMlaPrologV3GetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk, const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos,
    aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional, const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    bool queryNormFlag, int64_t weightQuantMode, int64_t kvCacheQuantMode, int64_t queryQuantMode, int64_t ckvkrRepoMode,
    int64_t quantScaleRepoMode, int64_t tileSize, double qcQrScale, double kcScale, const aclTensor *queryOut,
    const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOut, const aclTensor *queryNormOut, const aclTensor *dequantScaleQNormOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMlaPrologV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
        inner_ = nullptr;
        name_ = varName;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner_ = aclCreateTensor(shape.data(), shape.size(),
                dataType, shape.data(), 0, ACL_FORMAT_ND,
                shape.data(), shape.size(), static_cast<void *>(&addr));
            output = inner_;
            // 打印创建的空Tensor信息
            OP_LOGI("Created empty Tensor [%s] - Shape: %s, DataType: %s", 
                     name_.c_str(),
                     GetAclTensorShapeStr(inner_).c_str(),
                     DataTypetoString(dataType).c_str());
        } else {
            // 打印已存在的Tensor信息
            LogAclTensorInfo(name_, output);
        }
    }

    ~TensorHolder() {
        if (inner_) {
            // 打印销毁Tensor的日志
            OP_LOGI("Destroy Tensor [%s]", name_.c_str());
            aclDestroyTensor(inner_);
            inner_ = nullptr;
        }
    }
    
    void CheckTensorConditionalNotNull(bool conditional) const {
        if (inner_ && conditional) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Check %s != nullptr failed! Tensor info: %s", 
                     name_.c_str(), GetAclTensorShapeStr(inner_).c_str());
        } else if (!inner_ && !conditional) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Check %s == nullptr failed!", name_.c_str());
        }
    }

    bool IsTensorNotNull() const {
        return inner_ == nullptr;
    }

private:
    // 辅助函数 - 转换数据类型为字符串
    std::string DataTypetoString(aclDataType dtype) const {
        switch (dtype) {
            case ACL_FLOAT: return "ACL_FLOAT";
            case ACL_BF16: return "ACL_BF16";
            case ACL_INT8: return "ACL_INT8";
            case ACL_FLOAT8_E4M3FN: return "ACL_FLOAT8_E4M3FN";
            case ACL_FLOAT8_E8M0: return "ACL_FLOAT8_E8M0";
            default: return "UNKNOWN(" + std::to_string(static_cast<int>(dtype)) + ")";
        }
    }

    const aclTensor *inner_;
    std::string name_;
};

aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    aclTensor *kvCacheRef,
    aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional,
    const aclTensor *kNopeClipAlphaOptional,
    double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv,
    char *cacheModeOptional,
    int64_t weightQuantMode,
    int64_t kvCacheQuantMode,
    int64_t queryQuantMode,
    int64_t ckvkrRepoMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    double qcQrScale,
    double kcScale,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 打印函数入口及关键参数
    OP_LOGI("===== Enter aclnnMlaPrologV3WeightNzGetWorkspaceSize =====");
    OP_LOGI("Key params - weightQuantMode: %lld, kvCacheQuantMode: %lld, ckvkrRepoMode: %lld, tileSize: %lld",
             weightQuantMode, kvCacheQuantMode, ckvkrRepoMode, tileSize);
    OP_LOGI("rmsnormEpsilonCq: %lf, rmsnormEpsilonCkv: %lf, qcQrScale: %lf, kcScale: %lf",
             rmsnormEpsilonCq, rmsnormEpsilonCkv, qcQrScale, kcScale);
    
    // 打印核心输入Tensor信息
    LogAclTensorInfo("tokenX", tokenX);
    LogAclTensorInfo("weightDq", weightDq);
    LogAclTensorInfo("weightUqQr", weightUqQr);
    LogAclTensorInfo("weightUk", weightUk);
    LogAclTensorInfo("weightDkvKr", weightDkvKr);
    LogAclTensorInfo("rmsnormGammaCq", rmsnormGammaCq);
    LogAclTensorInfo("rmsnormGammaCkv", rmsnormGammaCkv);
    LogAclTensorInfo("ropeSin", ropeSin);
    LogAclTensorInfo("ropeCos", ropeCos);
    LogAclTensorInfo("kvCacheRef", kvCacheRef);
    LogAclTensorInfo("krCacheRef", krCacheRef);

    // 打印可选输入Tensor信息
    LogAclTensorInfo("cacheIndexOptional", cacheIndexOptional);
    LogAclTensorInfo("dequantScaleXOptional", dequantScaleXOptional);
    LogAclTensorInfo("actualSeqLenOptional", actualSeqLenOptional);

    // 打印输出Tensor信息
    LogAclTensorInfo("queryOut", queryOut);
    LogAclTensorInfo("queryRopeOut", queryRopeOut);

    printf("进入aclnnMlaPrologV3WeightNzGetWorkspaceSize：mla_prolog_v3\\op_host\\op_api\\aclnn_mla_prolog_v3_weight_nz.cpp\n");
    const int WEIGHT_QUANT_MODE_NO_QUANT = 0;
    const int WEIGHT_QUANT_MODE_PARTIAL_QUANT = 1;
    const int WEIGHT_QUANT_MODE_FULL_QUANT = 2;
    const int WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT = 3;
    const int KV_CACHE_QUANT_MODE_NO_QUANT = 0;
    const int KV_CACHE_QUANT_MODE_PER_TENSOR = 1;
    const int KV_CACHE_QUANT_MODE_PER_CHANNEL = 2;
    const int KV_CACHE_QUANT_MODE_PER_TILE = 3;

    auto dequantScaleQNopeHolder = TensorHolder(dequantScaleQNopeOutOptional, aclDataType::ACL_FLOAT, std::string("dequantScaleQNopeOut"));
    aclDataType queryNormDataType = weightQuantMode == WEIGHT_QUANT_MODE_NO_QUANT ? aclDataType::ACL_BF16 : aclDataType::ACL_INT8;
    aclDataType dequantScaleQNormDataType = weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT ? aclDataType::ACL_FLOAT8_E8M0 : aclDataType::ACL_FLOAT;
    if (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
        queryNormDataType = aclDataType::ACL_FLOAT8_E4M3FN;
    }
    auto queryNormHolder = TensorHolder(queryNormOutOptional, queryNormDataType, std::string("queryNormOut"));
    auto dequantScaleQNormHolder = TensorHolder(dequantScaleQNormOutOptional, dequantScaleQNormDataType, std::string("dequantScaleQNormOut"));
    
    // 打印创建后的可选输出Tensor信息
    LogAclTensorInfo("dequantScaleQNopeOutOptional", dequantScaleQNopeOutOptional);
    LogAclTensorInfo("queryNormOutOptional", queryNormOutOptional);
    LogAclTensorInfo("dequantScaleQNormOutOptional", dequantScaleQNormOutOptional);

    if (dequantScaleQNopeOutOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor dequantScaleQNopeOu!");
        return ge::GRAPH_FAILED;
    }
    if (queryNormOutOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor queryNormOut!");
        return ge::GRAPH_FAILED;
    }
    if (dequantScaleQNormOutOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor dequantScaleQNormOut!");
        return ge::GRAPH_FAILED;
    }
    // weightQuantMode == 2:全量化场景, weightQuantMode == 3:mxfp8全量化场景, kvCacheQuantMode == 1:KV_PER_TENSOR量化场景
    dequantScaleQNopeHolder.CheckTensorConditionalNotNull((weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT || weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) && kvCacheQuantMode == KV_CACHE_QUANT_MODE_PER_TENSOR); 
    bool queryNormFlag = queryNormHolder.IsTensorNotNull();
    // 打印queryNormFlag及量化模式相关信息
    OP_LOGI("queryNormFlag: %s, weightQuantMode: %lld (NO_QUANT:0, PARTIAL:1, FULL:2, MXFP8_FULL:3)",
             queryNormFlag ? "true" : "false", weightQuantMode);
    
    // weightQuantMode != 0:量化场景
    dequantScaleQNormHolder.CheckTensorConditionalNotNull(weightQuantMode != WEIGHT_QUANT_MODE_NO_QUANT && queryNormFlag);
    
    printf("即将调用aclnnMlaPrologV3WeightNzGetWorkspaceSize：mla_prolog_v3\\op_host\\op_api\\aclnn_mla_prolog_v3_weight_nz.cpp\n");
    aclnnStatus ret = aclnnInnerMlaPrologV3GetWorkspaceSize(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCacheRef, krCacheRef,
        cacheIndexOptional, dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional,
        dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional, smoothScalesCqOptional, actualSeqLenOptional, kNopeClipAlphaOptional,
        rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheModeOptional,
        queryNormFlag, weightQuantMode, kvCacheQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize,
        qcQrScale, kcScale, queryOut, queryRopeOut,
        dequantScaleQNopeOutOptional, queryNormOutOptional, dequantScaleQNormOutOptional,
        workspaceSize, executor);
    
    // 打印调用结果及workspaceSize
    OP_LOGI("aclnnInnerMlaPrologV3GetWorkspaceSize return: %d, workspaceSize: %llu",
             ret, workspaceSize ? *workspaceSize : 0);
    OP_LOGI("===== Exit aclnnMlaPrologV3WeightNzGetWorkspaceSize =====");
    
    return ret;
}

aclnnStatus aclnnMlaPrologV3WeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    // 打印函数入口及参数
    OP_LOGI("===== Enter aclnnMlaPrologV3WeightNz =====");
    OP_LOGI("workspaceSize: %llu, executor: %p, stream: %p",
             workspaceSize, executor, stream);
    
    printf("即将调用aclnnInnerMlaPrologV3：mla_prolog_v3\\op_host\\op_api\\aclnn_mla_prolog_v3_weight_nz.cpp\n");
    aclnnStatus ret = aclnnInnerMlaPrologV3(workspace, workspaceSize, executor, stream);
    
    // 打印调用结果
    OP_LOGI("aclnnInnerMlaPrologV3 return: %d", ret);
    OP_LOGI("===== Exit aclnnMlaPrologV3WeightNz =====");
    
    return ret;
}

} // namespace

#ifdef __cplusplus
}
#endif