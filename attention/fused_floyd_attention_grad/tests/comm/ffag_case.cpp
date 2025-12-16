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
 * \file ffag_case.cpp
 * \brief FusedFloydAttentionGrad 测试用例.
 */
#include "ffag_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define FFAG_KERNEL_PARAM                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key1, __gm__ uint8_t * value1, __gm__ uint8_t * key2, \
    __gm__ uint8_t * value2, __gm__ uint8_t * dy, __gm__ uint8_t * attenMask, __gm__ uint8_t * softmaxMax,              \
    __gm__ uint8_t * softmaxSum, __gm__ uint8_t * attentionIn, __gm__ uint8_t * dq, __gm__ uint8_t * dk1, \
    __gm__ uint8_t * dk2, __gm__ uint8_t * dv1, __gm__ uint8_t * dv2, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*FfagKernelFunc) FFAG_KERNEL_PARAM;

extern "C" __global__ __aicore__ void fused_floyd_attention_grad FFAG_KERNEL_PARAM;

using namespace ops::adv::tests::ffag;
using Tensor = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunFusedFloydAttentionGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<Tensor *> &inputs,
                             std::vector<Tensor *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FfagKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(),
                outputs[0]->GetDevData(), outputs[1]->GetDevData(), outputs[2]->GetDevData(), outputs[3]->GetDevData(), outputs[4]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingFusedFloydAttentionGradStub(gert::TilingContext* context)
{
    auto* ffagCase = static_cast<FfagCase*>(Case::GetCurrentCase());
    if (ffagCase != nullptr) {
        FfagCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!ffagCase->DoOpTiling(p)) {
            return p.ret;
        }
        return ffagCase->ffagTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool FfagCase::InitParam()
{
    query = Tensor("query", {mParam.b, mParam.n, mParam.s1, mParam.s2, mParam.d}, "BNS1S2D", mParam.qkvDataType, ge::FORMAT_ND);
    key1 = Tensor("key1", {mParam.b, mParam.n, mParam.s1, mParam.s3, mParam.d}, "BNS1S3D", mParam.qkvDataType, ge::FORMAT_ND);
    key2 = Tensor("key2", {mParam.b, mParam.n, mParam.s3, mParam.s2, mParam.d}, "BNS3S2D", mParam.qkvDataType, ge::FORMAT_ND);
    value1 = Tensor("value1", {mParam.b, mParam.n, mParam.s1, mParam.s3, mParam.d}, "BNS1S3D", mParam.qkvDataType, ge::FORMAT_ND);
    value2 = Tensor("value2", {mParam.b, mParam.n, mParam.s3, mParam.s2, mParam.d}, "BNS3S2D", mParam.qkvDataType, ge::FORMAT_ND);
    dy = Tensor("dy", {mParam.b, mParam.n, mParam.s1, mParam.s2, mParam.d}, "BNS1S2D", mParam.qkvDataType, ge::FORMAT_ND);
    softmaxMax = Tensor("softmaxMax", {mParam.b, mParam.n, mParam.s1, mParam.s2, 8}, "BNS1S28", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    softmaxSum = Tensor("softmaxSum", {mParam.b, mParam.n, mParam.s1, mParam.s2, 8}, "BNS1S28", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    attentionIn = Tensor("attentionIn", {mParam.b, mParam.n, mParam.s1, mParam.s2, mParam.d}, "BNS1S2D", mParam.qkvDataType, ge::FORMAT_ND);
    if (mParam.attenMaskType == AttenMaskShapeType::B_1_S1_1_S3) {
        attenMask = Tensor("attenMask", {mParam.b, 1, mParam.s1, 1, mParam.s3}, "B_1_S1_1_S3", ge::DataType::DT_BOOL, ge::FORMAT_ND);
    }
    dq = Tensor("dq", {mParam.b, mParam.n, mParam.s1, mParam.s2, mParam.d}, "BNS1S2D", mParam.qkvDataType, ge::FORMAT_ND);
    dk1 = Tensor("dk1", {mParam.b, mParam.n, mParam.s1, mParam.s3, mParam.d}, "BNS1S3D", mParam.qkvDataType, ge::FORMAT_ND);
    dk2 = Tensor("dk2", {mParam.b, mParam.n, mParam.s3, mParam.s2, mParam.d}, "BNS3S2D", mParam.qkvDataType, ge::FORMAT_ND);
    dv1 = Tensor("dv1", {mParam.b, mParam.n, mParam.s1, mParam.s3, mParam.d}, "BNS1S3D", mParam.qkvDataType, ge::FORMAT_ND);
    dv2 = Tensor("dv2", {mParam.b, mParam.n, mParam.s3, mParam.s2, mParam.d}, "BNS3S2D", mParam.qkvDataType, ge::FORMAT_ND);
    return true;
}

bool FfagCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("FusedFloydAttentionGrad");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key1, &value1, &key2, &value2, &dy, &attenMask, &softmaxMax, &softmaxSum, &attentionIn});
    rst = rst && mCtx.SetOutputs({&dq, &dk1, &dv1, &dk2, &dv2});
    rst = rst && mCtx.SetTilingDataMaxSize(4096);
    rst = rst && mCtx.SetAttrs({
                                {"scale_value", mParam.scaleValue}});
    #ifdef SUPPORT_KERNEL
        rst = rst && mCtx.SetKernelRunCbf(RunFusedFloydAttentionGrad);
        rst = rst && mCtx.SetKernelMainFunc((void *)fused_floyd_attention_grad);
    #endif
    rst = rst && mOpInfo.SetContext(&mCtx);
    auto* platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }
    ffagTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFusedFloydAttentionGrad");
    if(ffagTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, ffag(%p)",ffagTilingFunc);
        return false;
    }
    IMPL_OP(FusedFloydAttentionGrad).Tiling(TilingFusedFloydAttentionGradStub);
    return rst;
}

bool FfagCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool FfagCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

FfagCase::FfagCase(const char *name, bool enable, const char *dbgInfo, OpInfo prompt, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(prompt)), mParam(std::move(param))
{
    this->mOpInfo.mName = "FusedFloydAttentionGrad";
}

FfagCase::FfagCase()
{
}

FfagCase::Param::Param()
{
}

FfagCase::Param::Param(int64_t pB, int64_t pN, int64_t pS1, int64_t pS2, int64_t pS3, int64_t pD,
                      float pScaleValue)
    : b(pB), n(pN), s1(pS1), s2(pS2), s3(pS3), d(pD),
      scaleValue(pScaleValue)
{
}

bool FfagCase::DoOpTiling(DoTilingParam &tilingParam) {
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}