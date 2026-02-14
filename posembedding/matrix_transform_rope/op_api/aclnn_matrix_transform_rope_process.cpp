/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "aclnn_matrix_transform_rope_utils.h"
#include "aclnn_matrix_transform_rope_base_handler.h"
#include "aclnn_matrix_transform_rope.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "matrix_transform_rope.h"

using namespace op;
using namespace matrix_transform_rope;

/**
 * @brief HandlerFactory 类，用于根据 SOC 版本创建对应的 Handler
 *
 * 设计说明：
 * 1. 使用工厂模式管理不同 SOC 版本的 Handler
 * 2. 根据 NpuArch 注册和获取 Handler
 * 3. 支持动态添加新的 SOC 版本支持
 */
class MatrixTransformRopeHandlerFactory {
private:
    std::unordered_map<NpuArch, std::unique_ptr<MatrixTransformRopeHandler>> handlers_;

public:
    /**
     * @brief 注册 Handler
     * @param npuArch NPU 架构类型
     * @param handler 对应的 Handler 实例
     */
    void registerHandler(NpuArch npuArch, std::unique_ptr<MatrixTransformRopeHandler> handler)
    {
        handlers_[npuArch] = std::move(handler);
    }

    /**
     * @brief 获取 Handler
     * @param npuArch NPU 架构类型
     * @return 对应的 Handler 指针，如果未找到则返回 nullptr
     */
    MatrixTransformRopeHandler *getHandler(NpuArch npuArch)
    {
        auto it = handlers_.find(npuArch);
        return it != handlers_.end() ? it->second.get() : nullptr;
    }
};

/**
 * @brief 公共处理函数，根据 SOC 版本路由到对应的 Handler
 *
 * 设计说明：
 * 1. 获取当前 NPU 架构
 * 2. 注册不同 SOC 版本的 Handler
 * 3. 调用对应 Handler 的处理流程
 *
 * @param interfaceName 接口名称
 * @param params 参数结构体
 * @param workspaceSize 工作空间大小指针
 * @param executor 执行器指针的指针
 * @return aclnnStatus 状态码
 */
static aclnnStatus aclnnMatrixTransformRopeGetWorkspaceSizeCommon(const char* interfaceName,
    MatrixTransformRopeParamsBase &params, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    MatrixTransformRopeHandlerFactory factory;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();

    // 注册 DAV_2201 版本的 Handler
    factory.registerHandler(NpuArch::DAV_2201,
        std::make_unique<matrix_transform_rope_base::MatrixTransformRopeBaseHandler>());

    // 获取当前 SOC 版本对应的 Handler
    if (auto *handler = factory.getHandler(npuArch)) {
        handler->Initialize(interfaceName, params, workspaceSize, executor);
        return handler->Process();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s: soc version is not support", interfaceName);
    }

    return ACLNN_ERR_PARAM_INVALID;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMatrixTransformRopeGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 输入tensor，维度为4维，shape为[B, N, S, D]、[B, S, N, D]、[S, B, N, D]，数据类型支持：BF16/FP16/FP32, 支持ND格式。
 * @param [in] cos: 输入tensor，维度为4维，数据类型支持：BF16/FP16/FP32, 支持ND格式。当x为BNSD时，cos、sin支持11SD、B1SD、BNSD，当x为BSND时，cos、sin支持1S1D、BS1D、BSND，当x为SBND时，cos、sin支持S11D、SB1D、SBND。
 * @param [in] sin: 输入tensor，shape为4维，数据类型支持：BF16/FP16/FP32, 支持ND格式。当x为BNSD时，cos、sin支持11SD、B1SD、BNSD，当x为BSND时，cos、sin支持1S1D、BS1D、BSND，当x为SBND时，cos、sin支持S11D、SB1D、SBND。
 * @param [in] rotate: 输入tensor，维度为4维，shape为[D, D]，数据类型支持：BF16/FP16/FP32, 支持ND格式。
 * @param [out] out: 输入tensor，维度为4维，shape为[B, N, S, D]、[B, S, N, D]、[S, B, N, D]，和x保持一致，数据类型支持：BF16/FP16/FP32, 支持ND格式。
 * @param [out] workspaceSize: 返回需要在 npu device 侧申请的 workspace 大小
 * @param [out] executor: 返回 op 执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnMatrixTransformRopeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
    const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMatrixTransformRope,
                    DFX_IN(x, cos, sin, rotate),
                    DFX_OUT(out));

    // 使用 Builder 模式构建参数结构体
    MatrixTransformRopeParamsBase params =
        MatrixTransformRopeParamsBuilder::Create(x, cos, sin, rotate, out)
        .Build();

    // 调用公共接口
    return aclnnMatrixTransformRopeGetWorkspaceSizeCommon(__FUNCTION__, params, workspaceSize, executor);
}

/**
 * @brief aclnnMatrixTransformRope 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口获取
 * @param [in] executor: op 执行器，包含了算子计算流程
 * @param [in] stream: acl stream 流
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnMatrixTransformRope(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatrixTransformRope);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in MatrixTransformRope launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
