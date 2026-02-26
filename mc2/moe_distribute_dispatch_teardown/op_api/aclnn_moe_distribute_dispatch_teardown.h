/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_DISPATCH_TEARDOWN_
#define OP_API_INC_MOE_DISTRIBUTE_DISPATCH_TEARDOWN_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoeDistributeDispatchTeardown功能，接受EP域的alltoallv通信结果，并整理结果输出
 * @brief aclnnMoeDistributeDispatchTeardown的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。本卡发送的token数据。
 * @param [in] y: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。本卡发送的token数据。
 * @param [in] expertIds:
 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。aclnnMoeDistributeDispatchSetUp的输出，通信的cmd信息。
 * @param [in] commCmdInfo: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。每个token的topK个专家索引。
 * @param [in] xActiveMaskOptional:
 表示token是否参与通信，Tensor，数据类型bool，必须要1维，数据格式支持ND。可传入有效数据或空指针，空指针代表所有token都参与通信。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。字符串长度范围为[1, 128)，不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。在昇腾910_93场景中取值支持8/16/32/64/128/144/256/288。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。取值范围[1, 512]，且
    满足moeExpertNum%(epWorldSize-sharedExpertRankNum)等于0。
 * @param [in] expertShardType: 计算可选输入，int。专家共享类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。取值范围[0, 1]。0表示无共享专家。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家数量。支持传0表示无共享专家卡，不为0时需满足
    sharedExpertRankNum<epWorldSize且epWorldSize%sharedExpertRankNum等于0。
 * @param [in] quantMode: 计算可选输入，int，量化模式，支持0：非量化，2：动态量化。
 * @param [in] globalBs: 计算可选输入，int。在910_93中，当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize 或
    globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。
 * @param [in] expertTokenNumsType:
 计算可选输入，int。输出expertTokenNums中的值语义类型。支持0：expertTokenNumsOut中的输出
    为每个专家处理的token数的前缀和，1：expertTokenNumsOut中的输出为每个专家处理的token数量。
 * @param [in] comm_type: 计算输入，int。0系统自选择 1 AIV-SDMA
 * @param [in] commAlg: 计算输入，char*。通信亲和内存布局算法。当前版本不支持，传空指针即可。
 * @param [out] expandXOut: 计算输出，Tensor，必选输出，数据类型支持float16, bfloat16,
 int8，仅支持2维，数据格式支持ND。根据 expertIdx进行扩展过的token特征。
 * @param [out] dynamicScalesOut:
 计算输出，Tensor，必选输出，数据类型float32，仅支持1维，数据格式支持ND。quantMode为0时输出为空。
 * @param [out] expertTokenNumsOut:
 计算输出，Tensor，必选输出，数据类型int64，仅支持1维，数据格式支持ND。每个专家收到的token个数。
 * @param [out] epRecvCountsOut:
 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。表示从各卡接收的token数。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
    const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t commType, char* commAlg,
    aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeDispatchTeardown的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnMoeDistributeDispatchTeardownGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchTeardown(void* workspace, uint64_t workspaceSize,
                                                         aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_DISPATCH_