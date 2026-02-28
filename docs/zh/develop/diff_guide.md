# 算子多平台适配指南

本指南介绍算子在多平台间迁移的适配要点与方案。以Atlas A2 系列迁移至Ascend 950系列为例，对比硬件架构差异项及所涉适配点，并提供相关算子适配样例。


## 一、硬件架构及规格参数对比
### Atlas A2 系列硬件架构
[A2硬件架构图]

### Ascend 950 系列硬件架构
[A5硬件架构图]

### 芯片规格参数对比
[对比表格]

## 二、硬件能力变更引入适配点
**搬运单元**
1、删除L1到GM的数据通路
2、删除GM到L0A、L0B的数据通路
3、ND DMA灵活数据搬运，支持随路ND->NZ转换
4、支持Cube->Vector高效内部数据通路：L1<->UB、L0C->UB、FIXP->UB
5、引入集合通信加速器CCU1.0

**计算单元**
1、Vector由Membase架构切为Regbase架构
2、SIMT
3、Cube不再支持int4b_t
4、不支持4：2稀疏矩阵计算

**存储单元**
1、Local Buffer内存改进：Cube L0C 256KB、Vector UB 256KB

**其他**
1、同地址访问优化

## 三、算子迁移样例

### MC2 -- MatmulAllReduce

#### 1、同地址访问优化

基于A2硬件架构同地址访问冲突较多，Matmul部分采用错位分核模板；950系列对同地址访问做了相关优化，为获得更高性能收益，Matmul部分使用自适应滑窗模板，故算子适配调整Matmul模板，以fp非量化场景为例，A2使用`Mc2MatmulBaseKernel`；950使用`Mc2MatmulAswKernel`。涉及代码文件：[matmul_all_reduce.cpp](../../../mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp)

```CPP
// ...
// A2
template<int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_FP_MM>
__global__ __aicore__ void fp_matmul_all_reduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
    // ...
    if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM_CUBE_ONLY) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseKernel, Mc2CoreType::ON_CUBE);
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM && EMPTY_INPUT == MATMUL_ALLREDUCE_EMPTY_INPUT_T) {
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    } else if constexpr (MM_TYPE == MATMUL_ALLREDUCE_MM_TYPE_FP_MM && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    }
    // ...
}

// 950
template<TPL_APT_PARAMS_FP_MM>
__global__ __aicore__ void fp_matmul_all_reduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
    // ...
    if constexpr (!MATMUL_WITH_ADD) {
        INVOKE_MC2_910_OP_IMPL(
            Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2CoreType::ON_CUBE);
    } else if constexpr (MATMUL_WITH_ADD) {
        INVOKE_MC2_910_OP_IMPL(
            Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    }
    // ...
}
// ...
```

#### 2、通信优化
950引入集合通信加速器CCU1.0，降低了访存需求，减少了调度时延，为有效利用该特性，将算子跨片通信方式由A2的AICPU改为CCU通信。

**Eager模式**

设置NnopbaseSetHcclServerType枚举值，A2为NNOPBASE_HCCL_SERVER_AICPU，950为NNOPBASE_HCCL_SERVER_TYPE_CCU。涉及代码文件：[aclnn_matmul_all_reduce.cpp](../../../mc2/matmul_all_reduce/op_api/aclnn_matmul_all_reduce.cpp)

```CPP
// ...
aclnnStatus aclnnMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // ...
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    // ...
    return ACLNN_SUCCESS;
}
```

**Graph模式**

静态图GE侧创建通信task的任务类型，A2为aicpu kfc server + kfc_stream；950为ccu server + ccu_stream。涉及代码文件：[matmul_all_reduce_gen_task.cpp](../../../mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
ge::Status MatmulAllReduceCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    // A2
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}
// ...
```

静态图GenTask调用接口有区别，流程有差异。涉及代码文件：[matmul_all_reduce_gen_task.cpp](../../../mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
// A2
ge::Status MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(
    const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>>& tasks) {
    // ...
    // aicpu task
    ge::KernelLaunchInfo aicpu_task =
        ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME.c_str(), KERNEL_NAME_V1.c_str());
    // ...
}

// 950
ge::Status Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    // ccu task
    ge::KernelLaunchInfo ccuTask = ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
    // ...
}

ge::Status MatmulAllReduceGenTaskFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    // A2
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}
// ...
```

### 样例算子2
### 样例算子3
### ...