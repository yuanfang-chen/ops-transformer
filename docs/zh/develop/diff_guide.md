# 算子适配差异项对比

本指南以`MatmulAllReduce`算子开发为例，介绍算子多平台适配（Atlas A2 --> Ascend 950PR/Ascend 950DT）的案例与典型问题。

|差异项|A2|950|
|-----|----|----|
|Matmul模板|Mc2MatmulBaseKernel|Mc2MatmulAswKernel|
|通信配置|hccl初始化V1接口|hccl初始化V2接口|
|通信方式|AICPU通信|CCU通信|
|其余| -- |原型与config增加950相关,doc文档更新等|

## Matmul模板差异

以fp非量化场景为例，A2使用`Mc2MatmulBaseKernel`；950使用`Mc2MatmulAswKernel`。涉及代码文件：[matmul_all_reduce.cpp](../../../mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp)

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

## 通信配置差异

**hccl初始化**

A2调用hccl初始化V1版本接口；950调用V2版本接口。涉及代码文件：[matmul_all_reduce_base.h](../../../mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_base.h)

```CPP
// ...
// A2
hccl_.Init(GetHcclContext<0>());

// 950
hccl_.InitV2(GetHcclContext<0>(), tilingData_);
// ...
```

**cGm地址计算**

950需要额外申请一块WorkSpace 存放matmul输出，cGm地址要另行计算。涉及代码文件：
[matmul_all_reduce_tiling_base.cpp](../../../mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp)

```CPP
// ...
// 950
if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
 	gmcFloat = static_cast<uint64_t>(MutableRCSTilingData().rankM) *
 	static_cast<uint64_t>(MutableRCSTilingData().rankN) *
 	static_cast<uint64_t>(args_.outputDtypeSize);
}
// ...
```

[matmul_all_reduce_base.h](../../../mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_base.h)
```CPP
// ...
// A2
addrs_->cGM = hccl_.GetWindowsInAddr(hccl_.GetRankId());

// 950
addrs_->cGM = addrs_->workspaceGM + paramInTiling_->nd2NzWorkLen + paramInTiling_->biasLen;
// ...
```

## 通信方式差异


A2起AICPU通信；950起CCU通信。

**Eager模式**

由NnopbaseSetHcclServerType设置为NNOPBASE_HCCL_SERVER_AICPU或NNOPBASE_HCCL_SERVER_TYPE_CCU区分。涉及代码文件：[aclnn_matmul_all_reduce.cpp](../../../mc2/matmul_all_reduce/op_api/aclnn_matmul_all_reduce.cpp)

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

静态图GE侧创建通信task的任务类型要区分，A2为aicpu kfc server + kfc_stream；950为ccu server + ccu_stream。涉及代码文件：[matmul_all_reduce_gen_task.cpp](../../../mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

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

## 其余适配项

**算子原型**

算子原型文件需添加950相关OpAICoreConfig。
```CPP
explicit MatmulAllReduce(const char* name) : OpDef(name)
{
    // ...
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
        .ExtendCfgInfo("opFile.value", "matmul_all_reduce_apt");
    this->AICore().AddConfig("ascend950", aicore_config);
    this->MC2().HcclGroup("group");
    // ...
}
```

**config**
算子config增加950相关binary.json与simplified_key.ini，涉及文件：[matmul_all_reduce_binary.json](../../../mc2/matmul_all_reduce/op_host/config/ascend950/matmul_all_reduce_binary.json) [matmul_all_reduce_simplified_key.ini](../../../mc2/matmul_all_reduce/op_host/config/ascend950/matmul_all_reduce_simplified_key.ini)


**doc文档**
aclnn MD文档中，更新产品支持情况，涉及文件：[aclnnMatmulAllReduce.md](../../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md) 