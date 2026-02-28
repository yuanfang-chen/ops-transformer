# 算子多平台适配指南

本指南介绍算子在多平台间迁移的适配要点与方案。以Atlas A2 系列迁移至Ascend 950系列为例，对比硬件架构差异项及所涉适配点，并提供相关算子适配样例。


## 一、硬件架构及规格参数对比
### Atlas A2 系列硬件架构
[A2硬件架构图]

### Ascend 950 系列硬件架构
[A5硬件架构图]

### 芯片规格参数对比
<table thread>
  <tr>
    <th colspan="2" style="width: 25%;">规格项</th>
    <th style="width:25%;">Atlas A2</th>
    <th style="width:25%;">Ascend 950PR</th>
    <th style="width:25%;">Ascend 950DT</th>
  </tr>
  <tr>
    <td rowspan="4">AICore</td>
    <td>核数</td>
    <td>24</td>
    <td>32</td>
    <td>32</td>
  </tr>
  <tr>
    <td>频率</td>
    <td>1.8</td>
    <td>1.65</td>
    <td>1.65</td>
  </tr>
  <tr>
    <td>Cube算力规格</td>
    <td>353T/376T @BF16,FP16</td>
    <td>426T@BF16,FP16 757T@FP8,HIFP8,MXFP8,INT8 1514T@MXFP4</td>
    <td>426T@BF16,FP16 757T@FP8,HIFP8,MXFP8,INT8 1514T@MXFP4</td>
  </tr>
  <tr>
    <td>Vector算力规格(FP16)</td>
    <td>23.5T</td>
    <td>54T</td>
    <td>54T</td>
  </tr>
  <tr>
    <td rowspan="4">SOC</td>
    <td>L2 容量(MB)</td>
    <td>192</td>
    <td>128</td>
    <td>128</td>
  </tr>
  <tr>
    <td>L2 带宽(read)</td>
    <td>4.4TB/s</td>
    <td>5.28TB/s</td>
    <td>5.28TB/s</td>
  </tr>
  <tr>
    <td>跨Die带宽(单向有效带宽)</td>
    <td>180GB/s(1981)</td>
    <td>1.44TB/s(2Die)</td>
    <td>1.6TB/s(2Die)</td>
  </tr>
  <tr>
    <td>AICPU</td>
    <td>Linx910M 2GHz, 8C8T</td>
    <td>Linx816 1.5GHz, 8C16T</td>
    <td>Linx816 1.5GHz, 8C16T</td>
  </tr>
  <tr>
    <td rowspan="2">Memory</td>
    <td>Memory 容量(GB)</td>
    <td>64</td>
    <td>128</td>
    <td>144</td>
  </tr>
  <tr>
    <td>Memory 带宽</td>
    <td>1.6TB/s</td>
    <td>1.6TB/s</td>
    <td>4TB/s</td>
  </tr>
  <tr>
    <td rowspan="3">IO规格</td>
    <td>NvLink/UB-C/G</td>
    <td>392GB/s</td>
    <td>2016GB/s</td>
    <td>2016GB/s</td>
  </tr>
  <tr>
    <td>PCIE</td>
    <td>128GB/s</td>
    <td>128GB/s</td>
    <td>128GB/s</td>
  </tr>
  <tr>
    <td>RoCE/UBoE</td>
    <td>200Gbps</td>
    <td>400Gbps</td>
    <td>400Gbps</td>
  </tr>
  <tr>
    <td rowspan="1">TDP功耗</td>
    <td>标称典型功耗</td>
    <td>300~500W</td>
    <td>650W</td>
    <td>750W</td>
  </tr>
</table>

## 二、硬件能力变更引入适配点

<table thread>
  <tr>
    <th style="width: 25%;">硬件单元</th>
    <th style="width:35%;">硬件能力变更</th>
    <th style="width:40%;">典型影响范围</th>
  </tr>
  <tr>
    <td rowspan="5">搬运单元</td>
    <td>删除L1到GM的数据通路</td>
    <td>to be filled</td>
  </tr>
  <tr>
    <td>删除GM到L0A、L0B的数据通路</td>
    <td>to be filled</td>
  </tr>
  <tr>
    <td>ND DMA灵活数据搬运，支持随路ND->NZ转换</td>
    <td>to be filled</td>
  </tr>
  <tr>
    <td>支持Cube->Vector高效内部 数据通路:L1<->UB、L0C->UB、FIXP->UB</td>
    <td>to be filled</td>
  </tr>
  <tr>
    <td>引入集合通信加速器CCU1.0</td>
    <td>通算融合算子在Eager模式下调整HcclServerType；在Graph模式下改用CCU系列GE接口</td>
  </tr>
  <tr>
    <td rowspan="4">计算单元</td>
    <td>Vector由Membase架构切为Regbase架构</td>
    <td>原依赖Membase的访存pattern、对齐方式、寄存器数量假设等需要重新审查；模板/tiling可能需要更新到Regbase版本</td>
  </tr>
  <tr>
    <td>SIMT</td>
    <td>to be filled</td>
  </tr>
  <tr>
    <td>Cube不再支持int4b_t</td>
    <td>所有使用int4b_t的算子需要切换到支持的数据类型（如int8），并更新量化解算逻辑</td>
  </tr>
  <tr>
    <td>不支持4：2稀疏矩阵计算</td>
    <td>原依赖4：2稀疏特性提速的kernel需要改为稠密 或其他支持的稀疏策略，并更新性能预期说明</td>
  </tr>
  <tr>
  <td rowspan="1">存储单元</td>
    <td>Local Buffer内存改进：Cube L0C 256KB、Vector UB 256KB</td>
    <td>to be filled</td>
  </tr>
  <td rowspan="1">其他</td>
    <td>同地址访问优化</td>
    <td>涉及矩阵乘相关算子的模板可优化</td>
  </tr>
</table>

**其他**
1、同地址访问优化

## 三、推荐迁移步骤
1、确认算子涉及的计算单元类型（Cube/Vector/SIMT）和使用精度类型（FP/INT/稀疏等）。
2、确认涉及的数据搬运路径（ND->NZ、GM<->Lx、集合通信等）是否在平台间存在差异。
3、按硬件能力变更点逐项对照修改（Vector 架构、Cube 支持精度、L0/UB 容量、CCU 通信等）。
4、参考算子迁移样例调整/补齐 Atlas A2/Ascend 950 分支逻辑。

## 四、算子迁移样例

### MC2 -- MatmulAllReduce

#### 1、同地址访问优化

基于A2硬件架构同地址访问冲突较多，Matmul部分采用错位分核模板；950系列对同地址访问做了相关优化，为获得更高性能收益，Matmul部分使用自适应滑窗模板，核心收益为减少同地址访问冲突带来的stall，提升Cube/Vector pipeline利用率。故算子适配调整Matmul模板，以fp非量化场景为例，A2使用`Mc2MatmulBaseKernel`；950使用`Mc2MatmulAswKernel`。涉及代码文件：[matmul_all_reduce.cpp](../../../mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp)

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

## 五、常见问题与性能调优建议（FAQ/性能小贴士）

若算子在Ascend 950 上性能不升反降时，可优先排查：
1、是否仍然使用Atlas A2的错位分核模板
2、是否未开启CCU通信 仍走AICPU
3、tiling是否沿用了Atlas A2 的L0/UB假设，导致950的更大缓冲未被充分利用
...