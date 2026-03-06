#include "aclnn_mhc_sinkhorn.h"
#include "mhc_sinkhorn.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t MAX_DIM = 8;

static bool CheckNotNull(const aclTensor* x, const aclTensor* output, const aclTensor* normOut, const aclTensor* sumOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(normOut, return false);
    OP_CHECK_NULL(sumOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* output, const aclTensor* normOut, const aclTensor* sumOut)
{
    // 检查x的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    // 检查output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
    // 检查normOut的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(normOut, DTYPE_SUPPORT_LIST, return false);
    // 检查sumOut的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(sumOut, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    // 输入输出的格式需要一致
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. self [%s], out [%s].",
            ToString(self->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* x, const aclTensor* output, const aclTensor* normOut, const aclTensor* sumOut, int64_t numIters,
    int64_t outFlag)
{
    if (x->IsEmpty()) {
        return true;
    }
    // 校验self的shape是否等于out的shape
    OP_CHECK_SHAPE_NOT_EQUAL(x, output, return false);
    // 最大维度限制
    OP_CHECK_MAX_DIM(x, MAX_DIM, return false);

    // numIters在1~100范围内
    if (numIters <= 0 || numIters > 100) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "numIters value error, numIters must in 1 to 100, but got numIters = %ld .",
            numIters);
        return false;
    }

    // outFlag为0或1
    if (outFlag != 0 || outFlag != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "outFlag value error, outFlag must be 0 or 1, but got outFlag = %ld .", outFlag);
        return false;
    }

    // 维度必须是3或4
    auto xShape = x->GetViewShape();
    auto xDim = xShape.GetDimNum();
    if (xDim != 3 && xDim != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim must be 3 or 4, but got Dim = %ld .", xDim);
        return false;
    }

    // n0等于n1 且n为4,6,8
    int64_t n0 = 0;
    int64_t n1 = 0;
    if (xDim == 3) {
        n0 = xShape.GetDim(DIM_ONE);
        n1 = xShape.GetDim(DIM_TWO);
    } else if (xDim == 4) {
        n0 = xShape.GetDim(DIM_TWO);
        n1 = xShape.GetDim(DIM_THREE);
    }
    if ((n0 != n1) || (n0 != 4 && n0 != 6 && n0 != 8)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "n0 must equal n1, and n must be 4 or 6 or 8, but got n0 = %ld, n1 = %ld", n0, n1);
        return false;
    }

    return true;
}

static inline aclnnStatus CheckParams(
    const aclTensor* x, int64_t outFlag, float eps, int64_t numIters, const aclTensor* output, const aclTensor* normOut,
    const aclTensor* sumOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, output, normOut, sumOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, output, normOut, sumOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(x, output, normOut, sumOut, numIters, outFlag), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入输出format是否一致
    CHECK_RET(CheckFormat(x, output), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(
    const aclTensor* x, int64_t outFlag, float eps, int64_t numIters, aclTensor* output, aclTensor* normOut,
    aclTensor* sumOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGI("Enter aclnnMhcSinkhorn WorkspaceSize");
    L2_DFX_PHASE_1(aclnnMhcSinkhorn, DFX_IN(x, outFlag, eps, numIters), DFX_OUT(output, normOut, sumOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, outFlag, eps, numIters, output, normOut, sumOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入x转换成连续的tensor
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto kernelOut =
        l0op::MhcSinkhorn(xContiguous, outFlag, eps, numIters, output, normOut, sumOut, uniqueExecutor.get());

    // 固定写法，将计算结果拷贝到输出outRef上
    auto viewCopyResult = l0op::ViewCopy(kernelOut, output, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhorn(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcSinkhorn);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
