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
 * \file expert_dispatch_tiling.cpp
 * \brief
 */
#include "expert_dispatch_tiling.h"
#include "../../../mc2/common/inc/mc2_log.h"
#include "../../../mc2/3rd/common/op_host/op_tiling/debug_tiling.h"

using Ops::Transformer::OpTiling::TilingBaseClass;

namespace optiling
{
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t SIZE_31 = 31;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t SPLIT_K_THRESHOLD = 512;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_ID_MAX = 10240;

const static int64_t INPUT_X_INDEX = 0;
const static int64_t INPUT_EXPERT_ID_INDEX = 1;
const static int64_t INPUT_SCALE_INDEX = 2;
const static int64_t OUTPUT_DISPATCHED_X_INDEX = 0;
const static int64_t OUTPUT_DISPATCHED_ROW_IDX_INDEX = 1;
const static int64_t OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2;
const static int64_t OUTPUT_EXPERT_TOTAL_COUNT_INDEX = 3;
const static int64_t OUTPUT_DISPATCHED_SCALE_INDEX = 4;
const static int64_t ATTR_EXPERT_RANGE_INDEX = 0;
const static int64_t ATTR_EXPERT_RANGE_DIM = 2;

const static uint64_t TILINGKEY_BASE = 1000000;
const static uint64_t SORT_CORE_TILINGKEY_BASE = 100000;
const static uint64_t HIST_TILINGKEY_BASE = 10000;
const static uint64_t GATHER_OUT_TILINGKEY_BASE = 1000;

#define CHECK_FAIL(context, cond, ...)                      \
    do {                                                    \
        if (cond) {                                         \
            OP_LOGE(context->GetNodeName(), ##__VA_ARGS__); \
            return ge::GRAPH_FAILED;                        \
        }                                                   \
    } while (0)

#define CHECK_NULL(context, ptr, ...)                                                                    \
    do {                                                                                                 \
        if ((ptr) == nullptr) {                                                                          \
            const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
            OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", ##__VA_ARGS__);                               \
            REPORT_CALL_ERROR("EZ9999", "op[%s], %s is nullptr!", name, ##__VA_ARGS__);                  \
            return ge::GRAPH_FAILED;                                                                     \
        }                                                                                                \
    } while (0)

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

class ExpertDispatchTilingBase : public TilingBaseClass
{
public:
    explicit ExpertDispatchTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~ExpertDispatchTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    ge::graphStatus CheckInputShape();
    void Tiling4GatherOutCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4ExpertTokensCountCompute();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(ExpertVBSComputeTilingData* tilingData);
    void Tinlig4VBSOneCoreCompute(ExpertVBSComputeTilingData* tilingData);

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1024;
    int64_t totalLength_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t cols_ = 0;
    int64_t inputXDtypeSize_;

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;

    int64_t sortMode_ = 0;
    int64_t histMode_ = 0;
    int64_t gatherMode_ = 0;

    const gert::StorageShape* xShapePtr_ = nullptr;
    const gert::StorageShape* expertIdShapePtr_ = nullptr;
    const gert::StorageShape* scaleShapePtr_ = nullptr;

    const gert::ContinuousVector* expertRangeListPtr_;

    const gert::StorageShape* dispatchedXShapePtr_ = nullptr;
    const gert::StorageShape* dispatchedRowIdxShapePtr_ = nullptr;
    const gert::StorageShape* expertTokensCountShapePtr_ = nullptr;
    const gert::StorageShape* expertTotalCountShapePtr_ = nullptr;
    const gert::StorageShape* dispatchedScaleShapePtr_ = nullptr;

    const char* opName = "";
    ExpertDispatchTilingData expertDispatchTilingData;
};

void ExpertDispatchTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus ExpertDispatchTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName, "Fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.blockDim = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    expertDispatchTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpertDispatchTilingBase::CheckAttr()
{
    CHECK_FAIL(context_, expertRangeListPtr_->GetSize() != ATTR_EXPERT_RANGE_DIM,
               "The dim number of expert_range should be %ld.", ATTR_EXPERT_RANGE_DIM);
    const int64_t* expertRangeList = reinterpret_cast<const int64_t*>(expertRangeListPtr_->GetData());
    expertStart_ = expertRangeList[0];
    expertEnd_ = expertRangeList[1];
    expertDispatchTilingData.set_expertStart(expertStart_);
    expertDispatchTilingData.set_expertEnd(expertEnd_);
    expertDispatchTilingData.set_actualExpertNum(expertEnd_ - expertStart_);
    OP_LOGI(context_, "expert_start is: %ld, expert_end is: %ld, actualExpertNum is: %ld", expertStart_, expertEnd_,
            expertEnd_ - expertStart_);

    CHECK_FAIL(context_, expertStart_ < 0, "expert_start should be greater than or equal to 0");
    CHECK_FAIL(context_, expertStart_ >= expertEnd_, "expert_start should be less than expert_end");
    CHECK_FAIL(context_, expertEnd_ > EXPERT_ID_MAX, "expert_end should be less than or equal to %ld", EXPERT_ID_MAX);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpertDispatchTilingBase::CheckInputShape()
{
    // 获取输入shape
    const gert::Shape xShape = xShapePtr_->GetStorageShape();
    OP_LOGI(context_, "input x shape: %s", Ops::Base::ToString(xShape).c_str());
    const gert::Shape expertIdShape = expertIdShapePtr_->GetStorageShape();
    OP_LOGI(context_, "input expert_id shape: %s", Ops::Base::ToString(expertIdShape).c_str());
    const gert::Shape scaleShape = scaleShapePtr_->GetStorageShape();
    OP_LOGI(context_, "scale shape: %s", Ops::Base::ToString(scaleShape).c_str());
    // 学员补充：校验输入Shape大小，满足计算图匹配关系，参考CheckAttr()函数
    OP_CHECK_IF(xShape.GetDimNum() != DIM_TWO, OP_LOGE(context_,
        "The dim number of x should be %ld.", DIM_TWO), return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertIdShape.GetDimNum() != DIM_TWO, OP_LOGE(context_,
        "The dim number of expert_id should be %ld.", DIM_TWO), return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(0) != expertIdShape.GetDim(0), OP_LOGE(context_,
        "Input rows should be same."), return ge::GRAPH_FAILED);
    // 补充结束

    n_ = expertIdShape.GetDim(0);
    k_ = expertIdShape.GetDim(1);
    cols_ = xShape.GetDim(1);
    expertDispatchTilingData.set_n(n_);
    expertDispatchTilingData.set_k(k_);
    expertDispatchTilingData.set_cols(cols_);
    totalLength_ = n_ * k_;
    inputXDtypeSize_ =
        static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(INPUT_X_INDEX)->GetDataType()));
    OP_LOGI(context_, "Input x dtype size is: %ld.", inputXDtypeSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpertDispatchTilingBase::CheckOutShape()
{
    // 获取输出shape
    const gert::Shape dispatchedXShape = dispatchedXShapePtr_->GetStorageShape();
    OP_LOGI(context_, "dispatched_x shape: %s", Ops::Base::ToString(dispatchedXShape).c_str());
    const gert::Shape dispatchedRowIdxShape = dispatchedRowIdxShapePtr_->GetStorageShape();
    OP_LOGI(context_, "dispatched_row_idx shape: %s", Ops::Base::ToString(dispatchedRowIdxShape).c_str());
    const gert::Shape expertTokensCountShape = expertTokensCountShapePtr_->GetStorageShape();
    OP_LOGI(context_, "expert_tokens_count shape: %s", Ops::Base::ToString(expertTokensCountShape).c_str());
    const gert::Shape expertTotalCountShape = expertTotalCountShapePtr_->GetStorageShape();
    OP_LOGI(context_, "expert_total_count shape: %s", Ops::Base::ToString(expertTotalCountShape).c_str());
    const gert::Shape dispatchedScaleShape = dispatchedScaleShapePtr_->GetStorageShape();
    OP_LOGI(context_, "dispatched_scale shape: %s", Ops::Base::ToString(dispatchedScaleShape).c_str());
    // 学员补充：输出Shape校验，满足计算图匹配关系，参考CheckAttr()函数
    OP_CHECK_IF(dispatchedXShape.GetDimNum() != DIM_TWO, OP_LOGE(context_,
        "The dim number of dispatched_x should be %ld.", DIM_TWO), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dispatchedRowIdxShape.GetDimNum() != DIM_ONE, OP_LOGE(context_,
        "The dim number of dispatched_row_idx should be %ld.", DIM_ONE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dispatchedXShape.GetDim(0) != totalLength_, OP_LOGE(context_,
        "The first dim of dispatched_x should be %ld.", totalLength_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dispatchedXShape.GetDim(1) != cols_, OP_LOGE(context_,
        "The second dim of dispatched_x should be %ld.", cols_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dispatchedRowIdxShape.GetDim(0) != totalLength_, OP_LOGE(context_,
        "The first dim of dispatched_row_idx and expanded_expert_id should be %ld.", totalLength_), return ge::GRAPH_FAILED);
    // 补充结束

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpertDispatchTilingBase::GetShapeAttrsInfo()
{
    OP_LOGI(context_, "TilingContext: %s", context_->GetNodeName());

    // 获取输入shape
    xShapePtr_ = context_->GetInputShape(INPUT_X_INDEX);
    CHECK_NULL(context_, xShapePtr_, "x");
    expertIdShapePtr_ = context_->GetInputShape(INPUT_EXPERT_ID_INDEX);
    CHECK_NULL(context_, expertIdShapePtr_, "expert_id");
    scaleShapePtr_ = context_->GetInputShape(INPUT_SCALE_INDEX);
    CHECK_NULL(context_, scaleShapePtr_, "scale");

    // 获取输出shape
    dispatchedXShapePtr_ = context_->GetOutputShape(OUTPUT_DISPATCHED_X_INDEX);
    CHECK_NULL(context_, dispatchedXShapePtr_, "dispatched_x");
    dispatchedRowIdxShapePtr_ = context_->GetOutputShape(OUTPUT_DISPATCHED_ROW_IDX_INDEX);
    CHECK_NULL(context_, dispatchedRowIdxShapePtr_, "dispatched_row_idx");
    expertTokensCountShapePtr_ = context_->GetOutputShape(OUTPUT_EXPERT_TOKENS_COUNT_INDEX);
    CHECK_NULL(context_, expertTokensCountShapePtr_, "expert_tokens_count");
    expertTotalCountShapePtr_ = context_->GetOutputShape(OUTPUT_EXPERT_TOTAL_COUNT_INDEX);
    CHECK_NULL(context_, expertTotalCountShapePtr_, "expert_total_count");
    dispatchedScaleShapePtr_ = context_->GetOutputShape(OUTPUT_DISPATCHED_SCALE_INDEX);
    CHECK_NULL(context_, dispatchedScaleShapePtr_, "dispatched_scale");

    // 获取属性
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    expertRangeListPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_EXPERT_RANGE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, expertRangeListPtr_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpertDispatchTilingBase::DoOpTiling()
{
    auto ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4ExpertTokensCountCompute();
    Tiling4GatherOutCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

void ExpertDispatchTilingBase::ShowTilingData()
{
}

ge::graphStatus ExpertDispatchTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ExpertDispatchTilingBase::GetTilingKey() const
{
    return TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE + histMode_ * HIST_TILINGKEY_BASE +
           gatherMode_ * GATHER_OUT_TILINGKEY_BASE;
}

ge::graphStatus ExpertDispatchTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = totalLength_ * sizeof(float) * NUM_TWO * NUM_THREE;  // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        expertDispatchTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO;  // 多核同步需要的空间
    size_t scatterWorkspaceSize = totalLength_ * sizeof(int32_t);
    size_t expertTokensCountWorkspaceSize = (expertEnd_ - expertStart_) * sizeof(int32_t);
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, sizeof(int32_t));
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + expertTokensCountWorkspaceSize +
                     expertTokenTotalCountWorkspace + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    OP_LOGI(context_, "Allocate workspaceSize is: %ld", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpertDispatchTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    expertDispatchTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(expertDispatchTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ExpertDispatchTilingBase::Tinlig4VBSOneCoreCompute(ExpertVBSComputeTilingData* tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength_);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void ExpertDispatchTilingBase::Tinlig4VBSMultiCoreCompute(ExpertVBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, sortLoopMaxElement);    // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));  // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                             // 不能超过物理核数

    int64_t perCoreElements = totalLength_ / needCoreNum;  // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength_ - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            Ops::Base::CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement));  // 每个核处理的loop数
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                                (tilingData->get_perCoreLoops() - 1) *
                                                    tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(totalLength_ -
                                         (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                         SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                                 (tilingData->get_lastCoreLoops() - 1) *
                                                     tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    OP_TILING_CHECK(tilingData->get_lastCoreLastLoopElements() <= 0,
                    OP_LOGE(opName, "vbs tiling failed"),
                    ;);
}

void ExpertDispatchTilingBase::Tiling4VBSCompute()
{
    if (totalLength_ <= sortLoopMaxElement) {  // 排序只用到一个核排序
        sortMode_ = 0;
    } else {
        sortMode_ = 1;
    }

    auto tilingData = &expertDispatchTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (sortMode_ == 0UL) {  // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void ExpertDispatchTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &expertDispatchTilingData.vbsComputeParamsOp;
    auto tilingData = &expertDispatchTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) {  // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                             // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum);  // 需要的核数
}

void ExpertDispatchTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &expertDispatchTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void ExpertDispatchTilingBase::Tiling4ExpertTokensCountCompute()
{
    auto tilingData = &expertDispatchTilingData.expertTokensCountTilingDataOp;
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    // 学员补充（全载）： Expert Tokens Count Tiling Data 核间切分计算，推荐使用Ops::Base::CeilDiv
    perCoreElements = Ops::Base::CeilDiv(totalLength_, aivNum);
    needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreElements);
    lastCoreElements = totalLength_ - (needCoreNum - 1) * perCoreElements;

    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElements);

    int64_t perCoreLoops = 1;
    int64_t perCorePerLoopElements = perCoreElements;
    int64_t perCoreLastLoopElements = perCoreElements;
    // 学员补充（非全载）：Expert Tokens Count Tiling Data 非全载切分核内切分计算
     int64_t maxElementsPerLoop =
        (static_cast<int64_t>(aicoreParams_.ubSize) -
         Align(expertDispatchTilingData.get_actualExpertNum(), ONE_BLOCK_BYTE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         ONE_BLOCK_BYTE) /
        static_cast<int64_t>(sizeof(int32_t));
    perCoreLoops = Ops::Base::CeilDiv(perCoreElements, maxElementsPerLoop);
    perCorePerLoopElements = Ops::Base::CeilDiv(perCoreElements, perCoreLoops);
    perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    // 补充结束

    tilingData->set_perCoreLoops(perCoreLoops);
    tilingData->set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData->set_perCoreLastLoopElements(perCoreLastLoopElements);

    int64_t lastCoreLoops = 1;
    int64_t lastCorePerLoopElements = lastCoreElements;
    int64_t lastCoreLastLoopElements = lastCoreElements;
    // 学员补充（非全载）： Expert Tokens Count Tiling Data 非全载尾核切分计算
    lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, maxElementsPerLoop);
    lastCorePerLoopElements = Ops::Base::CeilDiv(lastCoreElements, lastCoreLoops);
    lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    // 补充结束

    tilingData->set_lastCoreLoops(lastCoreLoops);
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(lastCoreLastLoopElements);

    histMode_ = perCoreLoops == 1 ? 0 : 1;  // 全载 or 非全载

    OP_LOGI(context_,
            "ExpertTokensCountCompute Tilingdata, needCoreNum is: %ld, perCoreElements is: %ld, lastCoreElements is: "
            "%ld, perCoreLoops is: %ld, perCorePerLoopElements is: %ld, "
            "perCoreLastLoopElements "
            "is: %ld, lastCoreLoops is: %ld, lastCorePerLoopElements is: %ld, lastCoreLastLoopElements is: %ld",
            needCoreNum, perCoreElements, lastCoreElements, perCoreLoops, perCorePerLoopElements,
            perCoreLastLoopElements, lastCoreLoops, lastCorePerLoopElements, lastCoreLastLoopElements);
}

void ExpertDispatchTilingBase::Tiling4GatherOutCompute()
{
    auto tilingData = &expertDispatchTilingData.gatherOutComputeParamsOp;
    int64_t needCoreNum = 0;
    int64_t perCoreIndicesElements = 0;
    int64_t lastCoreIndicesElements = 0;
    // 学员补充（全载）： GatherOut核间切分计算，推荐使用Ops::Base::CeilDiv
    perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivNum);
    if (perCoreIndicesElements <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;
    // 补充结束

    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreIndicesElements(perCoreIndicesElements);
    tilingData->set_lastCoreIndicesElements(lastCoreIndicesElements);

    int64_t colsLoops = 1;
    int64_t perLoopCols = expertDispatchTilingData.get_cols();
    int64_t lastLoopCols = perLoopCols;
    int64_t colMultiple = NUM_TWO * inputXDtypeSize_;
    int64_t rowMultiple = NUM_TWO;
    int64_t perLoopMaxIndicesElements =
        (static_cast<int64_t>(aicoreParams_.ubSize) - Align(perLoopCols, inputXDtypeSize_) * colMultiple -
         ONE_BLOCK_BYTE * NUM_TWO) /
        rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = Ops::Base::CeilDiv(perLoopCols, NUM_TWO);
        perLoopMaxIndicesElements = (static_cast<int64_t>(aicoreParams_.ubSize) -
                                     Align(perLoopCols, inputXDtypeSize_) * colMultiple - ONE_BLOCK_BYTE * NUM_TWO) /
                                    rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        OP_LOGI(context_, "perLoopCols is: %ld, perLoopMaxIndicesElements is: %ld", perLoopCols,
                perLoopMaxIndicesElements);
    }
    colsLoops = Ops::Base::CeilDiv(expertDispatchTilingData.get_cols(), perLoopCols);
    lastLoopCols = expertDispatchTilingData.get_cols() - (colsLoops - 1) * perLoopCols;
    tilingData->set_colsLoops(colsLoops);
    tilingData->set_perLoopCols(perLoopCols);
    tilingData->set_lastLoopCols(lastLoopCols);

    int64_t perCoreIndicesLoops = 1;
    int64_t perCorePerLoopIndicesElements = perCoreIndicesElements;
    int64_t perCoreLastLoopIndicesElements = perCoreIndicesElements;
    perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    perCoreLastLoopIndicesElements = perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    tilingData->set_perCoreIndicesLoops(perCoreIndicesLoops);
    tilingData->set_perCorePerLoopIndicesElements(perCorePerLoopIndicesElements);
    tilingData->set_perCoreLastLoopIndicesElements(perCoreLastLoopIndicesElements);

    int64_t lastCoreIndicesLoops = 1;
    int64_t lastCorePerLoopIndicesElements = lastCoreIndicesElements;
    int64_t lastCoreLastLoopIndicesElements = lastCoreIndicesElements;
    lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    tilingData->set_lastCoreIndicesLoops(lastCoreIndicesLoops);
    tilingData->set_lastCorePerLoopIndicesElements(lastCorePerLoopIndicesElements);
    tilingData->set_lastCoreLastLoopIndicesElements(lastCoreLastLoopIndicesElements);

    gatherMode_ = (colsLoops == 1 && perCoreIndicesLoops == 1) ? 0 : 1;  // 全载 or 非全载

    OP_LOGI(
        context_,
        "GatherOut Tilingdata, needCoreNum is: %ld, perCoreIndicesElements is: %ld, lastCoreIndicesElements is: %ld, "
        "colsLoops is: %ld, perLoopCols is: %ld, lastLoopCols is: %ld, perCoreIndicesLoops is: %ld, "
        "perCorePerLoopIndicesElements is: %ld, perCoreLastLoopIndicesElements is: %ld, lastCoreIndicesLoops is: "
        "%ld, lastCorePerLoopIndicesElements is: "
        "%ld, lastCoreLastLoopIndicesElements is: %ld",
        needCoreNum, perCoreIndicesElements, lastCoreIndicesElements, colsLoops, perLoopCols, lastLoopCols,
        perCoreIndicesLoops, perCorePerLoopIndicesElements, perCoreLastLoopIndicesElements, lastCoreIndicesLoops,
        lastCorePerLoopIndicesElements, lastCoreLastLoopIndicesElements);
}

static ge::graphStatus TilingForExpertDispatch(gert::TilingContext* context)
{
    ExpertDispatchTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForExpertDispatch(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExpertDispatch)
    .Tiling(TilingForExpertDispatch)
    .TilingParse<ExpertDispatchCompileInfo>(TilingPrepareForExpertDispatch);

}  // namespace optiling
