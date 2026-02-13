#ifndef CAUSAL_CONV1D_UPDATE_TILING_H
#define CAUSAL_CONV1D_UPDATE_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "register/tilingdata_base.h"

namespace optiling {

constexpr int32_t X_INDEX = 0;
constexpr int32_t FILTER_INDEX = 1;
constexpr int32_t CONV_STATE_INDICES_INDEX = 2;
constexpr int32_t CONV_STATE_INDEX = 3;
constexpr int32_t ACCEPT_TOKEN_NUM_INDEX = 4;
constexpr int32_t Y_INDEX = 0;

struct CausalConv1dUpdateTilingData : public TilingData {
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t loopNumBS;
    int64_t loopNumDim;
    int64_t ubFactorBS;
    int64_t ubTailFactorBS;
    int64_t ubFactorDim;
    int64_t ubTailFactorDim;
    int64_t tailBlockloopNumBS;
    int64_t tailBlockloopNumDim;
    int64_t tailBlockubFactorBS;
    int64_t tailBlockubTailFactorBS;
    int64_t tailBlockubFactorDim;
    int64_t tailBlockubTailFactorDim;
};

class CausalConv1dUpdateTiling : public TilingBaseClass {
public:
    explicit CausalConv1dUpdateTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~CausalConv1dUpdateTiling() {}

    uint32_t GetTilingSize() override {
        return sizeof(CausalConv1dUpdateTilingData);
    }

    ge::graphStatus Tiling() override;

private:
    bool ValidateShapes();
    bool ValidateDataTypes();
    void CalculateTilingParams(CausalConv1dUpdateTilingData& tilingData);
    void CalculateCoreParams(CausalConv1dUpdateTilingData& tilingData, int64_t validBatch);
    void CalculateLoopParams(CausalConv1dUpdateTilingData& tilingData, bool isTailBlock);

    int64_t batchSize_;
    int64_t seqLen_;
    int64_t dim_;
    int64_t kernelSize_;
    int64_t inValidBatchNum_;
    int64_t pad_slot_id_;
    ge::DataType xDtype_;
    ge::DataType filterDtype_;
    ge::DataType cacheStateDtype_;
    ge::DataType indicesDtype_;
    ge::DataType acceptTokenDtype_;
    size_t xDtypeSize_;
};

} // namespace optiling

#endif // CAUSAL_CONV1D_UPDATE_TILING_H
