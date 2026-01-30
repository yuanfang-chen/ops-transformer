#ifndef MC2_QUANT_GROUPED_MATMUL_H
#define MC2_QUANT_GROUPED_MATMUL_H

#include "../../../3rd/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans>
class QuantGroupedMatmul {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
                                GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                const GmmTilingDataType *gmmTilingData, TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe)
    {
        xGM_ = xGM;
        wGM_ = weightGM;
        xScaleGM_ = xScaleGM;
        weightScaleGM_ = weightScaleGM;
        yGM_ = yGM;
        tilingData_ = tilingData;
        tPipe_ = tPipe;
        workspaceGM_ = workspaceGM;
        groupListGm_ = workspaceGM_;
        gmmTilingData_ = gmmTilingData;
        gmmArrayAddrIn_ = gmmArrayAddrIn;

        xGlobalBuffer_.SetGlobalBuffer((__gm__ xType *)this->xGM_);
        wGlobalBuffer_.SetGlobalBuffer((__gm__ wType *)this->wGM_);
        yGlobalBuffer_.SetGlobalBuffer((__gm__ yType *)this->yGM_);
        groupListGlobalBuffer_.SetGlobalBuffer((__gm__ int64_t *)groupListGm_);

        expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
        epWorldSize_ = tilingData_->commonTilingInfo.epWorldSize;
        H1_ = tilingData_->commonTilingInfo.H1;
        N1_ = tilingData_->commonTilingInfo.N1;
        const auto *recvCnt = &tilingData_->aicpuTiling.recvCnt[0];
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            for (uint32_t i = 0U; i < epWorldSize_; i++) {
                expertTokenNum_[e] += static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]);
            }
        }
    }

    __aicore__ inline void Process(uint32_t expertIdx)
    {
        UpdateAddr(expertIdx);
        groupListGlobalBuffer_.SetValue(0, expertTokenNum_[expertIdx]);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(groupListGlobalBuffer_);
        gmmASWKernel.Init(xGM_, wGM_, nullptr, xScaleGM_, groupListGm_, weightScaleGM_, yGM_, workspaceGM_,
                          &gmmTilingData_->gmmQuantParams, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_);
        gmmASWKernel.Process();
    }

    __aicore__ inline void End()
    {
    }

protected:
    __aicore__ inline void UpdateAddr(uint32_t expertIdx)
    {
        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr(expertTokenNum_[expertIdx] * H1_);
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(expertIdx * H1_);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr(expertTokenNum_[expertIdx] * N1_);
    }

private:
    using biasType = float;

    GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmASWKernel;
    GM_ADDR xGM_;
    GM_ADDR wGM_;
    GM_ADDR xScaleGM_;
    GM_ADDR weightScaleGM_;
    GM_ADDR yGM_;
    GM_ADDR groupListGm_;
    GM_ADDR workspaceGM_;
    GlobalTensor<xType> xGlobalBuffer_;
    GlobalTensor<wType> wGlobalBuffer_;
    GlobalTensor<yType> yGlobalBuffer_;
    GlobalTensor<int64_t> groupListGlobalBuffer_;
    const TilingDataType *tilingData_;
    TPipe *tPipe_;
    uint64_t expertTokenNum_[32] = {0};
    uint64_t expertNumInOneRank_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t H1_;
    uint64_t N1_;
    const GmmTilingDataType *gmmTilingData_;
    TILING_TYPE *gmmArrayAddrIn_;
};
} // namespace MC2KernelTemplate
#endif
// MC2_QUANT_GROUPED_MATMUL_H