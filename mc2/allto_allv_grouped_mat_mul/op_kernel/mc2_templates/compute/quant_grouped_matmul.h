#ifndef GMM_EXPERT_OP_H
#define GMM_EXPERT_OP_H

#include "kernel_operator.h"
#include "../grouped_matmul_apt/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#include "../grouped_matmul_apt/op_kernel/arch35/non_quant/grouped_matmul_basic_kernel.h"

using namespace AscendC;

namespace MC2KernelTemplate {

template <typename TilingDataType, typename xType, typename wType, typename biasType, typename ScaleType, typename yType, CubeFormat wFormat = CubeFormat::ND,
          bool aTrans = false, bool bTrans = false>
class QuantGroupedMatmul {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR WeightScaleGM, GM_ADDR yGM, const TilingDataType* tilingData, TPipe* tPipe) {
    }
    
    __aicore__ inline void Process(uint32_t expertIndex) {
    }

private:
    GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmASWKernel_;

    GM_ADDR xGM_ = nullptr;
    GM_ADDR weightGM_ = nullptr;
    GM_ADDR yGM_ = nullptr;
    GM_ADDR xScaleGM_ = nullptr;
    GM_ADDR WeightScaleGM_ = nullptr;
    GM_ADDR tilingGM_ = nullptr;

    const TilingDataType *tilingData_ = nullptr;

    uint32_t expertNumInOneRank_ = 0U;
    uint32_t rankDim_ = 8U;
    uint64_t axisH1_ = 0;
    uint64_t axisN1_ = 0;

    TPipe* tPipe_;

    GlobalTensor<xType> xGMTensor_;
    GlobalTensor<wType> weightGMeightGMTensor_;
    GlobalTensor<ScaleType> xScaleGMTensor_;
    GlobalTensor<ScaleType> weightScaleGMeightGMTensor_;
    GlobalTensor<yType> yGMTensor_;
};

template <typename TilingDataType, typename xType, typename wType, typename biasType, typename scaleType, typename yType, CubeFormat wFormat = CubeFormat::ND,
          bool aTrans = false, bool bTrans = false>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR WeightScaleGM, GM_ADDR yGM, GM_ADDR tilingGM, const TilingDataType* tilingData, TPipe* tPipe)
{
    tPipe_ = tPipe;

    xGM_ = xGM;
    weightGM_ = weightGM;
    yGM_ = yGM;
    xScaleGM_ = xScaleGM;
    WeightScaleGM_ = WeightScaleGM;
    tilingGM_ = tilingGM;

    tilingData_ = tilingData;
    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    axisH1_ = tilingData_->commonTilingInfo.H1;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    xGMTensor_.SetGlobalBuffer((__gm__ xType*)this->xGM_);
    weightGMeightGMTensor_.SetGlobalBuffer((__gm__ wType*)this->weightGM_);
    xScaleGMTensor_.SetGlobalBuffer((__gm__ xType*)this->xScaleGMTensor_);
    weightScaleGMeightGMTensor_.SetGlobalBuffer((__gm__ wType*)this->weightScaleGMeightGMTensor_);
    yGMTensor_.SetGlobalBuffer((__gm__ yType*)this->yGM_);

    tPipe_->Reset();
}

template <typename TilingDataType, typename xType, typename wType, typename biasType, typename scaleType, typename yType, CubeFormat wFormat = CubeFormat::ND,
          bool aTrans = false, bool bTrans = false>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::Process(uint32_t expertIndex){
    
    GET_NESTED_TILING_DATA_MEMBER_ADDR(TilingDataType, GroupedMatmulTilingData::GMMQuantTilingData, gmmQuantTilingData, gmmArray, gmmArrayAddr_, tilingGM_);
    uint64_t expertOffset[MAX_HANDLE_ID_NUM] = {0UL};
    uint64_t gmmTokenNum[MAX_HANDLE_ID_NUM] = {0UL};

    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        uint64_t curTokenNum = 0;
        for (uint32_t i = 0U; i < rankDim_; e++) {
            curTokenNum += static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]);
        }
        gmmTokenNum[e] = curTokenNum;  // 本专家的tokenNum
    }

    for (uint32_t i = 0U; i < rankDim_; e++) {
        if (i >= 1) {
            expertOffset[i] = expertOffset[i - 1] + gmmTokenNum[i - 1] * axisH1_;
        }
    }

    uint64_t inOffset = expertOffset[expertIndex];
    uint64_t outOffset = expertOffset[expertIndex] / axisH1_ * axisN1_;
    gmmASWKernel_.Init(xGM_, weightGM, nullptr, WeightScaleGM, 0, xScaleGM, yGM, nullptr,
 	            &tilingData_->gmmQuantTilingData.gmmQuantParams, &tilingData_->gmmQuantTilingData.mmTilingData, gmmArrayAddr_, tPipe_);

    gmmASWKernel_.Process(inOffset, outOffset);
}

template <typename TilingDataType, typename xType, typename wType, typename biasType, typename scaleType, typename yType, CubeFormat wFormat = CubeFormat::ND,
          bool aTrans = false, bool bTrans = false>
__aicore__ inline void GmmExpertOp<MMKernel, ExtraDataType, TilingDataType>::End()
{
}

} // namespace AscendC
#endif