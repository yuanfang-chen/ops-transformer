#ifndef GMM_EXPERT_OP_H
#define GMM_EXPERT_OP_H

#include "kernel_operator.h"
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"

using namespace AscendC;

namespace MC2KernelTemplate {

template <typename TilingDataType, typename xType, typename wType, typename biasType,
          typename scaleType, typename yType, 
          CubeFormat wFormat = CubeFormat::ND,
          bool aTrans = false, bool bTrans = false>
class QuantGroupedMatmul {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR yGM, const TilingDataType* tilingData, TPipe* tPipe) {
        // TODO
        // GET_TILING_DATA_MEMBER(Mc2GroupedMatmulTilingData::GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tilingGM_);
        // GET_TILING_DATA_MEMBER(Mc2GroupedMatmulTilingData::GMMQuantTilingData, mmTilingData, mmTilingData_, tilingGM_);
        // GET_TILING_DATA_MEMBER_ADDR(Mc2GroupedMatmulTilingData::GMMQuantTilingData, gmmArray, gmmArrayAddr_, tilingGM_);

        // gmmASWKernel.Init(permuteOutGM_,
        //     gmmwGM_, biasGM_, gmmxScaleGM_, 0, gmmWeightScaleGM_, gmmyGM_, workspaceGM_,
        //     &gmmQuantParams_, &mmTilingData_, gmmArrayAddr_,
        //     tPipe_);
    }
    
    __aicore__ inline void Process(uint32_t expertIdx) {
        gmmASWKernel.Process();
    }
protected:
    __aicore__ inline void UpdateAddr(uint32_t expertIdx) {

    }

private:
    GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmASWKernel_;
};

} // namespace AscendC
#endif