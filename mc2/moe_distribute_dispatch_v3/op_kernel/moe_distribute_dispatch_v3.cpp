#include "basic_api/kernel_basic_intf.h"

#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_tiling.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_tiling_key.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
#endif

using namespace MoeDistributeDispatchV2Impl;
using namespace Mc2Tiling;
using namespace AscendC;

// 新增必选输入mc2Context，在不同分支入口将mc2Context传入
template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v3(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, HasTp> op;
        // 分支例1
        op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::STATIC_QUANT, false, HasTp> op;
            // 分支例2
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, ScaleMode, HasTp> op;
            // 分支例3
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        }
    }
#endif
}