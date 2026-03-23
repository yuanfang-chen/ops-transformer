
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF(int32_t, row);
  TILING_DATA_FIELD_DEF(int32_t, col);
  TILING_DATA_FIELD_DEF(int32_t, k);
  TILING_DATA_FIELD_DEF(int32_t, kAlign);
  TILING_DATA_FIELD_DEF(int32_t, blockNum);
  TILING_DATA_FIELD_DEF(int32_t, workspaceSize);
  TILING_DATA_FIELD_DEF(int32_t, hasFinished);

  TILING_DATA_FIELD_DEF(int32_t, oneCoreRow);
  TILING_DATA_FIELD_DEF(int32_t, activateCore);
  TILING_DATA_FIELD_DEF(int32_t, tailRow);
  TILING_DATA_FIELD_DEF(int32_t, FormerTmpMinsize);
  TILING_DATA_FIELD_DEF(int32_t, TailTmpMinsize);

  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, FormerSoftmaxTilingData); // 将SoftMaxTiling结构体参数增加至TilingData结构体
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, TailSoftmaxTilingData); // 将SoftMaxTiling结构体参数增加至TilingData结构体

  TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, FormerTopkTilingData); // 将SoftMaxTiling结构体参数增加至TilingData结构体
  TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, TailTopkTilingData); // 将SoftMaxTiling结构体参数增加至TilingData结构体

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxTilingData)
}
