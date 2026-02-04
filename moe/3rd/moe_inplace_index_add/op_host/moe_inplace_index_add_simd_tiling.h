/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file moe_inplace_index_add_simd_sort_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_SIMD_TILING_H_
#define MOE_INPLACE_INDEX_ADD_SIMD_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddSimdTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore);
TILING_DATA_FIELD_DEF(int64_t, ubIndexFactor);
TILING_DATA_FIELD_DEF(int64_t, afterAxisFactor);

TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, mainCoreIndicesLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndicesLoop);
TILING_DATA_FIELD_DEF(int64_t, mainCoreTailIndices);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailIndices);

/* pre */
TILING_DATA_FIELD_DEF(int64_t, eachCorePreAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailCorePreAxisCount);
TILING_DATA_FIELD_DEF(int64_t, updateLoopSize);
TILING_DATA_FIELD_DEF(int64_t, updateTailNum);
TILING_DATA_FIELD_DEF(int64_t, indicesLoopSize);
TILING_DATA_FIELD_DEF(int64_t, indiceAxisTailNum);
TILING_DATA_FIELD_DEF(int64_t, isSplitPreAxis);
/* after */
TILING_DATA_FIELD_DEF(int64_t, eachCoreAfterAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAfterAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailUpdateLoopSize);
TILING_DATA_FIELD_DEF(int64_t, tailUpdateAxisNum);
TILING_DATA_FIELD_DEF(int64_t, isSplitAfterAxis);
TILING_DATA_FIELD_DEF(int64_t, isSplitIndicesAxis);
TILING_DATA_FIELD_DEF(int64_t, isWithAlpha);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_400000, MoeInplaceIndexAddSimdTilingData)

class MoeInplaceIndexAddSimdTiling : public MoeInplaceIndexAddTiling
{
public:
    explicit MoeInplaceIndexAddSimdTiling(gert::TilingContext* context) : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddSimdTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void DoOpTilingSplitAfter();
    void DoOpTilingSplitPre();
    void DoOpTilingSplitIndices();
    int64_t GetIndicesAlignBlockSize(int64_t indicesFactor);
    int64_t GetAfterAlignBlockSize(int64_t indicesFactor, int64_t afterFactor);
    void SetTilingData();

    MoeInplaceIndexAddSimdTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_