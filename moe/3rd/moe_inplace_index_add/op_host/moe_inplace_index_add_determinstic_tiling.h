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
 * \file moe_inplace_index_add_determinstic_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_DETERMINSTIC_TILING_H_
#define MOE_INPLACE_INDEX_ADD_DETERMINSTIC_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddDeterminsticTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);

/* for determinstic */
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumAfter);
TILING_DATA_FIELD_DEF(int64_t, ubIndexFactor);
TILING_DATA_FIELD_DEF(int64_t, afterAxisFactor);
TILING_DATA_FIELD_DEF(int64_t, ubVarOptiFactor);
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

TILING_DATA_FIELD_DEF(int64_t, ubQuantaIndxFactor);
TILING_DATA_FIELD_DEF(int64_t, ubVarFactor);
TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, eachCoreVarCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreVarCount);
TILING_DATA_FIELD_DEF(int64_t, isWithAlpha);
TILING_DATA_FIELD_DEF(int64_t, isDeterminstic);
TILING_DATA_FIELD_DEF(int64_t, isOpti);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_300000, MoeInplaceIndexAddDeterminsticTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_300001, MoeInplaceIndexAddDeterminsticTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_300002, MoeInplaceIndexAddDeterminsticTilingData)

class MoeInplaceIndexAddDeterminsticTiling : public MoeInplaceIndexAddTiling
{
public:
    explicit MoeInplaceIndexAddDeterminsticTiling(gert::TilingContext* context) : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddDeterminsticTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    int64_t GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes,
            int64_t originalSize, int64_t postAxisSize, ge::DataType idType);
    void DoOpTilingForDeterminsticSplitPre();
    void DoOpTilingForDeterminsticSplitAfter();
    void DoOpTilingForDeterminstic();
    void SetTilingData();
private:
    MoeInplaceIndexAddDeterminsticTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_