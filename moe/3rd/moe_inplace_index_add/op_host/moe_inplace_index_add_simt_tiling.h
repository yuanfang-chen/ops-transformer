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
 * \file moe_inplace_index_add_simt_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_SIMT_TILING_H_
#define MOE_INPLACE_INDEX_ADD_SIMT_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddSimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, colUbFactor);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
TILING_DATA_FIELD_DEF(int64_t, indicesUbFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100000, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100001, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100002, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100003, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100004, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100006, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100027, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100009, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100012, MoeInplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100100, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100101, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100102, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100103, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100104, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100106, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100127, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100109, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100112, MoeInplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101000, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101001, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101002, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101003, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101004, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101006, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101027, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101009, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101012, MoeInplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101100, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101101, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101102, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101103, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101104, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101106, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101127, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101109, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101112, MoeInplaceIndexAddSimtTilingData)

class MoeInplaceIndexAddSimtTiling : public MoeInplaceIndexAddTiling
{
public:
    explicit MoeInplaceIndexAddSimtTiling(gert::TilingContext* context) : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddSimtTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();
    MoeInplaceIndexAddSimtTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_