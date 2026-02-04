/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_inplace_index_add_simt_sort_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_SIMT_SORT_TILING_H_
#define MOE_INPLACE_INDEX_ADD_SIMT_SORT_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddSimtSortTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, indicesUbFactor);          // 整循环搬运indices数量
TILING_DATA_FIELD_DEF(int64_t, indicesLoopSize);          // 整核搬运indices循环次数
TILING_DATA_FIELD_DEF(int64_t, indiceAxisTailNum);        // 尾核尾循环搬运indices数量
TILING_DATA_FIELD_DEF(int64_t, tailBlockIndicesLoopSize); // 尾核搬运indices循环次数
TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);       // 每个整核处理indices数量
TILING_DATA_FIELD_DEF(int64_t, sortShareBufSize);
TILING_DATA_FIELD_DEF(int64_t, normalUpdatesPreNum);      // 整循环搬运update的pre份数
TILING_DATA_FIELD_DEF(int64_t, tailUpdatesPreNum);        // 尾循环搬运update的pre份数
TILING_DATA_FIELD_DEF(int64_t, updatesPreLoop);           // 搬1次indices时，要搬多份updates的循环次数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500000, MoeInplaceIndexAddSimtSortTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500001, MoeInplaceIndexAddSimtSortTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500010, MoeInplaceIndexAddSimtSortTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500011, MoeInplaceIndexAddSimtSortTilingData)

class MoeInplaceIndexAddSimtSortTiling : public MoeInplaceIndexAddTiling
{
public:
    explicit MoeInplaceIndexAddSimtSortTiling(gert::TilingContext* context) : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddSimtSortTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    uint64_t computeIndicesUbFactor();
    void TilingForSimtSort();
    void SetTilingData();
    MoeInplaceIndexAddSimtSortTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_