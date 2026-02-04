/**
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

/*!
 * \file moe_inplace_index_add_tiling.h
 * \brief
 */
#ifndef MOE_INPLACE_INDEX_ADD_TILING_H
#define MOE_INPLACE_INDEX_ADD_TILING_H
#include <cstdint>

namespace optiling  {
struct MoeInplaceIndexAddTilingData {
  int64_t block_num;
  int64_t indices_num;
  int64_t outer_loop;
  int64_t full_num_per_block;
  int64_t tail_num;
  int64_t axis_and_after_data_num_updates;
  int64_t axis_and_after_data_num_var;
  int64_t update_data_num;
  int64_t axis;
  int64_t updates_ub_size;
  int64_t indices_ub_size;
  int64_t tiling_core_num;
  int64_t var_shape_num;
  int64_t updates_shape_num;
};

struct MoeInplaceIndexAddCompileInfo {
  int64_t core_num;
  int64_t ub_size;
  int64_t var_size;
  int64_t var_data_each_block;
  int64_t indices_size;
  int64_t indices_data_each_block;
  int64_t soc_version;
  int64_t atomic_add;
  bool isAscendc{false};
  int64_t coreNum{0};
  int64_t ubSize{0};
};
}  // namespace optiling
#endif  // MOE_INPLACE_INDEX_ADD_TILING_H