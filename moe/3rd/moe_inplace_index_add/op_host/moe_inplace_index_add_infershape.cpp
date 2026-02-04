/**
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

/*!
 * \file elewise.cc
 * \brief
 */
#include "runtime_util.h"

using namespace ge;
namespace ops
{

static graphStatus InferDataType4SameAsInput(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4SameAsInput enter");
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4SameAsInput end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInplaceIndexAdd).InferDataType(InferDataType4SameAsInput);

}  // namespace ops
