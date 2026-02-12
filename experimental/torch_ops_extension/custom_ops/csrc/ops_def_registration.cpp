/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom, m) {
    m.def(R"(npu_incre_flash_attention( Tensor query, 
                                        Tensor key, 
                                        Tensor value, 
                                        *, 
                                        Tensor? padding_mask=None, 
                                        Tensor? atten_mask=None, 
                                        Tensor? pse_shift=None, 
                                        SymInt[]? actual_seq_lengths=None, 
                                        Tensor? antiquant_scale=None, 
                                        Tensor? antiquant_offset=None, 
                                        Tensor? block_table=None, 
                                        Tensor? dequant_scale1=None, 
                                        Tensor? quant_scale1=None, 
                                        Tensor? dequant_scale2=None, 
                                        Tensor? quant_scale2=None, 
                                        Tensor? quant_offset2=None, 
                                        Tensor? kv_padding_size=None, 
                                        int num_heads=1, 
                                        float scale_value=1.0, 
                                        str input_layout="BSH", 
                                        int num_key_value_heads=0, 
                                        int block_size=0, 
                                        int inner_precise=1) -> Tensor)");                                   
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
