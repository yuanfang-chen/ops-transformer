/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(npu_linalg, m) {
    m.def("qr_householder(Tensor A) -> (Tensor, Tensor)");
    m.def("tsqr(Tensor A, int? block_size=0) -> (Tensor, Tensor)");
    m.def("svd(Tensor A, int num_iterations=10) -> (Tensor, Tensor, Tensor)");
    m.def("svd_lowrank(Tensor A, *, Tensor? M=None, Tensor? omega=None, int q=160, int niter=2) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
