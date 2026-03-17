# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import numpy as np
import npu_linalg

DEVICE_ID = 6
torch_npu.npu.set_device(int(DEVICE_ID))
torch.npu.config.allow_internal_format = False

shape_list = [
(32, 16),
(512, 16),
(1024, 16),
]

def generate_tensors(shape):
    np.random.seed(0)
    x = torch.tensor(np.random.uniform(-10, 10, shape)).to(torch.float32)
    torch_npu.npu.set_device(int(DEVICE_ID))
    return x.to("npu:%s" % DEVICE_ID)


def batch_test_qr_householder(shape_list):
    torch.npu.synchronize()
    steps = 14  # skip_first + (wait + warmup + active) * repeat
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False, data_simplification=False
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=2, skip_first=10),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./perf_result/qr_hauseholder"),

        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=False,
        experimental_config=experimental_config) as prof:
            for step in range(steps):
                for idx, shapes in enumerate(shape_list):
                    npu_x = generate_tensors(shapes)
                    npu_q, npu_r = torch.ops.npu_linalg.qr_householder(npu_x)
                prof.step()

if __name__ == '__main__':
    batch_test_qr_householder(shape_list)

