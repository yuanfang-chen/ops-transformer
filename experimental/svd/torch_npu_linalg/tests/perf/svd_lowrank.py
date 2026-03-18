# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
from time import time
from decimal import Decimal

import torch
import torch_npu
import numpy as np
import npu_linalg


shape_list = [
    (1, 65536, 512),
    (6, 65536, 512),
    (12, 65536, 512),
    (24, 65536, 512),
    (48, 65536, 512),
    (1, 131072, 512),
]

def generate_tensors_cpu(shape):
    np.random.seed(0)
    return torch.tensor(np.random.uniform(-1, 1, shape)).to(torch.float32)


def generate_tensors(shape):
    x = generate_tensors_cpu(shape)
    torch_npu.npu.set_device(int(DEVICE_ID))
    return x.to(f"npu:{DEVICE_ID}")

def profile_svd_lowrank(shape_list):
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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./perf_result/svd_lowrank"),

        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=False,
        experimental_config=experimental_config) as prof:
            for step in range(steps):
                for idx, shapes in enumerate(shape_list):
                    npu_x = generate_tensors(shapes)
                    npu_u, npu_s, npu_v = torch.ops.npu_linalg.svd_lowrank(npu_x, q=168)
                prof.step()

def benchmark_svd_lowrank(model):
    with torch.no_grad():
        print('Shapes\t\tTorchTime[s]\tAscendTime[s]')
        csv = 'Shapes\tTorchTime[s]\tAscendTime[s]\n'

        for idx, shapes in enumerate(model):
            # Timing torch.svd_lowrank on CPU
            torch_times = []
            for i in range(100):
                x = generate_tensors_cpu(shapes)
                s = time()
                u, s_vals, v = torch.svd_lowrank(x, q=168)
                f = time()
                torch_times.append(f - s)

            # Timing npu_linalg.svd_lowrank on NPU
            ascend_times = []
            for i in range(100):
                npu_x = generate_tensors(shapes)
                torch.npu.synchronize()
                s = time()
                npu_u, npu_s, npu_v = torch.ops.npu_linalg.svd_lowrank(npu_x, q=168)
                torch.npu.synchronize()
                f = time()
                ascend_times.append(f - s)

            torch_median = np.median(torch_times[5:])
            ascend_median = np.median(ascend_times[5:])

            print(
                shapes,
                '\t',
                f'{Decimal(torch_median):.2E}',
                '\t',
                f'{Decimal(ascend_median):.2E}'
            )
            csv += (
                f'"{shapes}"\t'
                f'{torch_median:.2e}\t'
                f'{ascend_median:.2e}\n'
            )

    with open('svd_lowrank.csv', 'w') as f:
        f.write(csv)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile", "benchmark"],
        help="Choose which function to run: profile or benchmark",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Enter device id",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DEVICE_ID = args.device_id

    torch_npu.npu.set_device(int(DEVICE_ID))
    torch.npu.config.allow_internal_format = False

    if args.mode == "profile":
        profile_svd_lowrank(shape_list)
    elif args.mode == "benchmark":
        benchmark_svd_lowrank(shape_list)
