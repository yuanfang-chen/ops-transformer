# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import numpy as np
from ml_dtypes import float8_e5m2, float8_e8m0fnu, bfloat16
from typing import Union
# 不同输入类型具备不同的范围
DTYPE_RANGE = {
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.float16: (-65504, 65504),
    np.float32: (-1e38, 1e38),
    bfloat16: (-1e38, 1e38), 
    float8_e5m2: (1e-9, 1e6),  
    float8_e8m0fnu: (2**-126, 2**127),  
}
# 生成随机输入
def gen_random_data(size: int, dtype: Union[np.dtype, type], drange: str):
    # 获取随机输入的上界与下界
    try:
        low, high = map(float, drange.split())
        if low >= high:
            raise ValueError(f"range is not vaild! low={low} >= high={high}")
    except Exception as e:
        raise ValueError(f"load drange fail {e}") from e
    # 获取输入Dtype，校验类型是否在支持list里面
    if dtype not in DTYPE_RANGE:
        clip_low, clip_high = low, high
        print(f"cannt get {dtype} range , use {drange}")
    else:
        # 获取Dtype对应得上下界
        dtype_low, dtype_high = DTYPE_RANGE[dtype]
        clip_low = max(low, dtype_low)
        clip_high = min(high, dtype_high)
        if clip_low >= clip_high:
            raise ValueError(f"range of {dtype} {DTYPE_RANGE[dtype]} has no same to {drange}")
    # 生成随机数
    random_data = np.random.uniform(low=clip_low, high=clip_high, size=size).astype(np.float32)
    # 特殊类型处理
    if dtype == float8_e8m0fnu:
        log2_data = np.log2(np.abs(random_data) + 1e-10)
        round_log2 = np.round(log2_data)
        random_data = np.power(2, round_log2) * np.sign(random_data)
    elif "float8" in str(dtype):
        random_data = np.clip(random_data, dtype_low, dtype_high)
    # 生成随机输入
    target_data = random_data.astype(dtype)
    # 校验是否有NAN值
    nan_count = np.isnan(target_data.astype(np.float32)).sum()
    if nan_count > 0:
        raise RuntimeError(f"data contain {nan_count} NaN, please check!")
    return target_data
# 生成输入bin文件
def input_generate(data_name, data_len, rank_size, data_type, file_dir, data_range, seed):
    # 设置随机种子
    np.random.seed(seed)
    # 声明输入大小
    input_gm = np.zeros((rank_size, data_len), dtype=data_type)
    # 生成输入bin文件
    for i in range(rank_size):
        input_gm[i][:] = gen_random_data((data_len), dtype=data_type, drange=data_range)
    for i in range(rank_size):
        input_gm[i].tofile(f"./{file_dir}/input_{data_name}_{i}.bin")
    print(f"{data_name} golden generate success !")
    return input_gm
# CPU反量化实现
def cpu_dequant(x, scale, group_size):
    repeated_scale = np.repeat(scale, group_size, axis=-1)
    return x * repeated_scale
# CPU all_reduce实现
def cpu_all_reduce(output, op):
    if (op == "sum"):
        sum_output = np.sum(output, axis=0, keepdims=False)
    return sum_output
# 生成CPU output golden值
def output_generate(x, scales, bs, hidden_size, rank_size, mxfp, data_type, op, file_dir, data_name):
    # 声明输出形状
    output = np.zeros((rank_size, bs, hidden_size), dtype=data_type)
    # 根据ranksize大小生成多份scale
    for i in range(rank_size):
        x_rank = x[i].reshape(bs, hidden_size)
        # 不同量化方式具备不同的shape
        if mxfp == 0:
            scale_rank = scales[i].reshape(bs, hidden_size//128)
        else:
            scale_rank = scales[i].reshape(bs, hidden_size//64, 2)
            scale_rank = scale_rank.reshape(bs, -1)
        # 输出cpu计算
        output[i] = cpu_dequant(x_rank, scale_rank, 32 if mxfp else 128)
    # cpu all reduce计算
    output_allreduce = cpu_all_reduce(output, op)
    output_allreduce.flatten()
    # 保存cpu golden值
    output_allreduce.tofile(f"./{file_dir}/{data_name}.bin")
    return output_allreduce

def gen_golden_data():
    # 外部输入
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('case_name', type=str)
    parser.add_argument('bs', type=int)
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('input_tensor_range', type=str)
    parser.add_argument('input_tensor_type', type=str)
    parser.add_argument('scales_range', type=str)
    parser.add_argument('scales_type', type=str)
    parser.add_argument('output_type', type=str)
    parser.add_argument('ranksize', type=int)
    parser.add_argument('reduce_op', type=str)
    parser.add_argument('mxfp', type=int)
    parser.add_argument('seed', type=int)
    args = parser.parse_args()
    # 数据类型MAP
    type_map = {
        "int": np.int32,
        "int32_t": np.int32,
        "float16_t": np.float16,
        "float32_t": np.float32,
        "int8_t": np.int8,
        "fp8_e5m2_t": float8_e5m2,
        "fp8_e8m0_t": float8_e8m0fnu,
        "bfloat16_t": bfloat16
    }
    # 数据生成存放路径
    golden_dir = f"./golden/quantallreduce_{args.case_name}_{args.bs}_{args.hidden_size}"
    cmd = f"rm -rf {golden_dir}/* && mkdir -p {golden_dir}"
    os.system(cmd)
    # 读取分析输入
    input_tensor_type = type_map.get(args.input_tensor_type, 'float16_t')
    scales_type = type_map.get(args.scales_type, 'float16_t')
    output_type = type_map.get(args.output_type, 'float16_t')
    inputlen = args.bs * args.hidden_size
    if (args.mxfp == 0):
        scalelen = args.bs * (args.hidden_size // 128)
    else:
        scalelen = args.bs * (args.hidden_size // 64) * 2
    # 生成x输入
    input_x = input_generate("x", inputlen, args.ranksize, input_tensor_type, 
                        golden_dir, args.input_tensor_range, args.seed)
    # 生成scales输入
    input_scales = input_generate("scale", scalelen, args.ranksize, scales_type, 
                                golden_dir, args.scales_range, args.seed)
    print(f'input_x.shape {input_x.shape}')
    print(f'input_scales.shape {input_scales.shape}')
    # 生成cpu输出
    output_cpu = output_generate(input_x, input_scales, args.bs, args.hidden_size, 
        args.ranksize, args.mxfp, output_type, args.reduce_op, golden_dir, "output_cpu")
    print(f'output_cpu.shape {output_cpu.shape}')

if __name__ == '__main__':
    gen_golden_data()