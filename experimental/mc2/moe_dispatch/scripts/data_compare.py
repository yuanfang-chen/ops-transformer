# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import os
from typing import Tuple, Callable
from ml_dtypes import float8_e5m2, float8_e8m0fnu, bfloat16
import csv
from datetime import datetime, timezone, timedelta

TYPE_CONFIG = {
    "int32_t": {"dtype": np.int32, "mae": 1e-3, "cos": 0.9999},
    "float16_t": {"dtype": np.float16, "mae": 1e-3, "cos": 0.9999},
    "float32_t": {"dtype": np.float32, "mae": 1e-3, "cos": 0.9999},
    "int8_t": {"dtype": np.int8, "mae": 1e-3, "cos": 0.9999},
    "bfloat16_t": {"dtype": bfloat16, "mae": 5e-3, "cos": 0.999},
    "fp8_e5m2_t": {"dtype": float8_e5m2, "mae": 1e-2, "cos": 0.99},
    "fp8_e8m0_t": {"dtype": float8_e8m0fnu, "mae": 8e-3, "cos": 0.995},
}

def write_to_result_csv(case_name, mae, max_ae, mse, rmse, cos_sim, is_pass, csv_path="./result.csv"):
    utc_time = datetime.now(tz=timezone.utc)
    timestamp = utc_time + timedelta(hours=8)
    header = ["case_name", "mae", "max_ae", "mse", "rmse", "cos_sim", "is_pass", "timestamp"]
    data_row = [
        case_name,
        round(mae, 6),      
        round(max_ae, 6),
        round(mse, 6),
        round(rmse, 6),
        round(cos_sim, 6),
        is_pass,
        timestamp
    ]
    try:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data_row)
        
        print(f"data write success {csv_path}")
        return True    
    except Exception as e:
        print(f"data write fail : {str(e)}")
        return False

def calculate_metrics(cpu_data: np.ndarray, npu_data: np.ndarray, dtype: np.dtype):
    cpu_data_32 = cpu_data.astype(np.float32)
    npu_data_32 = npu_data.astype(np.float32)

    abs_error = np.abs(cpu_data_32 - npu_data_32)
    mae = float(np.mean(abs_error))
    max_ae = float(np.max(abs_error))
    mse = float(np.mean(np.square(cpu_data_32 - npu_data_32)))
    rmse = float(np.sqrt(mse)) if mse > 0 else 0.0
    cpu_norm = float(np.linalg.norm(cpu_data_32))
    npu_norm = float(np.linalg.norm(npu_data_32))
    if cpu_norm < 1e-10 or npu_norm < 1e-10:
        cos_sim = 1.0 if np.allclose(cpu_data_32, npu_data_32, atol=1e-3) else 0.0
    else:
        dot_product = float(np.dot(cpu_data_32.flatten(), npu_data_32.flatten()))
        cos_sim = dot_product / (cpu_norm * npu_norm)
        cos_sim = np.clip(cos_sim, -1.0, 1.0) 

    return mae, max_ae, mse, rmse, cos_sim

def compare(args, golden_dir: str):
    print("------------------开始进行精度对比---------------")
    if args.output_type not in TYPE_CONFIG:
        print(f"[ERROR] not support : {args.output_type}, support : {list(TYPE_CONFIG.keys())}")
        return False
    type_info = TYPE_CONFIG[args.output_type]
    target_dtype = type_info["dtype"]
    cpu_file = f"{golden_dir}/output_cpu.bin"
    if not os.path.exists(cpu_file):
        print(f"[ERROR] CPU golden data is not find : {cpu_file}")
        return False
    try:
        cpu_data = np.fromfile(cpu_file, dtype=target_dtype)
        print(f"[INFO] golden data length : {len(cpu_data)}, type : {args.output_type}")
    except Exception as e:
        print(f"[ERROR] load golden data fail {e}")
        return False
    all_pass = True
    for rank in range(args.ranksize):
        print(f"\n========== 对比 Rank-{rank} NPU 数据 ==========")
        npu_file = f"{golden_dir}/output_npu_{rank}.bin"
        
        if not os.path.exists(npu_file):
            print(f"[ERROR] Rank-{rank} file is not find : {npu_file}")
            all_pass = False
            continue        
        try:
            npu_data = np.fromfile(npu_file, dtype=target_dtype)
            print(f"[INFO] Rank-{rank} NPU data length : {len(npu_data)}")
        except Exception as e:
            print(f"[ERROR] Load Rank-{rank} NPU data failed : {e}")
            all_pass = False
            continue        
        if len(cpu_data) != len(npu_data):
            print(f"[ERROR] Rank-{rank} length is not equal to CPU={len(cpu_data)}, NPU={len(npu_data)}")
            all_pass = False
            continue        
        mae, max_ae, mse, rmse, cos_sim = calculate_metrics(cpu_data, npu_data, target_dtype)    
        print(f"[Rank-{rank} 精度指标]")
        print(f"  MAE：{mae:.6f} (阈值：{type_info['mae']})")
        print(f"  MaxAE：{max_ae:.6f}")
        print(f"  MSE：{mse:.6f}")
        print(f"  RMSE：{rmse:.6f}")
        print(f"  余弦相似度：{cos_sim:.6f} (阈值：{type_info['cos']})")  
        rank_pass = (mae < type_info["mae"]) and (cos_sim > type_info["cos"])
        if rank_pass:
            print(f"[PASS] Rank-{rank} 精度对比通过！")
        else:
            print(f"[FAIL] Rank-{rank} 精度对比失败！")
            all_pass = False
    print("\n------------------精度对比结果---------------")
    if all_pass:
        print(f"[PASS] 所有Rank（类型：{args.output_type}）精度对比通过！")
    else:
        print(f"[FAIL] 部分Rank（类型：{args.output_type}）精度对比失败！")
    write_to_result_csv(args.case_name, mae, max_ae, mse, rmse, cos_sim, all_pass)
    return all_pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('case_name', type=str)
    parser.add_argument("bs", type=int)
    parser.add_argument("hidden_size", type=int)
    parser.add_argument("output_type", type=str)
    parser.add_argument("ranksize", type=int)
    args = parser.parse_args()

    golden_dir = f"./golden/quantallreduce_{args.case_name}_{args.bs}_{args.hidden_size}"
    
    compare(args, golden_dir)