#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from functools import partial
from compressor_golden import cpu_compressor, get_seq_used_by_batch
import pandas as pd
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
import argparse
import custom_ops

def load_excel_test_cases(excel_file_path: str, sheetname: str):

    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[tuple]: 测试用例元组列表，每个元组包含 20+ 个字段。
                       若失败或跳过，则返回空列表。
    """
    # 优先使用传入的 sheetname，否则尝试从环境变量获取
    if sheetname is None:
        sheetname = 'Sheet1'

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            'Testcase_Name', 'batch_size', 'hidden_size', 'Seq_len', 'head_dim', 'block_size', 'rope_head_dim', 'cmp_ratio',
            'coff', 'norm_eps', 'start_p', 'rotary_mode', 'layout_x', 'data_type', 'cu_seqlens', 'seqused', 'start_pos',
            'x_datarange','wkv_datarange','wgate_datarange','ape_datarange','norm_weight_datarange','kv_state_datarange','score_state_datarange'
        ]

        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append((
                row['Testcase_Name'],
                row['batch_size'], 
                row['hidden_size'], 
                row['Seq_len'], 
                row['head_dim'], 
                row['block_size'], 
                row['rope_head_dim'], 
                row['cmp_ratio'], 
                row['coff'], 
                row['norm_eps'], 
                row['start_p'], 
                row['rotary_mode'], 
                row['layout_x'], 
                row['data_type'], 
                row['cu_seqlens'], 
                row['seqused'], 
                row['start_pos'], 
                row['x_datarange'], 
                row['wkv_datarange'], 
                row['wgate_datarange'], 
                row['ape_datarange'], 
                row['norm_weight_datarange'], 
                row['kv_state_datarange'], 
                row['score_state_datarange']
            ))

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None

class Generalized_operator():
    def forward(self,
                x,
                wkv,
                wgate,
                kv_state,
                score_state,
                ape,
                norm_weight, 
                rope_sin,
                rope_cos,
                block_table,
                cu_seqlens,
                seqused,
                start_pos,
                rope_head_dim,
                cmp_ratio,
                coff,
                norm_eps,
                rotary_mode):
        return cpu_compressor(
            x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos,
            block_table=block_table, cu_seqlens=cu_seqlens, seqused=seqused, start_pos=start_pos,
            rope_head_dim=rope_head_dim, cmp_ratio=cmp_ratio, coff=coff, norm_eps=norm_eps, rotary_mode=rotary_mode)


def compressor_output_single(data_case):
    casename = data_case[0]
    params = data_case[1:]

    batch_size, hidden_size, Seq_len, head_dim, block_size, rope_head_dim, cmp_ratio, coff, norm_eps, \
    start_p, rotary_mode, layout_x, data_type, cu_seqlens, seqused, start_pos, x_datarange, wkv_datarange,  \
    wgate_datarange, ape_datarange, norm_weight_datarange, kv_state_datarange, score_state_datarange = params

    if data_type == 'FP16':
        data_type =  torch.float16
    elif data_type == 'BF16':
        data_type = torch.bfloat16

    # 处理B+1
    # print(f"===== {cu_seqlens} =====")
    if isinstance(cu_seqlens, int):
        cu_seqlens = [cu_seqlens]
    elif isinstance(cu_seqlens, list):
        cu_seqlens = cu_seqlens
    elif cu_seqlens != None:
        cu_seqlens = [int(x.strip()) for x in cu_seqlens.split(',')]
    # print(f"===== {start_pos} =====")
    if isinstance(start_pos, int):
        start_pos = [start_pos]
    elif isinstance(start_pos, list):
        start_pos = start_pos
    elif start_pos != None and batch_size !=1:
        start_pos = [int(x.strip()) for x in start_pos.split(',')]
    elif start_pos != None:
        start_pos = [int(start_pos)]
    # print(f"===== {seqused} =====")
    if isinstance(seqused, int):
        seqused = [seqused]
    elif isinstance(seqused, list):
        seqused = seqused
    elif seqused != None and batch_size !=1:
        seqused = [int(x.strip()) for x in seqused.split(',')]
    elif seqused != None:
        seqused = [int(seqused)]

    x_datarange = [float(x.strip()) for x in x_datarange.split(',')]
    wkv_datarange = [float(x.strip()) for x in wkv_datarange.split(',')]
    wgate_datarange = [float(x.strip()) for x in wgate_datarange.split(',')]
    ape_datarange = [float(x.strip()) for x in ape_datarange.split(',')]
    norm_weight_datarange = [float(x.strip()) for x in norm_weight_datarange.split(',')]
    kv_state_datarange = [float(x.strip()) for x in kv_state_datarange.split(',')]
    score_state_datarange = [float(x.strip()) for x in score_state_datarange.split(',')]

    if Seq_len == 0 or  batch_size == 0:
        if Seq_len == 0:
            if layout_x == "TH":
                T = batch_size * Seq_len
                cu_seqlens = torch.zeros((batch_size+1), dtype=torch.int32)
            else:
                cu_seqlens = None

            seqused = torch.zeros((batch_size), dtype=torch.int32)
            if start_pos is not None:
                start_pos = torch.tensor(start_pos).to(torch.int32)
            else:
                start_pos = [0] * batch_size
            S_max = max(start_pos) + Seq_len
            block_table = torch.zeros(size=(batch_size, 0), dtype=torch.int32)
            block_num = 0
            for i in range(batch_size):
                block_num += math.ceil((int(start_pos[i])) / block_size)
            shuffled_indices = torch.randperm(batch_size)
            index = torch.arange(0, batch_size, 1, dtype=torch.int32)
            index = index[shuffled_indices]
            for i in range(batch_size):
                block_table[i] = index[i]

        if batch_size == 0:
            if layout_x == "TH":
                if cu_seqlens is not None:
                    cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
                else:
                    T = batch_size * Seq_len
                    cu_seqlens = torch.zeros((batch_size+1), dtype=torch.int32)
            else:
                cu_seqlens = None
            start_pos = None
            seqused = None
            S_max = Seq_len
            max_block_num_per_batch = (S_max + cmp_ratio + block_size - 1) // block_size
            block_table = torch.zeros(size=(batch_size, max_block_num_per_batch), dtype=torch.int32)
            block_num = 0
            shuffled_indices = torch.randperm(max_block_num_per_batch)
            index = torch.arange(0, max_block_num_per_batch, 1, dtype=torch.int32)
            index = index[shuffled_indices]
            for i in range(batch_size):
                block_table[i] = index[i]

        kv_state = torch.tensor(np.random.uniform(kv_state_datarange[0], kv_state_datarange[1], (block_num, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(score_state_datarange[0], score_state_datarange[1], (block_num, block_size, coff * head_dim))).to(torch.float32)
        
        # other input
        if layout_x == "TH":
            x_shape = (0, hidden_size)
            rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + batch_size), rope_head_dim)
            rope_cos_shape = rope_sin_shape
        else:
            x_shape = (batch_size, Seq_len, hidden_size)
            rope_sin_shape = (batch_size, (Seq_len + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
            rope_cos_shape = rope_sin_shape

        x = torch.tensor(np.random.uniform(x_datarange[0], x_datarange[1], x_shape)).to(data_type)
        wkv = torch.tensor(np.random.uniform(wkv_datarange[0], wkv_datarange[1], (coff * head_dim, hidden_size))).to(data_type)
        wgate = torch.tensor(np.random.uniform(wgate_datarange[0], wgate_datarange[1], (coff * head_dim, hidden_size))).to(data_type)
        ape = torch.tensor(np.random.uniform(ape_datarange[0], ape_datarange[1], (cmp_ratio, coff * head_dim))).to(torch.float32)
        norm_weight = torch.tensor(np.random.uniform(norm_weight_datarange[0], norm_weight_datarange[1], (head_dim))).to(data_type)
        rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type)
        rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type)
        ### ======================== gen input data finish =============================
        ### ======================== execute cpu start =================================
        cpu_kv_state = kv_state.clone()
        cpu_score_state = score_state.clone()

        test_operator = Generalized_operator()
        cpu_kv_state = kv_state.clone()
        cpu_score_state = score_state.clone()

        test_operator = Generalized_operator()
        cpu_result, kv_mask_result = test_operator.forward( x,
                                            wkv,
                                            wgate,
                                            cpu_kv_state,
                                            cpu_score_state,
                                            ape,
                                            norm_weight, 
                                            rope_sin,
                                            rope_cos,
                                            block_table = block_table,
                                            cu_seqlens = cu_seqlens,
                                            seqused = seqused,
                                            start_pos = start_pos,
                                            rope_head_dim = rope_head_dim,
                                            cmp_ratio = cmp_ratio,
                                            coff = coff,
                                            norm_eps = norm_eps,
                                            rotary_mode = rotary_mode)
        update_kv = cpu_kv_state != kv_state
        update_score = cpu_score_state != score_state
        output_tensors = {
            "params":params,
            "cpu_result": cpu_result,
            "kv_mask_result": kv_mask_result,
            "update_kv":update_kv,
            "update_score":update_score,
            "cpu_kv_state":cpu_kv_state,
            "cpu_score_state":cpu_score_state,
            "x":x,
            "wkv":wkv,
            "wgate":wgate,
            "kv_state":kv_state,
            "score_state":score_state,
            "ape":ape,
            "norm_weight":norm_weight, 
            "rope_sin":rope_sin,
            "rope_cos":rope_cos,
            "block_table":block_table,
            "cu_seqlens":cu_seqlens,
            "seqused":seqused,
            "start_pos":start_pos,
            "rope_head_dim":rope_head_dim,
            "cmp_ratio":cmp_ratio,
            "coff":coff,
            "norm_eps":norm_eps,
            "rotary_mode":rotary_mode
        }
        return  casename, output_tensors

    S_max = 0
    if layout_x == "TH":
        if cu_seqlens is not None:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        else:
            T = batch_size * Seq_len
            cu_seqlens = torch.arange(0, T + 1, Seq_len, dtype=torch.int32)
        
        if seqused is not None:
            S_max = max(seqused)
            seqused = torch.tensor(seqused).to(torch.int32)
        else:
            for i in range(1, batch_size + 1):
                if S_max < cu_seqlens[i] - cu_seqlens[i - 1]:
                    S_max = cu_seqlens[i] - cu_seqlens[i - 1] 

        if start_pos is not None:
            start_pos = torch.tensor(start_pos).to(torch.int32)
        else:
            start_pos = [0] * batch_size
        S_max = max(start_pos) + Seq_len
    else:
        cu_seqlens = None
        if start_pos == None:
            start_pos = [0] * batch_size
        else:
            start_pos = torch.tensor(start_pos).to(torch.int32)
        S_max = max(start_pos) + Seq_len

        if seqused is not None:
            seqused = torch.tensor(seqused).to(torch.int32)

    ### ======================== check input params start ========================
    print(f"params = {params}")

    actseqs = []
    if seqused is not None:
        actseqs = seqused
    else:
        if cu_seqlens is not None:
            for i in range(len(cu_seqlens) - 1):
                diff = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
                actseqs.append(diff)
        else:
            actseqs = [Seq_len] * batch_size

    max_block_num_per_batch = (S_max + cmp_ratio + block_size - 1) // block_size
    block_num = 0
    for i in range(batch_size):
        block_num += math.ceil((int(actseqs[i]) + int(start_pos[i])) / block_size)
    block_num = block_num - 1
    shuffled_indices = torch.randperm(block_num)
    index = torch.arange(1, block_num + 1, 1, dtype=torch.int32)
    index = index[shuffled_indices]
    index_id = 0
    block_table = torch.zeros(size=(batch_size, max_block_num_per_batch), dtype=torch.int32)
    for i in range(batch_size):
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0: #如果start_pos[i]正好被整除，这个batch的本次需要处理的第一个r的数据都通过x传入了，不需要从state中取，不需要预留空间
            cur_end = start_pos[i]
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            if index_id < block_num:
                block_table[i][j] = index[index_id]
                index_id += 1
        end_pos = get_seq_used_by_batch(i, Seq_len, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio 
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos       #如果start_pos[i] + end_pos正好被整除，下一个r的数据一个都没有给，所以不需要给下一个r预留空间
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            if block_table[i][j] == 0 and index_id < block_num:
                block_table[i][j] = index[index_id]
                index_id += 1
    kv_state = torch.tensor(np.random.uniform(kv_state_datarange[0], kv_state_datarange[1], (block_num + 1, block_size, coff * head_dim))).to(torch.float32)
    score_state = torch.tensor(np.random.uniform(score_state_datarange[0], score_state_datarange[1], (block_num + 1, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if layout_x == "TH":
        x_shape = (cu_seqlens[-1], hidden_size)
        rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + batch_size), rope_head_dim)
        rope_cos_shape = rope_sin_shape
    else:
        x_shape = (batch_size, Seq_len, hidden_size)
        rope_sin_shape = (batch_size, (Seq_len + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
        rope_cos_shape = rope_sin_shape

    x = torch.tensor(np.random.uniform(x_datarange[0], x_datarange[1], x_shape)).to(data_type)
    wkv = torch.tensor(np.random.uniform(wkv_datarange[0], wkv_datarange[1], (coff * head_dim, hidden_size))).to(data_type)
    wgate = torch.tensor(np.random.uniform(wgate_datarange[0], wgate_datarange[1], (coff * head_dim, hidden_size))).to(data_type)
    ape = torch.tensor(np.random.uniform(ape_datarange[0], ape_datarange[1], (cmp_ratio, coff * head_dim))).to(torch.float32)
    norm_weight = torch.tensor(np.random.uniform(norm_weight_datarange[0], norm_weight_datarange[1], (head_dim))).to(data_type)
    rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type)
    rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type)
    ### ======================== gen input data finish =============================
    ### ======================== execute cpu start =================================
    cpu_kv_state = kv_state.clone()
    cpu_score_state = score_state.clone()

    test_operator = Generalized_operator()
    cpu_result, kv_mask_result = test_operator.forward( x,
                                        wkv,
                                        wgate,
                                        cpu_kv_state,
                                        cpu_score_state,
                                        ape,
                                        norm_weight, 
                                        rope_sin,
                                        rope_cos,
                                        block_table = block_table,
                                        cu_seqlens = cu_seqlens,
                                        seqused = seqused,
                                        start_pos = start_pos,
                                        rope_head_dim = rope_head_dim,
                                        cmp_ratio = cmp_ratio,
                                        coff = coff,
                                        norm_eps = norm_eps,
                                        rotary_mode = rotary_mode)
    update_kv = cpu_kv_state != kv_state
    update_score = cpu_score_state != score_state

    output_tensors = {
        "params":params,
        "cpu_result": cpu_result,
        "kv_mask_result": kv_mask_result,
        "update_kv":update_kv,
        "update_score":update_score,
        "cpu_kv_state":cpu_kv_state,
        "cpu_score_state":cpu_score_state,
        "x":x,
        "wkv":wkv,
        "wgate":wgate,
        "kv_state":kv_state,
        "score_state":score_state,
        "ape":ape,
        "norm_weight":norm_weight, 
        "rope_sin":rope_sin,
        "rope_cos":rope_cos,
        "block_table":block_table,
        "cu_seqlens":cu_seqlens,
        "seqused":seqused,
        "start_pos":start_pos,
        "rope_head_dim":rope_head_dim,
        "cmp_ratio":cmp_ratio,
        "coff":coff,
        "norm_eps":norm_eps,
        "rotary_mode":rotary_mode
    }
    return  casename, output_tensors

def save_test_case(test_cases, file_path):
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(file_path, exist_ok=True)

    for idx, case in enumerate(test_cases):
        try:
            case_name, output_tensors = compressor_output_single(case)
            # 生成文件名
            input_filename = f"{case_name}.pt"
            input_filepath = os.path.join(file_path, input_filename)

            # 保存数据
            torch.save(output_tensors, input_filepath)
            print(f"测试用例已保存到: {input_filepath}")

        except Exception as e:
            print(f"[失败] 生成 pt 文件失败: {case[0]} (索引: {idx})")
            print(f"错误详情: {e}")

def main():
    parser = argparse.ArgumentParser(description='compressor_pt_save.py 接收路径参数')
    parser.add_argument('path1', type=str, help='第一个路径')
    parser.add_argument('path2', type=str, help='第二个路径')
    args = parser.parse_args()
    path1 = args.path1
    path2 = args.path2
    testcase =  load_excel_test_cases(path1, "Sheet1")
    save_test_case(testcase, path2)

if __name__ == "__main__":
    main()

