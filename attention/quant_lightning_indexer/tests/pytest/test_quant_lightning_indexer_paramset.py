#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

# 定义测试参数组合
TEST_PARAMS = {
    # "quant_li_default_a5":{
    #     "batch_size": [1],
    #     "q_seq": [1],
    #     "k_seq": [3000],
    #     "q_t_size":[1],
    #     "k_t_size":[3000],#压缩后的值
    #     "q_head_num": [24],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1]],
    #     "act_seq_k": [[3000]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [4],
    #     "q_seq": [3],
    #     "k_seq": [2304],
    #     "q_t_size":[12],
    #     "k_t_size":[2304],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,6,9,12]],
    #     "act_seq_k": [[1216,1216,1216,2304]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [512],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-0.01,-0.001]],
    #     "key_datarange":[[-448,448]],
    #     "weights_datarange":[[-163,163]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [8],
    #     "q_seq": [2],
    #     "k_seq": [8192],
    #     "q_t_size":[16],
    #     "k_t_size":[8192],#压缩后的值
    #     "q_head_num": [24],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[2,4,6,8,10,12,14,16]],
    #     "act_seq_k": [[8192,8192,8192,8192,8192,8192,8192,8192]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [8],
    #     "q_seq": [1454],
    #     "k_seq": [1454],
    #     "q_t_size":[11632],
    #     "k_t_size":[1454],#压缩后的值
    #     "q_head_num": [32],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1454,2908,4362,5816,7270,8724,10178,11632]],
    #     "act_seq_k": [[20,20,20,20,20,20,20,1454]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [4],
    #     "q_seq": [4095],
    #     "k_seq": [4095],
    #     "q_t_size":[16380],
    #     "k_t_size":[4095],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[4095,8190,12285,16380]],
    #     "act_seq_k": [[3936,2784,1152,4095]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-3,3]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [48],
    #     "q_seq": [3],
    #     "k_seq": [256],
    #     "q_t_size":[144],
    #     "k_t_size":[256],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144]],
    #     "act_seq_k": [[160,64,32,32,256,64,23,192,28,64,224,96,96,32,96,192,160,96,2,128,32,224,96,96,96,192,32,192,160,224,32,64,32,96,128,160,96,160,160,128,5,128,32,22,96,128,160,256]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [8],
    #     "q_seq": [12],
    #     "k_seq": [6144],
    #     "q_t_size":[96],
    #     "k_t_size":[6144],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[12,24,36,48,60,72,84,96]],
    #     "act_seq_k": [[5440,5440,5440,5440,5440,5440,5440,6144]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     # "key_datarange":[[1,1]],
    #     # "key_datarange":[[3,3]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     # "k_scale_datarange":[[1,1]],
    #     # "k_scale_datarange":[[16,16]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },#64pass 32 fail


    # "quant_li_default_a5":{
    #     "batch_size": [16],
    #     "q_seq": [4096],
    #     "k_seq": [4096],
    #     "q_t_size":[65536],
    #     "k_t_size":[4096],#压缩后的值
    #     "q_head_num": [24],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[4096,8192,12288,16384,20480,24576,28672,32768,36864,40960,45056,49152,53248,57344,61440,65536]],
    #     "act_seq_k": [[0,1331,75,439,3102,3730,4008,697,921,1829,1067,3788,848,1748,3500,4096]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-1,1]],
    #     "weights_datarange":[[-209,209]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{
    #     "batch_size": [4],
    #     "q_seq": [4095],
    #     "k_seq": [4095],
    #     "q_t_size":[16380],
    #     "k_t_size":[4095],#压缩后的值
    #     "q_head_num": [24],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[140],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[4095,8190,12285,16380]],
    #     "act_seq_k": [[3936,2784,1152,4095]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-3,3]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 8_de
    #     "batch_size": [1],
    #     "q_seq": [1],
    #     "k_seq": [2048],
    #     "q_t_size":[1],
    #     "k_t_size":[2048],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [1], # 取16的整数倍，最多支持到1024
    #     "block_num":[1],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1]],
    #     "act_seq_k": [[2048]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["TND"],
    #     "sparse_count": [1545],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-3,3]],
    #     "key_datarange":[[-0.01,-0.001]],
    #     "weights_datarange":[[-24,24]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    "quant_li_default_a5":{ # 1
        "batch_size": [18],
        "q_seq": [3],
        "k_seq": [3072],
        "q_t_size":[1],
        "k_t_size":[1],#压缩后的值
        "q_head_num": [16],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[469],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype":[torch.bfloat16],
        "dequant_dtype": [torch.float32],
        "actual_seq_dtype": [torch.int32],
        "act_seq_q": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]],
        "act_seq_k": [[3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3016,3072]], #PA场景非前缀和，表示每个batch_size的实际token数
        "query_quant_mode": [0],
        "key_quant_mode": [0],
        "layout_query": ["BSND"],
        "layout_key":["PA_BSND"],
        "sparse_count": [2048],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[0,255]],
        "k_scale_datarange":[[0,65504]],
        "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    },

    # "quant_li_default_a5":{ # 1_1 block
    #     "batch_size": [18],
    #     "q_seq": [3],
    #     "k_seq": [128],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [16],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[18],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]],
    #     "act_seq_k": [[128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [64],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[-1,1]],
    #     "weights_datarange":[[-130,130]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 1_2 block
    #     "batch_size": [18],
    #     "q_seq": [3],
    #     "k_seq": [127],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [16],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[36],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]],
    #     "act_seq_k": [[127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [64],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[-1,1]],
    #     "weights_datarange":[[-130,130]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 4
    #     "batch_size": [1],
    #     "q_seq": [8192],
    #     "k_seq": [8192],
    #     "q_t_size":[8192],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [24],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[73],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[8192]],
    #     "act_seq_k": [[8192]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-244,244]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 5
    #     "batch_size": [4],
    #     "q_seq": [4096],
    #     "k_seq": [16384],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [24],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[289],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1856,1856,1856,1856]],
    #     "act_seq_k": [[6187,4852,4665,16384]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[-2,1]],
    #     "weights_datarange":[[-66,66]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 8
    #     "batch_size": [21],
    #     "q_seq": [1],
    #     "k_seq": [2048],
    #     "q_t_size":[21],
    #     "k_t_size":[43008],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [1], # 取16的整数倍，最多支持到1024
    #     "block_num":[1],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]],
    #     "act_seq_k": [[2048,4096,6144,8192,10240,12288,14336,16384,18432,20480,22528,24576,26624,28672,30720,32768,34816,36864,38912,40960,43008]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["TND"],
    #     "sparse_count": [1545],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-3,3]],
    #     "key_datarange":[[-0.01,-0.001]],
    #     "weights_datarange":[[-24,24]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 9
    #     "batch_size": [16],
    #     "q_seq": [2],
    #     "k_seq": [3072],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [1], # 取16的整数倍，最多支持到1024
    #     "block_num":[1],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
    #     "act_seq_k": [[2842,2842,2842,2842,2842,2842,2842,2842,2842,2842,2842,2842,2842,2842,2842,2842]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [0],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[-1,-1]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 10
    #     "batch_size": [24],
    #     "q_seq": [3],
    #     "k_seq": [512],
    #     "q_t_size":[72],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[56],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72]],
    #     "act_seq_k": [[128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,512]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [706],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[1,2]],
    #     "weights_datarange":[[-177,177]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 11
    #     "batch_size": [12],
    #     "q_seq": [4],
    #     "k_seq": [3072],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[249],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,4,2,2,2,1,4,2,3,4,3,3]],
    #     "act_seq_k": [[2336,2080,2816,2656,3040,3040,3008,2784,2624,2240,32,3072]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[0.01,1]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 12
    #     "batch_size": [1],
    #     "q_seq": [8192],
    #     "k_seq": [3072],
    #     "q_t_size":[8192],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[45],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[8192]],
    #     "act_seq_k": [[3072]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [0],
    #     "query_datarange":[[-1,1]],
    #     "key_datarange":[[-1,1]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 13
    #     "batch_size": [2],
    #     "q_seq": [4096],
    #     "k_seq": [4096],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[65],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[942,843]],
    #     "act_seq_k": [[3087,4096]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[1,2]],
    #     "key_datarange":[[-0.01,0.01]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 14
    #     "batch_size": [4],
    #     "q_seq": [2048],
    #     "k_seq": [4096],
    #     "q_t_size":[8192],
    #     "k_t_size":[16384],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[1],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[2048,4096,6144,8192]],
    #     "act_seq_k": [[4096,8192,12288,16384]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["TND"],
    #     "sparse_count": [1762],
    #     "sparse_mode": [3],
    #     "query_datarange":[[0,0.001]],
    #     "key_datarange":[[-5,5]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{ # 15
    #     "batch_size": [8],
    #     "q_seq": [1024],
    #     "k_seq": [3072],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[1],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[527,599,272,897,718,642,108,980]],
    #     "act_seq_k": [[2980,2155,2216,173,2193,2252,2639,2636]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [0],
    #     "query_datarange":[[0,0.001]],
    #     "key_datarange":[[-3,3]],
    #     "weights_datarange":[[-2,2]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{
    #     "batch_size": [1],
    #     "q_seq": [512],
    #     "k_seq": [65536],
    #     "q_t_size":[1],
    #     "k_t_size":[1],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [256], # 取16的整数倍，最多支持到1024
    #     "block_num":[256],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[512]],
    #     "act_seq_k": [[65536]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["BSND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-100,100]],
    #     "key_datarange":[[-100,100]],
    #     "weights_datarange":[[-100,100]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{
    #     "batch_size": [4],
    #     "q_seq": [2048],
    #     "k_seq": [4096],
    #     "q_t_size":[8192],
    #     "k_t_size":[16384],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [1], # 取16的整数倍，最多支持到1024
    #     "block_num":[1],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[2048,4096,6144,8192]],
    #     "act_seq_k": [[4096,8192,12288,16384]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["TND"],
    #     "sparse_count": [1762],
    #     "sparse_mode": [3],
    #     "query_datarange":[[0,0.001]],
    #     "key_datarange":[[-5,5]],
    #     "weights_datarange":[[-255,255]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },

    # "quant_li_default_a5":{
    #     "batch_size": [4],
    #     "q_seq": [3],
    #     "k_seq": [2304],
    #     "q_t_size":[12],
    #     "k_t_size":[2304],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,6,9,12]],
    #     "act_seq_k": [[1216,1216,1216,2304]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-0.01,-0.001]],
    #     "key_datarange":[[-448,448]],
    #     "weights_datarange":[[-163,163]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [2],
    #     "q_seq": [1],
    #     "k_seq": [2304],
    #     "q_t_size":[4],
    #     "k_t_size":[2304],#压缩后的值
    #     "q_head_num": [32],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,4]],
    #     "act_seq_k": [[1216,2304]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     # "act_seq_k": [[1216,3000]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-0.01,-0.001]],
    #     # "query_datarange":[[1,1]],
    #     "key_datarange":[[-448,448]],
    #     # "key_datarange":[[1,1]],
    #     "weights_datarange":[[-163,163]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     # "k_scale_datarange":[[1,1]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [24],
    #     "q_seq": [1600],
    #     "k_seq": [1600],
    #     "q_t_size":[38400],
    #     "k_t_size":[1600],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[1600,3200,4800,6400,8000,9600,11200,12800,14400,16000,17600,19200,20800,22400,24000,25600,27200,28800,30400,32000,33600,35200,36800,38400]],
    #     "act_seq_k": [[670,1404,830,872,613,585,1297,326,855,247,687,64,564,1305,673,1162,318,568,925,733,1413,377,1437,1600]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a5":{
    #     "batch_size": [64],
    #     "q_seq": [916],
    #     "k_seq": [916],
    #     "q_t_size":[58624],
    #     "k_t_size":[916],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[916,1832,2748,3664,4580,5496,6412,7328,8244,9160,10076,10992,11908,12824,13740,14656,15572,16488,17404,18320,19236,20152,21068,21984,22900,23816,24732,25648,26564,27480,28396,29312,30228,31144,32060,32976,33892,34808,35724,36640,37556,38472,39388,40304,41220,42136,43052,43968,44884,45800,46716,47632,48548,49464,50380,51296,52212,53128,54044,54960,55876,56792,57708,58624]],
    #     "act_seq_k": [[128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,916]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [2048],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-448,448]],
    #     "key_datarange":[[-20,20]],
    #     "weights_datarange":[[-123,123]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
    # "quant_li_default_a3":{
    #     "batch_size": [1],
    #     "q_seq": [3],
    #     "k_seq": [8192],
    #     "q_t_size":[3],
    #     "k_t_size":[8192],#压缩后的值
    #     "q_head_num": [64],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[17],
    #     "qk_dtype": [torch.int8],
    #     "dequant_dtype": [torch.float16],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3]],
    #     "act_seq_k": [[8196]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [512],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-100,100]],
    #     "key_datarange":[[-100,100]],
    #     "weights_datarange":[[-25,25]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[4] #1/2/4/8/16/32/64/128
    # }
    # "quant_li_default_a5":{
    #     "batch_size": [4],
    #     "q_seq": [3],
    #     "k_seq": [2304],
    #     "q_t_size":[12],
    #     "k_t_size":[2304],#压缩后的值
    #     "q_head_num": [16],
    #     "k_head_num": [1],
    #     "head_dim": [128],
    #     "block_size": [128], # 取16的整数倍，最多支持到1024
    #     "block_num":[80000],
    #     "qk_dtype": [torch.float8_e4m3fn],
    #     "dequant_dtype": [torch.float32],
    #     "actual_seq_dtype": [torch.int32],
    #     "act_seq_q": [[3,6,9,12]],
    #     "act_seq_k": [[1216,1216,1216,2304]], #PA场景非前缀和，表示每个batch_size的实际token数
    #     "query_quant_mode": [0],
    #     "key_quant_mode": [0],
    #     "layout_query": ["TND"],
    #     "layout_key":["PA_BSND"],
    #     "sparse_count": [512],
    #     "sparse_mode": [3],
    #     "query_datarange":[[-0.01,-0.001]],
    #     "key_datarange":[[-448,448]],
    #     "weights_datarange":[[-163,163]],
    #     "q_scale_datarange":[[0,255]],
    #     "k_scale_datarange":[[0,65504]],
    #     "cmp_ratio":[1] #1/2/4/8/16/32/64/128
    # },
}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
if "Ascend910_93" in properties.name:
    ENABLED_PARAMS = [TEST_PARAMS["quant_li_default_a3"]]
elif "Ascend950" in properties.name:
    ENABLED_PARAMS = [TEST_PARAMS["quant_li_default_a5"]]