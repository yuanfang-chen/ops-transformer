#!/usr/bin/env python3
# coding: utf-8
import random
import copy
import math
import numpy as np
from typing import Union, List

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig

@GENERATOR_REGISTRY.register("ascend_aclnn_fused_infer_attention_score_v3")
class SwaCompressedGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        # SWA 场景固定约束
        self.layout = 'TND'
        self.sparseMode = 4  # 必须为4

    def generate_integer_tokens(self, s1, s2):
        """
        核心逻辑：生成满足 SWA 约束的 preTokens 和 nextTokens
        Ref: 开发提供的数学逻辑
        """
        # 1. preToken 范围 (-s1 + 1, s2 - 1)
        preToken_min = -s1 + 1
        preToken_max = s2 - 1
        
        # 防御性校验
        if preToken_min > preToken_max:
             return 0, 0 

        preToken = random.randint(preToken_min, preToken_max)

        # 2. nextToken 范围 (-s2 + 1, s1 - 1)
        # 且必须满足 nextToken >= -preToken
        nextToken_min = max(-s2 + 1, -preToken)
        nextToken_max = s1 - 1

        if nextToken_min > nextToken_max:
             return self.generate_integer_tokens(s1, s2)

        nextToken = random.randint(nextToken_min, nextToken_max)
        return preToken, nextToken

    def generate_factor(self):
        """
        生成因子
        """
        self.D = random.randint(1, 256)
        self.blockSize = random.choice([16, 32, 64, 128])
        # 3. 随机覆盖 Page Attention
        self.use_page_attention = random.choice([True, False])
        self.use_atten_mask = random.choice([True])
        
        # --- 1. 规模控制 ---
        rand_prob = random.random()
        if rand_prob < 0.1:
            # [10%] 稍大场景 (单Batch)
            self.B = 1
            avg_seq_len = random.randint(2048, 4096)
        else:
            # [90%] 常规场景 (多Batch)
            self.B = random.randint(1, 4)
            avg_seq_len = random.randint(128, 1024)

        # --- 2. GQA/MHA ---
        self.KVN = random.randint(1, 8)
        # GQA Group: 1(MHA) or 2/4/8
        gqa_group = random.choice([2, 4, 8])
        self.QN = self.KVN * gqa_group
        
        # --- 3. Dtype ---
        self.queryDtype = random.choice(['fp16', 'bf16'])
        
        # --- 4. 生成 SeqLengths ---
        self.qSeqList = []
        self.kvSeqList = []
        self.totalQTokens = 0
        self.totalKvTokens = 0
        self.max_kv_len = 0

        min_q_seqlen = float('inf')
        min_kv_seqlen = float('inf')

        for _ in range(self.B):
            # Q SeqLen
            q_len = max(1, int(avg_seq_len * random.uniform(0.8, 1.2)))
            
            # KV SeqLen Constraint: qSeqlen <= kvSeqlen
            kv_len = q_len + random.randint(0, self.blockSize * 4)
            
            # 硬件约束上限
            if kv_len > 128 * 1024: kv_len = 128 * 1024
            self.max_kv_len = max(self.max_kv_len, kv_len)
            if q_len > kv_len: q_len = kv_len
            
            # 计算最小长度用于 SWA 参数生成
            min_q_seqlen = min(min_q_seqlen, q_len)
            min_kv_seqlen = min(min_kv_seqlen, kv_len)

            # 更新 Total
            self.totalQTokens += q_len
            self.totalKvTokens += kv_len

            # 列表存入累积长度 (Cumulative Lengths / Offsets)
            self.qSeqList.append(self.totalQTokens)
            self.kvSeqList.append(self.totalKvTokens)
    

        # --- 5. 生成 SWA 参数 ---
        self.preTokensVal, self.nextTokensVal = self.generate_integer_tokens(min_q_seqlen, min_kv_seqlen)

        # --- 6. 其他参数 ---
        # self.innerPrecise = random.choice([0, 1])
        # swa只能走innerPrecise=0
        self.innerPrecise = random.choice([0])
        self.softmaxLseFlag = random.choice([True, False])
        
        # Mask: 针对 sparseMode=4，强制生成 Mask (1)
        self.mask_type_flag = 1 

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        self.generate_factor()
        print(f"=== [SWA-Compressed] B:{self.B}, QN:{self.QN}, KVN:{self.KVN}, "
              f"Pre:{self.preTokensVal}, Next:{self.nextTokensVal}, "
              f"Sparse:{self.sparseMode}, MaskType:{self.mask_type_flag} ===")
        
        # 设置对应的执行器名称 (保持和你之前的一致)
        case_config.aclnn_api_type = "executor_aclnn_fused_infer_attention_score_v4_swa"
        case_config.api_type = "executor_fused_infer_attention_score_v4_swa"
        
        inputs_map = {}
        for item in case_config.inputs:
            if isinstance(item, list) and len(item) > 0:
                inputs_map[item[0].name] = item
            elif not isinstance(item, list):
                inputs_map[item.name] = item

        # --- Tensor 获取 ---
        query = inputs_map.get('query')
        key = inputs_map.get('key')
        value = inputs_map.get('value')
        attenMask = inputs_map.get('attenMaskOptional')
        blockTable = inputs_map.get('blockTableOptional')
        pseShift = inputs_map.get('pseShiftOptional') # 获取 pseShift
        
        attentionOut = inputs_map.get('attentionOut')
        softmaxLse = inputs_map.get('softmaxLse')

        # --- Attrs 获取 ---
        actualSeqLengths = inputs_map.get('actualSeqLengthsOptional')
        actualSeqLengthsKv = inputs_map.get('actualSeqLengthsKvOptional')
        
        numHeads = inputs_map.get('numHeads')
        numKeyValueHeads = inputs_map.get('numKeyValueHeads')
        sparseMode = inputs_map.get('sparseMode')
        innerPrecise = inputs_map.get('innerPrecise')
        inputLayout = inputs_map.get('inputLayout')
        blockSize = inputs_map.get('blockSize')
        softmaxLseFlag = inputs_map.get('softmaxLseFlag')
        scaleValue = inputs_map.get('scaleValue')
        preTokens = inputs_map.get('preTokens')
        nextTokens = inputs_map.get('nextTokens')

        # --- 1. 设置 Attr Values ---
        inputLayout.range_values = [self.layout]
        numHeads.range_values = [self.QN]
        numKeyValueHeads.range_values = [self.KVN]
        sparseMode.range_values = [self.sparseMode]
        innerPrecise.range_values = [self.innerPrecise]
        blockSize.range_values = [self.blockSize]
        scaleValue.range_values = [1.0 / (self.D ** 0.5)]
        
        preTokens.range_values = [self.preTokensVal]
        nextTokens.range_values = [self.nextTokensVal]

        if softmaxLseFlag:
            softmaxLseFlag.range_values = [self.softmaxLseFlag]

        # --- 2. 设置 Dtype ---
        for tensor in [query, key[0], value[0], attentionOut]:
            if tensor: tensor.dtype = self.queryDtype
        # if attenMask: attenMask.dtype = self.queryDtype 
        softmaxLse.dtype = 'fp32'

        # --- 3. 设置 Shape (TND Layout) ---
        query.shape = [self.totalQTokens, self.QN, self.D]
        attentionOut.shape = [self.totalQTokens, self.QN, self.D]
        softmaxLse.shape = [self.totalQTokens, self.QN, 1]
        
        kv_shape = [self.totalKvTokens, self.KVN, self.D]
        key[0].shape = kv_shape
        value[0].shape = kv_shape

        # --- 4. 设置 Mask & BlockTable & pseShift ---
        if not self.use_atten_mask:
            attenMask.range_values = ["None"]
        else:
            attenMask.shape = [2048, 2048]
        
        if self.use_page_attention:
            max_blocks_per_seq = (self.max_kv_len + self.blockSize - 1) // self.blockSize
            blockTable.shape = [self.B, max_blocks_per_seq]
        else:
            blockTable.range_values = ["None"]
            
        if pseShift:
            pseShift.range_values = ["None"]

        # --- 5. 处理 List 类型的 seqLengths ---
        def expand_seq_len_config(config_list, value_list):
            if not config_list: return
            template = config_list[0]
            config_list.clear()
            for b in range(self.B):
                new_item = copy.deepcopy(template)
                new_item.range_values = [value_list[b]]
                config_list.append(new_item)

        expand_seq_len_config(actualSeqLengths, self.qSeqList)
        expand_seq_len_config(actualSeqLengthsKv, self.kvSeqList)

        # --- 6. SoftmaxLse 输出控制 ---
        if not self.softmaxLseFlag:
            softmaxLse.range_values = [1]
            softmaxLse.shape = [0]

        return case_config