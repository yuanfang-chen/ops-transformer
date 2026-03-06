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

@GENERATOR_REGISTRY.register("ascend_aclnn_fused_infer_attention_score_v4")
class DualScenarioGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
    
    def get_boundary_kv_len(self, q_len: int, max_limit: int = 16384) -> int:
        """
        构造针对 Tiling 边界的 KV 长度
        规则: KV - Q = BlockSize * n ± 1
        """
        # Tiling BlockSize 通常是 128 (也可能是 64 或 256，这里按需求取 128)
        block_size = 128
        
        # 随机选择倍数 n
        # 确保 kv 至少比 q 大一点，且不要超太多以免 OOM
        max_n = (max_limit - q_len) // block_size
        if max_n < 1: 
            return q_len + 1 # 空间不够，回退到普通逻辑
            
        n = random.randint(1, min(8, max_n))
        
        # 随机选择边界偏移: -1 或 +1
        offset = random.choice([-1, 1])
        
        diff = block_size * n + offset
        kv_len = q_len + diff
        
        # 最终校验
        if kv_len > max_limit: kv_len = max_limit
        if kv_len < q_len: kv_len = q_len # 理论上不会发生，防御性编程
        
        return kv_len

    def generate_factor(self, case_config):
        # --- 1. 场景与规模概率控制 ---
        rand_prob = random.random()
        
        if rand_prob < 0.01:
            # [0.01] 极限长序列
            self.mode_tag = "Extreme-16k"
            self.B = 1
            target_total_tokens = 16384
        elif rand_prob < 0.31:
            # [0.30] 大规模场景
            self.mode_tag = "Large-32k"
            self.B = random.randint(1, 50) 
            target_total_tokens = random.randint(28000, 32000)
        else:
            # [0.69] 极速场景
            self.mode_tag = "Fast-2k"
            self.B = random.randint(1, 16)
            target_total_tokens = random.randint(1024, 2048)

        # --- 2. 场景选择 ---
        self.scenario = random.choice(['MHA', 'GQA'])
        # self.D = random.choice([16,32,48,64,80,96,112,128,256])
        self.D = random.randint(1, 256)

        # --- 3. 场景参数 ---
        if self.scenario == 'MHA':
            self.queryDtype = 'fp16'
            self.kvDtype = 'fp16'
            self.KVN = random.randint(1, 8) 
            self.QN = self.KVN  
            self.innerPrecise = 1
            self.sparseMode = 0 
            self.blockTableValid = False
        else:
            self.queryDtype = random.choice(['fp16', 'bf16'])
            self.kvDtype = self.queryDtype
            self.KVN = random.randint(1, 4)
            self.gqa_group = random.randint(2, 4)
            self.QN = self.KVN * self.gqa_group
            self.innerPrecise = 0
            self.sparseMode = random.choice([0, 0, 3]) 
            self.blockTableValid = False 

        if case_config.is_boundary:
            self.generate_upper_case(case_config)

        # --- 4. 生成 SeqLengths (核心修改) ---
        self.layout = 'TND'
        self.softmaxLseFlag = random.choice([True, False])
        
        self.qSeqList = []      
        self.kvSeqList = []    
        self.totalQTokens = 0
        self.totalKvTokens = 0
        
        self.pre_tokens_val = 128
        self.next_tokens_val = 128

        avg_seq = max(1, target_total_tokens // self.B)
        
        # 边界测试触发概率
        PROB_BOUNDARY_TEST = 0.2

        for i in range(self.B):
            # 1. 生成 Q SeqLen
            if self.mode_tag == "Extreme-16k":
                qSeqCurB = 16384
            else:
                base_seq = int(avg_seq * random.uniform(0.8, 1.2))
                qSeqCurB = max(1, min(base_seq, 16384))
            
            # 2. 生成 KV SeqLen (集成 Tiling 边界逻辑)
            # 如果不是极限模式，且触发了概率，则构造特殊 KV 长度
            if self.mode_tag != "Extreme-16k" and random.random() < PROB_BOUNDARY_TEST:
                kvSeqCurB = self.get_boundary_kv_len(qSeqCurB)
            else:
                # 普通逻辑: 随机增加一点点
                kv_increment = random.randint(0, 32)
                kvSeqCurB = qSeqCurB + kv_increment
            
            # 二次校验上限
            if kvSeqCurB > 16384:
                kvSeqCurB = 16384
                if qSeqCurB > 16384: qSeqCurB = 16384

            self.totalQTokens += qSeqCurB
            self.totalKvTokens += kvSeqCurB
            
            self.qSeqList.append(self.totalQTokens)
            self.kvSeqList.append(self.totalKvTokens)

    def generate_upper_case(self,case_config):
        # 边界用例 可根据atten_mask的shape判断是否为上下边界用例，以便构造边界用例的shape
        if 2147483649 in case_config.inputs[4].shape:
            # 上边界用例
            axis_constraints = {
                "B": 100,
                "QN": 256,
                "S_Q": 65536,
                "D": 256 
            }
            # 1. 随机选择一个轴
            selected_axis = random.choice(list(axis_constraints.keys()))
            boundary_value = axis_constraints[selected_axis]
            setattr(self, selected_axis, boundary_value)
            self.KVN = self.QN

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        self.generate_factor(case_config)
        print(f"=== [{self.mode_tag}] Scenario: {self.scenario} | B: {self.B} | TotalQ: {self.totalQTokens} | Sparse: {self.sparseMode} ===")
        
        inputs_map = {
            item[0].name if isinstance(item, list) and item else item.name: item
            for item in case_config.inputs
        }

        query = inputs_map.get('query')
        key = inputs_map.get('key')
        value = inputs_map.get('value')
        attentionOut = inputs_map.get('attentionOut')
        softmaxLse = inputs_map.get('softmaxLse')
        
        attenMask = inputs_map.get('attenMaskOptional')
        blockTable = inputs_map.get('blockTableOptional')
        
        actualSeqLengths = inputs_map.get('actualSeqLengthsOptional')
        actualSeqLengthsKv = inputs_map.get('actualSeqLengthsKvOptional')

        numHeads = inputs_map.get('numHeads')
        numKeyValueHeads = inputs_map.get('numKeyValueHeads')
        sparseMode = inputs_map.get('sparseMode')
        innerPrecise = inputs_map.get('innerPrecise')
        inputLayout = inputs_map.get('inputLayout')
        softmaxLseFlag = inputs_map.get('softmaxLseFlag')
        scaleValue = inputs_map.get('scaleValue')
        preTokens = inputs_map.get('preTokens')
        nextTokens = inputs_map.get('nextTokens')

        inputLayout.range_values = [self.layout]
        numHeads.range_values = [self.QN]
        numKeyValueHeads.range_values = [self.KVN]
        innerPrecise.range_values = [self.innerPrecise]
        sparseMode.range_values = [self.sparseMode]
        scaleValue.range_values = [1.0 / (self.D ** 0.5)]
        
        preTokens.range_values = [self.pre_tokens_val]
        nextTokens.range_values = [self.next_tokens_val]

        if softmaxLseFlag:
            softmaxLseFlag.range_values = [self.softmaxLseFlag]

        for tensor in [query, key[0], value[0], attentionOut]:
            if tensor: tensor.dtype = self.queryDtype
        softmaxLse.dtype = 'fp32'
        
        query.shape = [self.totalQTokens, self.QN, self.D]
        attentionOut.shape = [self.totalQTokens, self.QN, self.D]
        softmaxLse.shape = [self.totalQTokens, self.QN, 1]
        
        kv_shape = [self.totalKvTokens, self.KVN, self.D]
        key[0].shape = kv_shape
        value[0].shape = kv_shape

        if blockTable: 
            blockTable.range_values = ["None"]

        if self.sparseMode == 0:
            if attenMask: attenMask.range_values = ["None"]
        elif self.sparseMode in [3]:
            FIXED_S = 2048
            if attenMask:
                attenMask.shape = [FIXED_S, FIXED_S]
                attenMask.dtype = self.queryDtype 
        else:
            if attenMask: attenMask.range_values = ["None"]

        if not self.softmaxLseFlag:
            softmaxLse.range_values = [1]
            softmaxLse.shape = [0]

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

        return case_config