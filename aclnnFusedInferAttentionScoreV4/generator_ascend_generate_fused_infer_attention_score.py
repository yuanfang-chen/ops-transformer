#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import random
from typing import Union, List
import numpy as np
import math

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat,AclTensorList
"""
FIAV3的splitFuse功能下支持了当前低精度计算功能，该功能的开启条件如下：

支持Q的layout为TND
支持KV的layout为TND
仅支持QKV的输入为fp16
KV的headDim必须相等，且上限为256，且需要是16的整数倍
仅支持MHA: 
    MHA (Multi-Head Attention) num_heads == num_key_value_heads, 
    GQA (Grouped-Query Attention) num_heads = G * num_key_value_heads(G > 1)
仅支持innerPrecise=1
不支持输入qRope, kRope, learnableSink
sparseMode只支持0且不传mask
"""

@GENERATOR_REGISTRY.register("ascend_aclnn_fused_infer_attention_score_v3")
class DefaultGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_factor(self):
        # tensor输入参数
        self.B = random.randint(1, 10)
        self.KVN = random.randint(1, 8)
        
        # [Constraint] 仅支持 MHA，QN 必须等于 KVN
        self.QN = self.KVN 
        
        # B060支持到256
        self.D = random.randint(1, 256)
        
        self.CeilS = random.randint(1, 2048)
        
        # [Constraint] 必须关闭 PagedAttention 才能生成 KV 的 TND 布局
        self.pageAttentionFlag = False 
        
        if self.pageAttentionFlag:
            self.BlockSize = 128
        
        # [Constraint] sparseMode只支持0且不传mask
        self.sparseMode = 0

        # [Constraint] 仅支持innerPrecise=1
        self.innerPrecise = 1

        self.softmaxLseFlag = random.choice([True, False])

        # [Constraint] 支持Q、KV的layout为TND
        self.layout = 'TND'
        
        # [Constraint] 仅支持QKV输入为fp16
        self.queryDtype = 'fp16'

        self.kvDtype = self.queryDtype

        self.qSeqList = []
        self.kvSeqList = []
        self.totalQTokens = 0
        self.totalKvTokens = 0
        self.maxQS = 0
        self.maxKvS = 0
        for i in range(self.B):
            qSeqCurB = random.randint(1, self.CeilS)
            kvSeqCurB = random.randint(qSeqCurB, self.CeilS)
            self.maxQS = max(self.maxQS, qSeqCurB)
            self.maxKvS = max(self.maxKvS, kvSeqCurB)
            self.totalQTokens += qSeqCurB
            self.totalKvTokens += kvSeqCurB
            if self.layout == 'TND':
                self.qSeqList.append(self.totalQTokens)
                if self.pageAttentionFlag:
                    self.kvSeqList.append(kvSeqCurB)
                else:
                    self.kvSeqList.append(self.totalKvTokens)
            else:
                self.qSeqList.append(qSeqCurB)
                self.kvSeqList.append(kvSeqCurB)
        self.maxNumBlocksPerQuery = 0
        self.numBlocks = 0
        if self.pageAttentionFlag:
            self.maxNumBlocksPerQuery = (self.maxKvS + self.BlockSize - 1) // self.BlockSize
            minNumBlocks = self.maxNumBlocksPerQuery * self.B
            self.numBlocks = minNumBlocks
        
        # print("======kvSeqlist======")
        # print(self.kvSeqList)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        print("===start parse (after_case_config)===")
        self.generate_factor()
        
        
        # 1. 整理输入：将 inputs 列表映射为字典
        # case_config.inputs 可能包含 InputCaseConfig 对象，也可能包含 List[InputCaseConfig] (attrs)
        inputs_map = {}
        
        for item in case_config.inputs:
            if isinstance(item, list):
                # 这是一个 attrs 列表 (如 actualSeqLengthsOptional)
                if len(item) > 0:
                    inputs_map[item[0].name] = item
            else:
                # 这是一个普通的 Tensor/Attribute
                inputs_map[item.name] = item

        # 2. 提取变量
        query = inputs_map.get('query')
        key = inputs_map.get('key')
        value = inputs_map.get('value')
        
        attenMask = inputs_map.get('attenMaskOptional')
        blockTable = inputs_map.get('blockTableOptional')
        
        attentionOut = inputs_map.get('attentionOut')
        softmaxLse = inputs_map.get('softmaxLse')

        # 列表类型的输入 (注意：这里拿到的是列表对象的引用)
        actualSeqLengths = inputs_map.get('actualSeqLengthsOptional')
        actualSeqLengthsKv = inputs_map.get('actualSeqLengthsKvOptional')

        # 属性 Scalar
        numHeads = inputs_map.get('numHeads')
        numKeyValueHeads = inputs_map.get('numKeyValueHeads')
        sparseMode = inputs_map.get('sparseMode')
        innerPrecise = inputs_map.get('innerPrecise')
        inputLayout = inputs_map.get('inputLayout')
        blockSize = inputs_map.get('blockSize')
        softmaxLseFlag = inputs_map.get('softmaxLseFlag')
        scaleValue = inputs_map.get('scaleValue')

        # ---------------- Logic Start ----------------

        # --- 基础标量 ---
        inputLayout.range_values = [self.layout]
        numHeads.range_values = [self.QN]
        numKeyValueHeads.range_values = [self.KVN]
        innerPrecise.range_values = [self.innerPrecise]
        sparseMode.range_values = [self.sparseMode]
        scaleValue.range_values = [1.0 / (self.D ** 0.5)]
        
        # if blockSize: 
        #     blockSize.range_values = [self.BlockSize]
        if softmaxLseFlag: 
            softmaxLseFlag.range_values = [self.softmaxLseFlag]

        # --- Dtype ---
        query.dtype = self.queryDtype
        key[0].dtype = self.queryDtype
        value[0].dtype = self.queryDtype
        attentionOut.dtype = self.queryDtype
        
        # SoftmaxLse 即使在 fp16 计算下，通常输出也是 fp32
        softmaxLse.dtype = 'fp32'
        if blockTable:
            blockTable.dtype = 'int32'

        # --- Shape & Layout ---
        if self.layout == 'TND':
            query.shape = [self.totalQTokens, self.QN, self.D]
            attentionOut.shape = [self.totalQTokens, self.QN, self.D]
            softmaxLse.shape = [self.totalQTokens, self.QN, 1]
            
            if self.pageAttentionFlag:
                # BSH / Paged Layout
                kv_shape = [self.numBlocks, self.BlockSize, self.KVN * self.D]
                key[0].shape = kv_shape
                value[0].shape = kv_shape
            else:
                # TND Layout for KV
                kv_shape = [self.totalKvTokens, self.KVN, self.D]
                key[0].shape = kv_shape
                value[0].shape = kv_shape

        # --- Mask ---
        # 要求 sparseMode=0 且不传 Mask
        attenMask.range_values = ["None"]

        # --- Block Table ---
        if self.pageAttentionFlag and blockTable:
             blockTable.shape = [self.B, self.maxNumBlocksPerQuery]
             blockTable.range_values = [0, self.numBlocks]
        elif blockTable:
             blockTable.range_values = ["None"]

        # --- Softmax Lse Flag ---
        if not self.softmaxLseFlag:
            softmaxLse.range_values = [1]
            softmaxLse.shape = [0]

        # --- Seq Lengths (List 处理) ---
        # 直接修改列表内容：先截断，再赋值
        
        # 1. 处理 actualSeqLengths
        if actualSeqLengths:
            # 截断多余的元素 (原位修改)
            del actualSeqLengths[self.B:]
            # 赋值
            for i, item in enumerate(actualSeqLengths):
                item.range_values = [self.qSeqList[i]]

        # 2. 处理 actualSeqLengthsKv
        if actualSeqLengthsKv:
            del actualSeqLengthsKv[self.B:]
            for i, item in enumerate(actualSeqLengthsKv):
                item.range_values = [self.kvSeqList[i]]

        # 不需要重新赋值 case_config.inputs，因为是对其中的 List 对象做了原位修改
        return case_config