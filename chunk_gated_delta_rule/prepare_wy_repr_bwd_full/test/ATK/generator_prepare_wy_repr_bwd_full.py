# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import random
from typing import Union, List

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig


K_INDEX = 0
V_INDEX = 1
BETA_INDEX = 2
A_INDEX = 3
DA_INDEX = 4
DW_INDEX = 5
DU_INDEX = 6
G_INDEX = 7
CU_SEQ_LEN_INDEX= 8
CHUNK_INDICES_INDEX = 9
CHUNK_SIZE_INDEX=10
QKV_TYOE_INDEX = 13


@GENERATOR_REGISTRY.register("generator_prepare_wy_repr_bwd_full")
class ChunkBwdDvLocalGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        qkv_type = "fp16"
        if case_config.inputs[QKV_TYOE_INDEX].dtype == "bf16":
            qkv_type = "bf16"

        case_config.inputs[K_INDEX].dtype = qkv_type
        case_config.inputs[V_INDEX].dtype = qkv_type
        case_config.inputs[A_INDEX].dtype = qkv_type
        case_config.inputs[DA_INDEX].dtype = qkv_type
        case_config.inputs[DW_INDEX].dtype = qkv_type
        case_config.inputs[DU_INDEX].dtype = qkv_type
        if case_config.inputs[G_INDEX].dtype != "fp32":
            case_config.inputs[G_INDEX].dtype = qkv_type
        case_config.inputs[BETA_INDEX].dtype = case_config.inputs[G_INDEX].dtype
        B = 1
        H = 4
        T = 512
        K = 128
        V = 128
        chunk_size = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        print(chunk_size)
        case_config.inputs[K_INDEX].shape = [B, H, T, K]
        case_config.inputs[V_INDEX].shape = [B, H, T, V]
        case_config.inputs[BETA_INDEX].shape = [B, H, T]
        case_config.inputs[A_INDEX].shape = [B, H, T, chunk_size]
        case_config.inputs[DA_INDEX].shape = [B, H, T, chunk_size]
        case_config.inputs[DW_INDEX].shape = [B, H, T, K]
        case_config.inputs[DU_INDEX].shape = [B, H, T, V]
        case_config.inputs[G_INDEX].shape = [B, H, T]
        # case_config.inputs[CU_SEQ_LEN_INDEX].shape = [B, H, T, V]
        # case_config.inputs[CHUNK_INDICES_INDEX].shape = [B, H, T, V]

        # chunkSize = case_config.inputs[CHUNK_SIZE_INDEX].range_values

        return case_config