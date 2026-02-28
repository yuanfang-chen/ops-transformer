import random
import math
import copy
from typing import Union, List

try:
    import numpy as np
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.configs.case_config import InputCaseConfig, CaseConfig
except ImportError:
    pass

@GENERATOR_REGISTRY.register("generate_rain_fusion_attention")
class RainFusionAttentionGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        # 这些默认值会被具体的 Scenario 覆盖
        self.q_input_layout = "TND"
        self.kv_input_layout = "TND"
        self.dtype = "fp16"

        # 定义三种固定的场景配置
        self.scenarios = [
            # 场景 1: 典型Shape场景 (Customer)
            {
                "name": "Typical_Customer",
                "q_shape": [48484, 5, 128],
                "kv_shape": [48484, 5, 128],
                "dtype": "fp16",
                "block_shape": [128, 128],
                "q_seq_lens": [12121, 12121, 12121, 12121],
                "kv_seq_lens": [12121, 12121, 12121, 12121],
                "inner_precise": 0
            },
            # 场景 2: 泛化场景 1
            {
                "name": "Generalization_1",
                "q_shape": [18028, 8, 64],
                "kv_shape": [13120, 2, 64],
                "dtype": "bf16",
                "block_shape": [256, 256],
                "q_seq_lens": [4507, 4507, 4507, 4507],
                "kv_seq_lens": [3280, 3280, 3280, 3280],
                "inner_precise": 0
            },
            # 场景 3: 泛化场景 2
            {
                "name": "Generalization_2",
                "q_shape": [26040, 8, 128],
                "kv_shape": [36712, 8, 128],
                "dtype": "fp16",
                "block_shape": [256, 256],
                "q_seq_lens": [3255, 3255, 3255, 3255, 3255, 3255, 3255, 3255],
                "kv_seq_lens": [4589, 4589, 4589, 4589, 4589, 4589, 4589, 4589],
                "inner_precise": 1
            }
        ]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # ==========================================
        # 1. 随机选择一个固定场景
        # ==========================================
        scenario = random.choice(self.scenarios)
        # 如果需要调试特定场景，可以暂时写死，例如:
        # scenario = self.scenarios[0] 
        
        print(f"[Generator] Selected Scenario: {scenario['name']}")

        # 提取参数
        q_lens = scenario["q_seq_lens"]
        kv_lens = scenario["kv_seq_lens"]
        bs_x, bs_y = scenario["block_shape"]
        dtype = scenario["dtype"]
        batch_size = len(q_lens)
        
        # 维度信息
        total_tokens_q, q_head_num, head_dim = scenario["q_shape"]
        total_tokens_kv, kv_head_num, _ = scenario["kv_shape"]
        
        # ==========================================
        # 2. 设置 Block Shape
        # ==========================================
        # Index 5: blockShape
        x_block_obj = case_config.inputs[5][0]
        y_block_obj = case_config.inputs[5][1]
        
        x_block_obj.range_values = bs_x
        y_block_obj.range_values = bs_y
        
        # ==========================================
        # 3. 设置 Batch 和 SeqLens
        # ==========================================
        # Index 7: actualSeqLenghts (Q)
        q_seq_input_list = case_config.inputs[7]
        # Index 8: actualSeqLenghtsKv (KV)
        kv_seq_input_list = case_config.inputs[8]
        
        # 调整 Q List 长度以匹配 Batch
        current_q_batch = len(q_seq_input_list)
        if current_q_batch < batch_size:
            base_item = q_seq_input_list[-1] if current_q_batch > 0 else None
            for _ in range(batch_size - current_q_batch):
                q_seq_input_list.append(copy.deepcopy(base_item))
        elif current_q_batch > batch_size:
            del q_seq_input_list[batch_size:]

        # 调整 KV List 长度以匹配 Batch
        current_kv_batch = len(kv_seq_input_list)
        if current_kv_batch < batch_size:
            base_item = kv_seq_input_list[-1] if current_kv_batch > 0 else None
            for _ in range(batch_size - current_kv_batch):
                kv_seq_input_list.append(copy.deepcopy(base_item))
        elif current_kv_batch > batch_size:
            del kv_seq_input_list[batch_size:]

        # 填充具体的 Length 数值
        total_q_blocks = 0
        max_kv_block_num_global = 0

        for b in range(batch_size):
            # 设置 Q Len
            q_len = q_lens[b]
            q_seq_input_list[b].range_values = q_len
            q_seq_input_list[b].dtype = 'int'
            
            # 设置 KV Len
            kv_len = kv_lens[b]
            kv_seq_input_list[b].range_values = kv_len
            kv_seq_input_list[b].dtype = 'int'
            
            # 计算 Block 数量
            s_block_num_q = (q_len + bs_x - 1) // bs_x
            s_block_num_kv = (kv_len + bs_y - 1) // bs_y
            
            total_q_blocks += s_block_num_q
            max_kv_block_num_global = max(max_kv_block_num_global, s_block_num_kv)

        # ==========================================
        # 4. 设置 Tensor Shape 和 Dtype
        # ==========================================
        
        # 0: Query
        case_config.inputs[0].shape = [total_tokens_q, q_head_num, head_dim]
        case_config.inputs[0].dtype = dtype
        
        # 1: Key
        case_config.inputs[1].shape = [total_tokens_kv, kv_head_num, head_dim]
        case_config.inputs[1].dtype = dtype
        case_config.inputs[1].range_values = case_config.inputs[0].range_values
        
        # 2: Value
        case_config.inputs[2].shape = [total_tokens_kv, kv_head_num, head_dim]
        case_config.inputs[2].dtype = dtype
        case_config.inputs[2].range_values = case_config.inputs[0].range_values
        
        # 3: selectIdx
        # 确保 max_kv_block_num_global 至少为 1
        max_kv_block_num_global = max(1, max_kv_block_num_global)
        case_config.inputs[3].shape = [total_q_blocks, q_head_num, max_kv_block_num_global]
        case_config.inputs[3].range_values = [-1, max_kv_block_num_global - 1]
        
        # 4: selectNumIds
        case_config.inputs[4].shape = [total_q_blocks, q_head_num]
        case_config.inputs[4].range_values = [0, max_kv_block_num_global]
        
        # 12: numKeyValueHeads (如果有)
        if len(case_config.inputs) > 12:
            input_name = case_config.inputs[12].name
            if input_name in ["kvHeadNum", "numKeyValueHeads"]:
                case_config.inputs[12].range_values = kv_head_num

        # 14: scaleValue (如果有)
        if len(case_config.inputs) > 14:
             scale_val = case_config.inputs[14].range_values
             if scale_val == 0.0 or scale_val is None:
                 case_config.inputs[14].range_values = 1.0 / math.sqrt(head_dim)

        # 15: innerPrecise
        if len(case_config.inputs) > 15:
            case_config.inputs[15].range_values = scenario["inner_precise"]

        # 17: output (通常与 Query 维度一致)
        if len(case_config.inputs) > 17:
            case_config.inputs[17].shape = [total_tokens_q, q_head_num, head_dim]
            case_config.inputs[17].dtype = dtype

        return case_config