import random
import math
import copy
from typing import Union, List
# 保持原有的 ATK 引用
try:
    import numpy as np
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.configs.case_config import InputCaseConfig, CaseConfig
except ImportError:
    # 仅为了在无 ATK 环境下不报错，实际运行环境应包含上述包
    pass

@GENERATOR_REGISTRY.register("generate_rain_fusion_attention")
class RainFusionAttentionGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.q_input_layout = "TND"
        self.kv_input_layout = "TND"
        self.dtype = "fp16"
        self.headDim_range = [64, 128] 

    def get_boundary_kv_len(self, q_len: int, max_limit: int = 2048) -> int:
        """
        构造针对 Tiling 边界的 KV 长度
        规则: KV - Q = BlockSize * n ± 1
        """
        block_size = 128
        
        # 确保 max_limit 至少不小于 q_len
        max_limit = max(q_len, max_limit)
        
        if q_len >= max_limit:
            return max_limit

        max_n = (max_limit - q_len) // block_size
        
        if max_n < 1: 
            return q_len + 1 
            
        n = random.randint(1, min(4, max_n)) 
        offset = random.choice([-1, 1])
        
        diff = block_size * n + offset
        kv_len = q_len + diff
        
        if kv_len > max_limit: kv_len = max_limit
        if kv_len < q_len: kv_len = q_len 
        
        return int(kv_len)

    def _get_valid_kv_head_num(self, q_head_num):
        """
        根据约束计算满足条件的 kv_head_num
        约束: 
        1. N1 >= N2 && N1 % N2 == 0
        2. G = N1 / N2, G < 128 && 128 % G == 0
        """
        valid_kv_nums = []
        # 满足 128 % G == 0 且 G < 128 的 Group Size
        valid_groups = [1, 2, 4, 8, 16, 32, 64]
        
        for g in valid_groups:
            if q_head_num % g == 0:
                n2 = q_head_num // g
                if n2 > 0:
                    valid_kv_nums.append(n2)
        
        if not valid_kv_nums:
            return None # 此时说明当前的 q_head_num 无法满足约束
            
        return random.choice(valid_kv_nums)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # ==========================================
        # 1. 基础参数与 HeadNum 约束修复
        # ==========================================
        q_input = case_config.inputs[0]
        
        # 重新随机 Head Dim
        head_dim = random.choice(self.headDim_range)
        q_head_num = q_input.shape[1]
        
        # 计算合法的 KV Head Num (N2)
        kv_head_num = self._get_valid_kv_head_num(q_head_num)
        
        # 【关键修复】如果当前的 q_head_num 找不到合法的 N2 (例如 q_head_num 是奇怪的质数)
        # 强制将 q_head_num 重置为一个好用的值 (如 32)，确保 G 约束满足
        if kv_head_num is None:
            q_head_num = 32
            q_input.shape[1] = q_head_num
            kv_head_num = self._get_valid_kv_head_num(q_head_num) # 此时一定有值
        
        # 更新 Query, Key, Value 的 Shape 和属性
        # Index 0: Query
        q_input.shape[2] = head_dim
        
        # Index 1, 2: Key, Value
        for idx in [1, 2]:
            case_config.inputs[idx].shape[1] = kv_head_num
            case_config.inputs[idx].shape[2] = head_dim
            case_config.inputs[idx].range_values = q_input.range_values
            case_config.inputs[idx].dtype = q_input.dtype

        # 更新 kvHeadNum / numKeyValueHeads (Index 12)
        if len(case_config.inputs) > 12:
            input_name = case_config.inputs[12].name
            if input_name in ["kvHeadNum", "numKeyValueHeads"]:
                case_config.inputs[12].range_values = kv_head_num

        # ==========================================
        # 2. Block Shape 约束
        # ==========================================
        # 假设 inputs[5] 是包含两个对象的列表 [x_obj, y_obj]
        x_block_obj = case_config.inputs[5][0]
        y_block_obj = case_config.inputs[5][1]
        
        bs_x = random.choice([64, 128])
        bs_y = random.choice([64, 128, 256])
        # 约束：保持你原有的逻辑，BS_Y 按 128 对齐
        bs_y = ((bs_y + 127) // 128) * 128
        
        x_block_obj.range_values = bs_x
        y_block_obj.range_values = bs_y
        
        x_block = max(1, bs_x)
        y_block = bs_y

        # ==========================================
        # 3. 场景与 Batch/SeqLen 生成
        # ==========================================
        rand_prob = random.random()
        mode_tag = "Fast-2k"
        target_batch_size = 1
        target_total_tokens = 2048
        kv_max_limit = 4096 

        if rand_prob < 0.01: # Extreme
            mode_tag = "Extreme-16k"
            target_batch_size = 1
            target_total_tokens = 16384
            kv_max_limit = 32768 
        elif rand_prob < 0.31: # Large
            mode_tag = "Large-32k"
            target_batch_size = random.randint(1, 16) # 稍微调小最大Batch防止OOM
            target_total_tokens = random.randint(16000, 32000)
            kv_max_limit = 65536 
        else: # Fast
            mode_tag = "Fast-2k"
            target_batch_size = random.randint(1, 8)
            target_total_tokens = random.randint(1024, 2048)
            kv_max_limit = 4096

        # 处理 Sequence Lengths Lists (Input 7 & 8)
        # 假设 case_config.inputs[7] 是一个列表，存放每个batch的len对象
        q_seq_input_list = case_config.inputs[7]
        kv_seq_input_list = case_config.inputs[8]
        
        # 独立调整 Q List 长度
        current_q_batch = len(q_seq_input_list)
        if current_q_batch < target_batch_size:
            diff = target_batch_size - current_q_batch
            if current_q_batch > 0:
                base_item = q_seq_input_list[-1]
                for _ in range(diff):
                    q_seq_input_list.append(copy.deepcopy(base_item))
        elif current_q_batch > target_batch_size:
            del q_seq_input_list[target_batch_size:]

        # 独立调整 KV List 长度 (修复 IndexError)
        current_kv_batch = len(kv_seq_input_list)
        if current_kv_batch < target_batch_size:
            diff = target_batch_size - current_kv_batch
            if current_kv_batch > 0:
                base_item = kv_seq_input_list[-1]
                for _ in range(diff):
                    kv_seq_input_list.append(copy.deepcopy(base_item))
        elif current_kv_batch > target_batch_size:
            del kv_seq_input_list[target_batch_size:]

        # ==========================================
        # 4. T 维度 (Total Q Blocks) 计算
        # ==========================================
        query_total_tokens = 0
        kv_total_tokens = 0
        total_q_blocks = 0
        max_kv_block_num_global = 0
        
        avg_tokens_per_batch = max(64, target_total_tokens // target_batch_size)

        for b in range(target_batch_size):
            # Q 长度波动
            variation = int(avg_tokens_per_batch * 0.2)
            q_len = random.randint(max(64, avg_tokens_per_batch - variation), 
                                   avg_tokens_per_batch + variation)
            
            q_seq_input_list[b].range_values = q_len
            q_seq_input_list[b].dtype = 'int'
            
            # KV 长度生成
            if random.random() < 0.5:
                safe_limit = max(q_len + 128, kv_max_limit)
                kv_len = self.get_boundary_kv_len(q_len, max_limit=safe_limit)
            else:
                max_add = 512 if mode_tag == "Fast-2k" else 2048
                upper_bound = max(q_len, min(kv_max_limit, q_len + max_add))
                kv_len = random.randint(q_len, upper_bound)
            
            kv_seq_input_list[b].range_values = kv_len
            kv_seq_input_list[b].dtype = 'int'
            
            # 累加 Token 总数 (用于 TND 输入 Tensor)
            query_total_tokens += q_len
            kv_total_tokens += kv_len
            
            # 【核心修复】计算 Block 数 (T 必须是 ceil 之和)
            s_block_num_q = (q_len + x_block - 1) // x_block
            s_block_num_kv = (kv_len + y_block - 1) // y_block
            
            total_q_blocks += s_block_num_q
            max_kv_block_num_global = max(max_kv_block_num_global, s_block_num_kv)

        # ==========================================
        # 5. 更新输入 Shape 和 Range
        # ==========================================
        
        # 0: Query [Total, N, D]
        case_config.inputs[0].shape = [query_total_tokens, q_head_num, head_dim]
        
        # 1: Key [Total, N_kv, D]
        case_config.inputs[1].shape = [kv_total_tokens, kv_head_num, head_dim]
        
        # 2: Value [Total, N_kv, D]
        case_config.inputs[2].shape = [kv_total_tokens, kv_head_num, head_dim]
        
        # 3: selectIdx [Total_Q_Blocks, N, Max_KV_Blocks]
        max_kv_block_num_global = max(1, max_kv_block_num_global)
        case_config.inputs[3].shape = [total_q_blocks, q_head_num, max_kv_block_num_global]
        # range_values 设为有效范围，防止 Inspect Kwargs 越界
        # 注意：这里设的是全局最大范围，具体到每个 batch 仍需生成器内部保证不越界
        # 但至少 range 不能小于 -1
        case_config.inputs[3].range_values = [-1, max_kv_block_num_global - 1]
        
        # 4: selectNumIdx [Total_Q_Blocks, N]
        case_config.inputs[4].shape = [total_q_blocks, q_head_num]
        # 【关键修复】Inspect Kwargs 报错点：必须给 selectNumIdx 一个合法的范围
        # 它表示每个 Q block 有多少个有效的 KV block，范围是 [0, max_kv_block_num_global]
        case_config.inputs[4].range_values = [0, max_kv_block_num_global]

        # 15: innerPrecise
        if len(case_config.inputs) > 15:
            if q_input.dtype == 'bf16':
                case_config.inputs[15].range_values = 0
            else:
                case_config.inputs[15].range_values = random.choice([0, 1])

        # Scale (Index 14)
        if len(case_config.inputs) > 14:
             scale_val = case_config.inputs[14].range_values
             if scale_val == 0.0 or scale_val is None:
                 case_config.inputs[14].range_values = 1.0 / math.sqrt(head_dim)
        
        case_config.inputs[17].shape = [query_total_tokens, q_head_num, head_dim]
        case_config.inputs[17].dtype = q_input.dtype
        return case_config