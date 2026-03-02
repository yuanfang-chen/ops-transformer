import random
from typing import Union, List
import numpy as np
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig

@GENERATOR_REGISTRY.register("generate_rain_fusion_attention")
class RainFusionAttentionGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.input_layout = "TND"
        self.input_layout_range = ["TND", "BNSD"]
        self.q_input_layout = "TND"
        self.kv_input_layout = "TND"
        self.dtype = "fp16"
        self.maxQSeqlen = 0
        self.maxKvSeqlen = 0
        self.query_range = None
        self.query_total = 0
        self.kv_total = 0
        self.head_num = 0
        self.kv_num = 0
        self.head_dim = 0
        self.headDim_range = [64, 128]
        self.batch = 0

    def find_max_factor_not_exceeding_b(self, a, b):
        if a <= 0 or b <= 0:
            return 1
        upper_limit = min(a, b)
        for x in range(upper_limit, 0, -1):
            if a % x == 0:
                return x
        return 1

    def after_input_config(
            self,
            index: int,
            input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        '''
        当参数之间有相互依赖关系时，需要覆写此函数来约束生成的输入参数信息
        :param index: 用例参数列表的下标，0：表示第0个参数
        :param input_case: 随机生成的用例信息
        :return: 修改后需要返回的输入信息对象
        '''
        # self.input_layout = random.choice(self.input_layout_range)
        # self.q_input_layout = self.input_layout
        # self.kv_input_layout = self.input_layout
        
        if index == 0 and input_case.name == "query":
            self.query_total = input_case.shape[0] # T
            self.head_num = input_case.shape[1] # N
            self.head_dim = random.choice(self.headDim_range) # D
            self.query_range = input_case.range_values
            self.dtype = input_case.dtype

        if index == 1 and input_case.name == "key":
            self.kv_total = input_case.shape[0]
            input_case.dtype = self.dtype
            input_case.range_values = self.query_range
            self.kv_num = input_case.shape[1]
            self.kv_num = self.find_max_factor_not_exceeding_b(self.head_num, self.kv_num)

        if index == 2 and input_case.name == "value":
            input_case.dtype = self.dtype
            input_case.range_values = self.query_range

        if index == 5:
            block_size = input_case[0].range_values
            if block_size == 0:
                attr_info.range_values = random.randint(1, 1024)

        if index == 7:
            self.batch = len(input_case)
            for attr_info in input_case:
                attr_info.range_values = random.randint(1, self.query_total)
                attr_info.dtype = 'int'

        if index == 8:
            numbers = len(input_case)
            if numbers >= self.batch:
                del input_case[self.batch:]
            else:
                random_case = self.gens[index].get_scalar_by_number(self.batch - numbers)
                input_case.extend(random_case)
            for attr_info in input_case:
                attr_info.range_values = random.randint(1, self.kv_total)
                attr_info.dtype = 'int'
        if index == 10:
            self.input_layout = input_case.range_values
            self.q_input_layout = self.input_layout
        if index == 11:
            input_case.range_values = self.input_layout
        if index == 15:
            if self.dtype == 'bf16':
                input_case.range_values = 0

        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        query_total = 0
        kv_total = 0
        blocks_total = 0
        max_kv_block_num = 0
        x_block = case_config.inputs[5][0].range_values
        y_block = case_config.inputs[5][1].range_values
        if isinstance(x_block, list) and len(x_block) >= 2:
            x_block = random.randint(x_block[0], x_block[1])
        if isinstance(y_block, list) and len(y_block) >= 2:
            y_block = random.randint(y_block[0], y_block[1])
        x_block = max(1, x_block)
        y_block = ((y_block + 127) // 128) * 128
        for b in range(self.batch):
            q_seqlen = case_config.inputs[7][b].range_values
            kv_seqlen = case_config.inputs[8][b].range_values
            self.maxQSeqlen = max(q_seqlen, self.maxQSeqlen)
            self.maxKvSeqlen = max(kv_seqlen, self.maxKvSeqlen)
            query_total += q_seqlen
            kv_total += kv_seqlen
            s_block_num_q = (q_seqlen + x_block - 1) // x_block
            s_block_num_kv = (kv_seqlen + y_block - 1) // y_block

            blocks_total += s_block_num_q
            max_kv_block_num = max(max_kv_block_num, s_block_num_kv)
        scale = case_config.inputs[14].range_values
        if scale == 0.0:
            scale == 1.0 / np.sqrt(self.head_dim)

        if self.q_input_layout == "TND":
            case_config.inputs[0].shape = [query_total, self.head_num, self.head_dim]
            case_config.inputs[1].shape = [kv_total, self.kv_num, self.head_dim]
            case_config.inputs[2].shape = [kv_total, self.kv_num, self.head_dim]
            case_config.inputs[17].shape = [query_total, self.head_num, self.head_dim]
            case_config.inputs[17].dtype = self.dtype
            case_config.inputs[18].shape = [query_total, self.head_num, 1]
            case_config.inputs[18].dtype = 'fp32'
        elif self.q_input_layout == "BNSD":
            case_config.inputs[0].shape = [self.batch, self.head_num, self.maxQSeqlen, self.head_dim]
            case_config.inputs[1].shape = [self.batch, self.kv_num, self.maxKvSeqlen, self.head_dim]
            case_config.inputs[2].shape = [self.batch, self.kv_num, self.maxKvSeqlen, self.head_dim]
            case_config.inputs[17].shape = [self.batch, self.head_num, self.maxQSeqlen, self.head_dim]
            case_config.inputs[17].dtype = self.dtype
            case_config.inputs[18].shape = [self.batch, self.head_num, self.maxQSeqlen, 1]
            case_config.inputs[18].dtype = 'fp32'
        case_config.inputs[3].shape = [blocks_total, self.head_num, max_kv_block_num]
        case_config.inputs[4].shape = [blocks_total, self.head_num]
        case_config.inputs[5][0].range_values = x_block
        case_config.inputs[5][1].range_values = y_block
        case_config.inputs[12].range_values = self.kv_num
        return case_config
