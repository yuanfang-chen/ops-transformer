# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torch.distributed as dist

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from ascend910b_aclnnMoeDistributeDispatchGolden import MoeDistributeDispatchGolden
import logging

# region 图模式相关类和函数
def compile_model(model, graph_type, is_full_core=True):
    if graph_type == 0:
        logging.debug("graph_type=0，测试场景：单算子模式")
        return model

    from torchair.configs.compiler_config import CompilerConfig
    if is_full_core:
        compiler_config = CompilerConfig()
        compiler_config.debug.aclgraph.clone_input.value = False
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
    torch._dynamo.reset()
    if graph_type == 1:
        # 传统入图模式，静态shape
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
        logging.debug("graph_type=1, 测试场景: 传统入图模式，静态shape")
    return compiled_model

class MOE_DISTRIBUTE_DISPATCH_MODEL(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        group_ep: str,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        *,
        scales: torch.Tensor = None,
        expert_scales: torch.Tensor = None,
        group_tp: str = "",
        quant_mode: int = 0,
        global_bs: int = 0,
        expert_token_nums_type: int = 1
    ):
        output = torch_npu.npu_moe_distribute_dispatch(
            x=x,
            expert_ids=expert_ids,
            scales=scales,
            x_active_mask=x_active_mask,
            expert_scales=expert_scales,
            group_ep=group_ep,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            group_tp=group_tp,
            quant_mode=quant_mode,
            global_bs=global_bs,
            expert_token_nums_type=expert_token_nums_type)
        return output

@register("execute_MoeDistributeDispatch")
class MoeDistributeDispatch(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MoeDistributeDispatch, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.group_ep = None
        self.model = MOE_DISTRIBUTE_DISPATCHV2_MODEL()
        self.compiled_model = compile_model(MOE_DISTRIBUTE_DISPATCHV2_MODEL(), graph_type=1)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        x = input_data.kwargs['x'][rank_id]
        expert_ids = input_data.kwargs['expertIds'][rank_id]
        expert_scales = input_data.kwargs['expertScales'][rank_id]
        ep_world_size = world_size
        ep_rank_id = rank_id
        moe_expert_num = input_data.kwargs['moeExpertNum']
        quant_mode = input_data.kwargs['quantMode']
        global_bs = input_data.kwargs['globalBs']
        expert_token_nums_type = input_data.kwargs['expertTokenNumsType']
        graph_type = input_data.kwargs['graphType']
        scales = input_data.kwargs['scales']
        if quant_mode == 0:
            scales = None
        if self.name == "cpu" or self.dist_task_info.is_bm:
            bs_list = input_data.kwargs['bsList'].cpu()
            x_world = input_data.kwargs['xWorld'].cpu()
            expert_ids_world = input_data.kwargs['expertIdsWorld']
            expert_scales_world = input_data.kwargs['expertScalesWorld']
            if self.name == "cpu":
                bs_list = bs_list.cpu()
                x_world = x_world.cpu()
                expert_ids_world = expert_ids_world.cpu()
                expert_scales_world = expert_scales_world.cpu()
                if scales is not None:
                    scales = scales.cpu()
            golden_generator = MoeDistributeDispatchGolden(bs_list, x_world, expert_ids_world,
                moe_expert_num, ep_world_size, quant_mode, expert_token_nums_type, scales=scales, expert_scales_world=expert_scales_world)
            # 真值—— is_golden
            is_golden = False if self.dist_task_info.is_bm else True
            expand_x_world, dynamic_scales_world, expand_idx_world, expert_token_nums_world, ep_send_counts_world, tp_recv_counts_world, expand_scales_world = golden_generator.process(is_golden=is_golden)
            return expand_x_world[ep_rank_id], dynamic_scales_world[ep_rank_id].view(-1), expand_idx_world[ep_rank_id].view(-1), expert_token_nums_world[ep_rank_id].view(-1), ep_send_counts_world[ep_rank_id].view(-1), tp_recv_counts_world[ep_rank_id], expand_scales_world[ep_rank_id].view(-1), input_data.kwargs['bsList'][ep_rank_id]

        if self.group_ep is None:
            self.group_ep = self.get_hcomm_info(rank_id)
            logging.debug(f"[rank:{rank_id}] group_ep: {self.group_ep}.")
        group_ep = self.group_ep
        if graph_type == 1:
            output = self.compiled_model(
                x=x, expert_ids=expert_ids, group_ep=group_ep, ep_world_size=ep_world_size, ep_rank_id=ep_rank_id, moe_expert_num=moe_expert_num,
                expert_scales=expert_scales, group_tp=group_ep,
                quant_mode=quant_mode, global_bs=global_bs,
                scales=scales,
                expert_token_nums_type=expert_token_nums_type
            )
        elif graph_type == 0:
            output = self.model(
                x=x, expert_ids=expert_ids, group_ep=group_ep, ep_world_size=ep_world_size, ep_rank_id=ep_rank_id, moe_expert_num=moe_expert_num,
                expert_scales=expert_scales,
                scales=scales,
                quant_mode=quant_mode, global_bs=global_bs,
                expert_token_nums_type=expert_token_nums_type
            )
        else:
            logging.error(f"[rank:{rank_id}] Unsupport graph_type:{graph_type}.")
            output = (None,)
        logging.debug(f"[rank:{rank_id}] finished.")
        return output + (input_data.kwargs['bsList'][ep_rank_id],)

    def get_hcomm_info(self, rank_id: int):
        rank_num_per_server = 8
        torch_npu.npu.set_device(rank_id % rank_num_per_server)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank_id)
        return hcomm_info

    def print_info(self, input_data: InputDataset):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        for key in input_data.kwargs.keys():
            data = input_data.kwargs[key]
            if torch.is_tensor(data):
                logging.debug(f"[rank:{rank_id}/{world_size}] {key}: {data.shape}.")
            else:
                logging.debug(f"[rank:{rank_id}/{world_size}] {key}: {data}.")

    def init_by_input_data(self, input_data: InputDataset):
        """
        :param input_data:
        :return:
        to modify input data
        """
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size

        if input_data.kwargs['bsList'].shape[0] >= world_size:
            bs_list = input_data.kwargs['bsList'][:world_size]
        else:
            bs_list = input_data.kwargs['bsList'][0].repeat(world_size)[:world_size]
        input_data.kwargs['bsList'] = bs_list
        global_bs = bs_list.sum().item()
        input_data.kwargs['globalBs'] = global_bs

        x_world = input_data.kwargs['x'][:global_bs]
        device = input_data.kwargs['expertIds'].device
        _, k = input_data.kwargs['expertIds'].shape
        moe_expert_num = input_data.kwargs['moeExpertNum']
        total_expert_num = moe_expert_num
        # 1. 保存 torch 的原始随机状态
        torch_rng_state = torch.get_rng_state()
        # 2. 从配置中获取一个确定性的种子
        seed = self.task_result.case_config.id
        # 3. 为 torch 设置种子
        torch.manual_seed(seed)
        expert_ids_world = torch.argsort(torch.rand(global_bs, total_expert_num), dim=1).to(torch.int32)[:, :k].to(device)
        expert_scales_world = input_data.kwargs['expertScalesOptional']
        # 4. 恢复 torch 的原始随机状态
        torch.set_rng_state(torch_rng_state)
        self.bs_list = bs_list.cpu()
        self.x_world = x_world.cpu()
        self.expert_ids_world = expert_ids_world.cpu()
        self.expert_scales_world = expert_scales_world
        self.scales = None
        if self.expert_scales_world is not None:
            self.expert_scales_world = self.expert_scales_world[:global_bs]

        start_idx = bs_list[:rank_id].sum()
        end_idx = start_idx + bs_list[rank_id]
        input_data.kwargs['x'] = x_world[start_idx : end_idx]
        input_data.kwargs['xWorld'] = x_world
        input_data.kwargs['expertIds'] = expert_ids_world[start_idx : end_idx]
        input_data.kwargs['expertIdsWorld'] = expert_ids_world
        input_data.kwargs['expertScalesOptional'] = self.expert_scales_world[start_idx : end_idx]
        input_data.kwargs['expertScalesWorld'] = self.expert_scales_world