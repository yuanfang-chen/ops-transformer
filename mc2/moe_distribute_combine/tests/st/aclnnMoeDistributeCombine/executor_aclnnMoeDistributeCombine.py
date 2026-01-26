import os
import torch
import torch_npu
import torch.distributed as dist

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import sys
sys.path.append('../../../moe_distribute_dispatch/tests/st/aclnnMoeDistributeDispatch')

from ascend910b_aclnnMoeDistributeDispatchGolden import MoeDistributeDispatchGolden
from ascend910b_aclnnMoeDistributeCombineGolden import MoeDistributeCombineGolden
import logging

class MOE_DISTRIBUTE_COMBINE_MODEL(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        expand_x: torch.Tensor,
        expert_ids: torch.Tensor,
        expand_idx: torch.Tensor,
        ep_send_counts: torch.Tensor,
        expert_scales: torch.Tensor,
        group_ep: str,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        *,
        expand_scales: torch.Tensor = None,
        group_tp: str = "",
        global_bs: int = 0,
        comm_quant_mode: int = 0,
    ):
        x = torch_npu.npu_moe_distribute_combine(
            expand_x=expand_x,
            expert_ids=expert_ids,
            expand_idx=expand_idx,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            expand_scales=expand_scales,
            group_ep=group_ep,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            group_tp=group_tp,
            global_bs=global_bs,
            comm_quant_mode=comm_quant_mode)
        return x

def compile_model(model, graph_type, is_full_core=True):
    if graph_type == 0:
        logging.debug("graph_type=0，测试场景：单算子模式")
        return model

    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.debug.aclgraph.clone_input.value = False
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
    torch._dynamo.reset()
    if graph_type == 1:
        # 传统入图模式，静态shape
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
        logging.debug("graph_type=1, 测试场景: 传统入图模式，静态shape")
    return compiled_model

@register("execute_MoeDistributeCombine")
class MoeDistributeCombine(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MoeDistributeCombine, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.group_ep = None
        self.model = MOE_DISTRIBUTE_COMBINE_MODEL()
        self.compiled_model = compile_model(MOE_DISTRIBUTE_COMBINE_MODEL(), graph_type=1)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        comm_quant_mode = input_data.kwargs['commQuantMode'] if self._is_layered() else 0
        graph_type = input_data.kwargs['graphType']
        ep_world_size = world_size
        ep_rank_id = rank_id

        if self.name == "cpu" or self.dist_task_info.is_bm:
            x = input_data.kwargs['x']
            expert_scales = input_data.kwargs['expertScales']
            expert_ids = input_data.kwargs['expertIds']
            if self.name == "cpu":
                x = x.cpu()
                expert_scales = expert_scales.cpu()
                expert_ids = expert_ids.cpu()
            golden_generator = MoeDistributeCombineGolden(x,
                expert_scales, expert_ids,
                input_data.kwargs['moeExpertNum'], ep_world_size, comm_quant_mode=comm_quant_mode)
            # 真值—— is_golden
            is_golden = False if self.dist_task_info.is_bm else True
            return golden_generator.process(is_golden)

        if self.group_ep is None:
            self.group_ep = self.get_hcomm_info(rank_id)
            logging.debug(f"[rank:{rank_id}/{world_size}] group_ep: {self.group_ep}.")
        group_ep = self.group_ep
        if graph_type == 1:
            output = self.compiled_model(
                expand_x=input_data.kwargs['expandX'][ep_rank_id], expert_ids=input_data.kwargs['expertIds'][ep_rank_id], 
                expand_idx=input_data.kwargs['expandIdx'][ep_rank_id], ep_send_counts=input_data.kwargs['epSendCounts'][ep_rank_id],
                expert_scales=input_data.kwargs['expertScales'][ep_rank_id],
                group_ep=group_ep, ep_world_size=ep_world_size, ep_rank_id=ep_rank_id, moe_expert_num=input_data.kwargs['moeExpertNum'],
                expand_scales=input_data.kwargs['expandScales'][ep_rank_id], group_tp=group_ep,
                comm_quant_mode=comm_quant_mode, global_bs=input_data.kwargs['globalBs'],
            )
        elif graph_type == 0:
            output = self.model(
                expand_x=input_data.kwargs['expandX'][ep_rank_id], expert_ids=input_data.kwargs['expertIds'][ep_rank_id], 
                expand_idx=input_data.kwargs['expandIdx'][ep_rank_id], ep_send_counts=input_data.kwargs['epSendCounts'][ep_rank_id],
                expert_scales=input_data.kwargs['expertScales'][ep_rank_id],
                group_ep=group_ep, ep_world_size=ep_world_size, ep_rank_id=ep_rank_id, moe_expert_num=input_data.kwargs['moeExpertNum'],
                expand_scales=input_data.kwargs['expandScales'][ep_rank_id],
                comm_quant_mode=comm_quant_mode, global_bs=input_data.kwargs['globalBs'],
            )
        else:
            logging.error(f"[rank:{rank_id}] Unsupport graph_type:{graph_type}.")
        return output

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

    def _is_layered(self):
        is_layered = False
        if (os.getenv('HCCL_INTRA_PCIE_ENABLE') == "1" and os.getenv('HCCL_INTRA_ROCE_ENABLE') == "0"):
            is_layered = True
        return is_layered

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
        expert_scales_world = input_data.kwargs['expertScales']
        # 4. 恢复 torch 的原始随机状态
        torch.set_rng_state(torch_rng_state)
        self.bs_list = bs_list.cpu()
        self.x_world = x_world.cpu()
        self.expert_ids_world = expert_ids_world.cpu()
        self.expert_scales_world = expert_scales_world[:global_bs].cpu()
        input_data.kwargs['expertScalesWorld'] = expert_scales_world

        input_generator = MoeDistributeDispatchGolden(self.bs_list,
            self.x_world, self.expert_ids_world,
            moe_expert_num, world_size, expert_scales_world=self.expert_scales_world)
        expand_x_world, expand_idx_world, ep_send_counts_world, tp_recv_counts_world, expand_scales_world = input_generator.process_for_combine()
        start_idx = bs_list[:rank_id].sum()
        end_idx = start_idx + bs_list[rank_id]
        input_data.kwargs['x'] = x_world[start_idx : end_idx].to(x_world.device)
        input_data.kwargs['expandX'] = expand_x_world[rank_id].to(x_world.device)
        input_data.kwargs['expertIds'] = expert_ids_world[start_idx : end_idx]
        input_data.kwargs['expertIdsWorld'] = expert_ids_world
        input_data.kwargs['expertScales'] = expert_scales_world[start_idx : end_idx]
        input_data.kwargs['expertScalesWorld'] = expert_scales_world
        input_data.kwargs['expandIdx'] = expand_idx_world[rank_id].to(x_world.device)
        input_data.kwargs['epSendCounts'] = ep_send_counts_world[rank_id].to(x_world.device)
        input_data.kwargs['expandScalesOptional'] = expand_scales_world[rank_id].to(x_world.device)