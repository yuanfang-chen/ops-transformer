import math
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
try:
   import torch_npu
except ImportError:
   pass

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

@register("execute_aclnnMatmulReduceScatter")
class DistFunctionApi(BaseApi):
    def __init__(self, task_result):
        super(DistFunctionApi, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = int(self.dist_task_info.rank)
        world_size = self.dist_task_info.world_size
        input_chunk = input_data.kwargs['x1']
        weight_chunk = input_data.kwargs['x2']
        
        
        if weight_chunk.shape[0]!=input_chunk.shape[1] and weight_chunk.shape[1] == input_chunk.shape[1]:
            weight_chunk = weight_chunk.transpose(0, 1)

        if self.name == "cpu":
            input_chunk = input_chunk.cpu().to(torch.float32)
            weight_chunk = weight_chunk.cpu().to(torch.float32).contiguous()
            output = torch.matmul(input_chunk, weight_chunk)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            scatter_shape_m = input_chunk.shape[0] // world_size
            scatter_output = output.narrow(0, rank_id * scatter_shape_m, scatter_shape_m)
            return scatter_output

        if self.dist_task_info.is_bm:
            
            if input_chunk.shape == []:
                return torch.tensor([])

            tensor_scatter_shape = [input_chunk.shape[0] // world_size, weight_chunk.shape[1]]
            tensor_scatter = torch.zeros(tensor_scatter_shape, dtype=input_chunk.dtype).npu()
            output = torch.matmul(input_chunk, weight_chunk)
            dist._reduce_scatter_base(tensor_scatter, output, op=ReduceOp.SUM) 
            scatter_output = tensor_scatter
            return scatter_output
        else:
            if dist.is_available():
                from torch.distributed.distributed_c10d import _get_default_group
                default_pg = _get_default_group()
                if torch.__version__ > '2.0.1':
                    hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
                else:
                    hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            if self.task_result.case_config.id % 5 == 4 and 'bias' in input_data.kwargs:
                bias = input_data.kwargs['bias']
            else:
                bias = None
            output_npu = torch_npu.npu_mm_reduce_scatter_base(input_chunk, weight_chunk,
                                                      hcomm_info, world_size, reduce_op="sum", bias=bias)
            return output_npu

