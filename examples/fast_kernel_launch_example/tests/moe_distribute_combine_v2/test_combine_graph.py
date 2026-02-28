import os
from typing import List, Optional, Dict, Any, Tuple
from torch.multiprocessing import Process, Manager, set_start_method
import torch
import torch_npu
import torch.distributed as dist
import torchair
from torchair.configs.compiler_config import CompilerConfig
import ascend_ops
from ascend_ops import MoeDistributeBuffer

# Constants
MASTER_PORT = 50001

class Config:
    """Configuration for MoE distributed training."""
    quant_mode: int = 0
    comm_quant_mode: int = 0
    dispatch_comm_alg: str = "fullmesh_v1"
    aic_core_num: int = 24
    aiv_core_num: int = 48
    rank_per_dev: int = 16
    shared_expert_num: int = 0
    rank_num_per_shared_expert: int = 0
    local_moe_expert_num: int = 8
    expert_shard_type: int = 0
    master_ip: str = "[::1]"
    is_2d_mask: int = 0
    is_mask: int = 0
    batch_size: int = 8
    hidden_size: int = 7168
    topk: int = 8
    out_dtype: int = 0
    random_seed: int = 0
    tp_world_size: int = 1
    
    def __init__(self):
        """Initialize configuration and compute derived values."""
        self.world_size = self.rank_per_dev
        self.ep_world_size = self.world_size
        self.shared_expert_rank_num = self.shared_expert_num * self.rank_num_per_shared_expert
        self.moe_rank_num = self.ep_world_size - self.shared_expert_rank_num
        self.moe_expert_num = self.moe_rank_num * self.local_moe_expert_num
        self.global_batch_size = self.batch_size * self.ep_world_size
        self.input_dtype = torch.bfloat16 if self.out_dtype == 0 else torch.float16
        self.is_topk_balance = self.dispatch_comm_alg == "fullmesh_v2"


cfg = Config()

buff_size = MoeDistributeBuffer.get_ccl_buffer_size(cfg.world_size, cfg.batch_size, cfg.hidden_size, cfg.moe_expert_num,
                                                    cfg.topk, cfg.shared_expert_num, cfg.shared_expert_rank_num,
                                                    cfg.dispatch_comm_alg)
buff_size_str = f'{buff_size}'
os.environ['HCCL_BUFFSIZE'] = buff_size_str


class MOECascadeModel(torch.nn.Module):
    """MoE Distribute and Combine Cascade Graph Model."""
    def __init__(self, group_ep, ccl_buffersize):
        super().__init__()
        print("Init group_ep")
        self.group_ep = group_ep
        self.distribute_buffer = MoeDistributeBuffer(group_ep, ccl_buffersize)
    
    def forward(self, **kwargs) -> torch.Tensor:
        with torchair.scope.limit_core_num(cfg.aic_core_num, cfg.aiv_core_num):
            # Dispatch
            dispatch_fn = self.distribute_buffer.npu_moe_distribute_dispatch_v2
            out = dispatch_fn(
                x=kwargs['x'], expert_ids=kwargs['expert_ids'],
                x_active_mask=kwargs['x_active_mask'],
                expert_shard_type=cfg.expert_shard_type,
                shared_expert_num=cfg.shared_expert_num,
                shared_expert_rank_num=cfg.shared_expert_rank_num,
                moe_expert_num=cfg.moe_expert_num,
                quant_mode=cfg.quant_mode, global_bs=cfg.global_batch_size,
                expert_scales=kwargs['expert_scales'],
                expert_token_nums_type=1, comm_alg=cfg.dispatch_comm_alg,
            )
            
            expand_x = out[0]
            if expand_x.dtype == torch.int8:
                expand_x = expand_x.to(cfg.input_dtype)
            
            # Combine
            combine_fn = self.distribute_buffer.npu_moe_distribute_combine_v2
            return combine_fn(
                expand_x=expand_x, expert_ids=kwargs['expert_ids'],
                assist_info_for_combine=out[2],
                ep_send_counts=out[4],
                expert_scales=kwargs['expert_scales'],
                x_active_mask=kwargs['x_active_mask'],
                ori_x=None,
                expert_shard_type=cfg.expert_shard_type,
                shared_expert_num=cfg.shared_expert_num,
                shared_expert_rank_num=cfg.shared_expert_rank_num,
                moe_expert_num=cfg.moe_expert_num,
                comm_quant_mode=cfg.comm_quant_mode,
                global_bs=cfg.global_batch_size,
                comm_alg="",
            )


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Compile model for NPU backend."""
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
    print("is_aclgraph=1, Test scenario: aclgraph graph mode")
    return torch.compile(model, backend=npu_backend)


def set_device(rank: int) -> None:
    """Set NPU device for given rank."""
    torch_npu.npu.set_device(rank % cfg.rank_per_dev)
    print(f"Current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank: int, ep_ranks: List[List[int]], tp_ranks: List[List[int]]) -> Tuple:
    """Initialize HCCL communication groups."""
    print(f"[INFO] device_{rank} Creating HCCL communication link")
    dist.init_process_group(
        backend="hccl", rank=rank, world_size=cfg.world_size,
        init_method=f"tcp://{cfg.master_ip}:{MASTER_PORT}",
    )
    
    ep_group = tp_group = None
    for ranks in ep_ranks:
        grp = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            ep_group = grp
    
    for ranks in tp_ranks:
        grp = dist.new_group(backend="hccl", ranks=ranks)
        if rank in ranks:
            tp_group = grp
    
    ep_hcomm = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm = tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return ep_hcomm, tp_hcomm, ep_group, tp_group


def create_rank_groups() -> Tuple[List[List[int]], List[List[int]]]:
    """Create EP and TP rank groups."""
    ep_ranks = [list(range(i, cfg.world_size, cfg.tp_world_size)) for i in range(cfg.tp_world_size)]
    tp_ranks = [list(range(i * cfg.tp_world_size, (i + 1) * cfg.tp_world_size)) for i in range(cfg.ep_world_size)]
    return ep_ranks, tp_ranks


def build_model_kwargs(x, expert_ids, x_active_mask, group_ep, group_tp, ep_rank_id, tp_rank_id, expert_scales):
    """Build keyword arguments for model forward."""
    mask = x_active_mask.to(torch.bool).npu() if (cfg.is_mask or cfg.is_2d_mask) and x_active_mask is not None else None
    return {
        'x': x.to(cfg.input_dtype).npu(),
        'expert_ids': expert_ids.to(torch.int32).npu(),
        'x_active_mask': mask,
        'group_ep': group_ep, 'group_tp': group_tp,
        'ep_rank_id': ep_rank_id, 'tp_rank_id': tp_rank_id,
        'expert_scales': expert_scales.to(torch.float32).npu(),
    }


def run_cascade_npu(queue, rank, x, expert_ids, x_active_mask, expert_scales, ori_x):
    """Run MoE cascade graph on NPU."""
    set_device(rank)
    ep_ranks, tp_ranks = create_rank_groups()
    ep_hcomm, tp_hcomm, ep_group, _ = init_hccl_comm(rank, ep_ranks, tp_ranks)
    
    kwargs = build_model_kwargs(x, expert_ids, x_active_mask, ep_hcomm, tp_hcomm,
                                 rank // cfg.tp_world_size, rank % cfg.tp_world_size, expert_scales)
    
    model = compile_model(MOECascadeModel(ep_group, buff_size))
    _ = model(**kwargs)
    torch.npu.synchronize()
    print(f"rank {rank} epid {rank // cfg.tp_world_size} tpid {rank % cfg.tp_world_size} npu finished!\n")


def parse_rank_input(rank: int, server_kwargs: Dict) -> Dict:
    """Parse input data for specific rank."""
    ep_id, tp_id = rank // cfg.tp_world_size, rank % cfg.tp_world_size
    mask = server_kwargs["x_active_mask_list"][tp_id][ep_id]
    return {
        "queue": server_kwargs["result_queue"], "rank": rank,
        "expert_scales": server_kwargs["expert_scales_list"][tp_id][ep_id],
        "x": server_kwargs["x_list"][tp_id][ep_id],
        "expert_ids": server_kwargs["expert_ids_list"][tp_id][ep_id],
        "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
        "ori_x": server_kwargs["ori_x_list"][tp_id][ep_id],
    }


def run_npu_processes(target_func, server_kwargs: Dict) -> None:
    """Launch NPU processes for distributed execution."""
    rank_list = list(range(cfg.world_size))
    print(f"Rank list is: {rank_list}")
    
    manager = Manager()
    server_kwargs["result_queue"] = manager.Queue()
    set_start_method("forkserver", force=True)
    
    procs = []
    for rank in rank_list:
        proc = Process(target=target_func, kwargs=parse_rank_input(rank, server_kwargs))
        proc.start()
        procs.append(proc)
    
    for proc in procs:
        proc.join()


def get_available_ranks() -> Tuple[List[int], int]:
    """Generate available rank list."""
    shared = list(range(cfg.shared_expert_rank_num))
    moe = list(range(cfg.shared_expert_rank_num, cfg.world_size))
    print(f"available_list={shared + moe}")
    return list(range(cfg.world_size)), 1000


def gen_x() -> List[torch.Tensor]:
    """Generate input tensors."""
    return [torch.arange(1, cfg.global_batch_size + 1, dtype=cfg.input_dtype).view(-1, 1).repeat(1, cfg.hidden_size)
            for _ in range(cfg.tp_world_size)]


def gen_expert_ids() -> List[torch.Tensor]:
    """Generate expert assignment IDs."""
    def gen_random(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]
    
    def gen_balanced(round_idx):
        return torch.arange(round_idx, round_idx + cfg.moe_expert_num, dtype=torch.int32) % cfg.moe_expert_num
    
    result = []
    for _ in range(cfg.tp_world_size):
        if cfg.is_topk_balance:
            num_rounds = (cfg.global_batch_size * cfg.topk) // cfg.moe_expert_num + 1
            ids = [gen_balanced(i) for i in range(num_rounds)]
        else:
            ids = [gen_random(cfg.topk, 0, cfg.moe_expert_num) for _ in range(cfg.global_batch_size)]
        result.append(torch.cat(ids)[:cfg.global_batch_size * cfg.topk].view([cfg.global_batch_size, cfg.topk]))
    return result


def gen_x_active_mask() -> List[torch.Tensor]:
    """Generate active token masks."""
    def gen_card_mask():
        real_num = 0 if cfg.is_mask else cfg.batch_size
        return torch.cat((torch.ones(real_num), torch.zeros(cfg.batch_size - real_num))).to(torch.bool)
    
    result = []
    for _ in range(cfg.tp_world_size):
        masks = [gen_card_mask() for _ in range(cfg.ep_world_size)]
        if cfg.is_2d_mask:
            for j in range(cfg.ep_world_size):
                out = torch.zeros(cfg.batch_size, cfg.topk, dtype=torch.bool)
                idx = masks[j].nonzero().squeeze(-1)
                if idx.numel() > 0:
                    out[idx] = torch.rand(len(idx), cfg.topk) > 0.5
                masks[j] = out
        result.append(torch.cat(masks))
    return result


def gen_expert_scales() -> List[torch.Tensor]:
    """Generate expert scaling factors."""
    return [torch.empty([cfg.global_batch_size, cfg.topk], dtype=torch.float32).uniform_(-1, 1)
            for _ in range(cfg.tp_world_size)]


if __name__ == "__main__":
    torch.manual_seed(cfg.random_seed)
    
    available_ranks, _ = get_available_ranks()
    x_list = gen_x()
    expert_ids_list = gen_expert_ids()
    x_active_mask_list = gen_x_active_mask()
    expert_scales_list = gen_expert_scales()
    
    run_npu_processes(run_cascade_npu, {
        "x_list": [list(t.chunk(cfg.ep_world_size)) for t in x_list],
        "expert_ids_list": [list(t.chunk(cfg.ep_world_size)) for t in expert_ids_list],
        "x_active_mask_list": [list(t.chunk(cfg.ep_world_size)) for t in x_active_mask_list],
        "expert_scales_list": [list(t.chunk(cfg.ep_world_size)) for t in expert_scales_list],
        "ori_x_list": [list(t.chunk(cfg.ep_world_size)) for t in x_list],
    })

    print("run npu success.")