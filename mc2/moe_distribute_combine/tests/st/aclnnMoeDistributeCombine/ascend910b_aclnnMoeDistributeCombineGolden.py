import os
import torch

def quant_int(d, group_size=-1, is_bf16=False):
    origin_shape = d.shape
    if not is_bf16:
        d = d.to(torch.float16)
    if group_size != -1:
        d = d.reshape(-1, group_size)
    d_max = d.abs().amax(-1, keepdim=True)
    if is_bf16:
        t2 = torch.tensor(1.0, dtype=torch.float32, device=d.device) / torch.tensor(127, dtype=torch.float32, device=d.device)
    else:
        t2 = torch.tensor(1.0, dtype=torch.float32, device=d.device) / torch.tensor(127, dtype=torch.float16, device=d.device)
    scale = d_max * t2
    d_q = ((d) / scale).nan_to_num_()
    if is_bf16:
        d_q = d_q.to(torch.float16).round().to(torch.int8)
        d_q = d_q.to(torch.float16).to(torch.float32)
        scale = scale.to(torch.bfloat16).to(torch.float32)
    else:
        d_q = ((d) / scale).round().to(torch.int8)
        d_q = d_q.to(torch.float16)
    return (d_q * scale).reshape(origin_shape) # FP32

def gen_hierarchy_golden(x, expert_ids, expert_scales, server_num, moe_expert_num_per_server, is_comm_quant=False):
    dtype = x.dtype
    k = expert_ids.shape[-1]
    bs = x.shape[0]
    h = x.shape[-1]
    if x.dim() == 2:
        x = x.unsqueeze(1).repeat_interleave(k, dim=1)
    golden = torch.zeros_like(x[:, 0, :])
    for i in range(bs):
        if is_comm_quant and dtype == torch.float16:
            res = torch.zeros(h, dtype=torch.float16)
        else:
            res = torch.zeros(h, dtype=torch.float32)
        for server_id in range(server_num):
            res_server = torch.zeros(h, dtype=torch.float32)
            flag = 0
            expert_ids_sorted, idx = expert_ids[i].sort()
            expert_scales_sorted = expert_scales[i][idx]
            x_sorted = x[i][idx]
            for token, expert_id, expert_scale in zip(x_sorted, expert_ids_sorted, expert_scales_sorted):
                if(expert_id // moe_expert_num_per_server == server_id):
                    tmp = token.float() * expert_scale.float()
                    res_server += tmp
                    flag = 1
            if(flag):
                if is_comm_quant:
                    if dtype == torch.float16:
                        res_server = quant_int(res_server, 16) # FP16
                    else:
                        res_server = quant_int(res_server, 8, is_bf16=True) # FP32
                else:
                    res_server = res_server.to(dtype).to(torch.float32) # FP32 -> BF16 / FP16 -> FP32
                res += res_server
        if is_comm_quant and dtype == torch.float16:
            golden[i] = res # FP16
        else:
            golden[i] = res.to(dtype)
    return golden

class MoeDistributeCombineGolden:
    """
    原先EP_TEST的标杆生成逻辑，标杆稳定性肯定是最高的，建议CPU优先使用
    """
    def __init__(self, x, expert_scales, expert_ids, moe_expert_num, ep_world_size, comm_quant_mode=0, x_active_mask_2d=None,
        const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, is_layered=None):
        self.server_num = max(ep_world_size // 8, 1)
        self.moe_expert_num_in_server = moe_expert_num // self.server_num 
        self.x = x
        self.expert_scales = expert_scales
        self.expert_ids = expert_ids
        self.comm_quant_mode = comm_quant_mode
        self.x_active_mask_2d = x_active_mask_2d
        self.const_expert_alpha_1 = const_expert_alpha_1
        self.const_expert_alpha_2 = const_expert_alpha_2
        self.const_expert_v = const_expert_v
        self.is_layered = is_layered

    def process(self, is_golden=True):
        if is_golden or not self._is_layered():
            return self._get_bm_or_fullmesh_golden(is_golden)
        is_comm_quant = self.comm_quant_mode == 2
        return gen_hierarchy_golden(
            x=self.x, expert_ids=self.expert_ids, 
            expert_scales=self.expert_scales, server_num=self.server_num,
            moe_expert_num_per_server=self.moe_expert_num_in_server, is_comm_quant=is_comm_quant)

    def _is_layered(self):
        if self.is_layered is not None:
            return self.is_layered
        is_layered = False
        if (os.getenv('HCCL_INTRA_PCIE_ENABLE') == "1" and os.getenv('HCCL_INTRA_ROCE_ENABLE') == "0"):
            is_layered = True
        return is_layered

    def _get_bm_or_fullmesh_golden(self, is_golden=True):
        if is_golden:
            dtype = torch.float64
        else:
            dtype = torch.float32
        x = self.x
        origin_dtype = x.dtype
        x = x.to(dtype)
        if x.dim() == 2:
            x = x.unsqueeze(-2)
        expert_scales = self.expert_scales.unsqueeze(-1).to(dtype)
        if self.x_active_mask_2d is None:
            golden_output = (x * expert_scales).sum(dim=-2)
            if not is_golden:
                golden_output = golden_output.to(origin_dtype)
        return golden_output