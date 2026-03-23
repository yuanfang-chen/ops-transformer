# npu\_low\_latency\_ccl\_buffer\_size<a name="ZH-CN_TOPIC_0000002343094194"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002203575834_section14441124184111"></a>

-   API功能：
    
    需与[npu\_low\_latency_dispatch](npu_low_latency_dispatch.md)和[npu\_low\_latency_combine](npu_low_latency_combine.md)配套使用，用于计算dispatch_V3和combine_v3算子所需的HCCL通信buffer_size大小（单位：MB）。该接口为静态方法，可在初始化`MoeDistributeBuffer`前调用。
-   计算公式：

    -   计算token实际长度：

        $$token\_actual\_len = Align32(hidden * max_out\_dtype\_size) + scale\_expand\_index\_buffer$$

        其中max_out_dtype_size为token类型转为字节数，scale_expand_index_buffer为scale（32B）+ expand\_idx（3 × 4B）

    -   根据`comm_alg`计算dispatch阶段每个token所需大小：

        若`comm_alg`为"fullmesh\_v2"：

        $$token\_need\_size\_dispatch = Align480(token\_actual\_len) / full\_mesh\_data\_align * win\_addr\_align$$

        若`comm_alg`为"fullmesh\_v1"或""：

        $$token\_need\_size\_dispatch = Align512(token\_actual\_len)$$

    -   计算combine阶段每个token所需大小：

        $$token\_need\_size\_combine = Align512(hidden * max_out\_dtype\_size)$$
    
        其中，full_mesh_data_align为480对齐，win_addr_align为512对齐

    -   计算最ccl_buffer_size大小：

        $$dispatch\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_dispatch * world\_size * local\_moe\_expert\_num$$
        $$combine\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_combine * (topk + num\_shared\_expert)$$
        $$ccl\_buffer\_size = Align1MB(dispatch\_buffer\_size + combine\_buffer\_size) / 1MB$$

## 函数原型<a name="zh-cn_topic_0000002203575834_section45077510412"></a>

```
get_low_latency_ccl_buffer_size(world_size, num_max_dispatch_tokens_per_rank, hidden, num_moe_expert, topk, num_shared_expert=0, num_shared_expert_ranks=0, comm_alg="") -> int
```

## 参数说明<a name="zh-cn_topic_0000002203575834_section112637109430"></a>

-   **world\_size** (`int`)：必选参数，表示通信域的大小。取值范围[2, 768]。
-   **num\_max\_dispatch\_tokens\_per\_rank** (`int`)：必选参数，表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小。
-   **hidden** (`int`)：必选参数，表示hidden size隐藏层大小。取值范围[1024, 8192]。
-   **num\_moe\_expert** (`int`)：必选参数，MoE专家数量，取值范围[1, 1024]，并且满足以下条件：num\_experts\%(ep\_world\_size - shared\_expert\_rank\_num)\=0。
-   **topk** (`int`)：必选参数，表示选取topK个专家，取值范围为0 < K ≤ 16，同时满足0 < K ≤ num_experts + zero_expert_num + copy_expert_num + const_expert_num。
-   **num\_shared\_expert** (`int`)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为0。
-   **num\_shared\_expert\_ranks** (`int`)：可选参数，表示共享专家卡数量。必须满足`world_size - num_shared_expert_ranks > 0`，默认值为0。
-   **comm\_alg** (`string`)：可选参数，表示通信算法。支持取值："fullmesh\_v1"、"fullmesh\_v2"或""（默认）。

## 输出说明<a name="zh-cn_topic_0000002203575834_section22231435518"></a>

`int`

表示计算得到的ccl_buffer_size大小，单位为MB。

## 约束说明<a name="zh-cn_topic_0000002203575834_section12345537164215"></a>
-   npu_low_latency_ccl_buffer_size、npu_low_latency_dispatch和npu_low_latency_combine必须配套使用
-   BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0<BS≤512。
-   本文公式中的“/”表示整除。

## 调用示例<a name="zh-cn_topic_0000002203575834_section14459801436"></a>

-   单算子调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    from npu_ops_transformer.ops import MoeDistributeBuffer

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    num_experts = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0


    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1


    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2


    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v


    def get_new_group(rank):
        for i in range(tp_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info


    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 16,
        }


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                                [10, 12, 13, 15, 19, 4, 18, 1],
                                [19, 33, 1, 17, 9, 5, 0, 32],
                                [19, 11, 17, 0, 10, 5, 7, 9],
                                [10, 16, 11, 17, 33, 8, 9, 3],
                                [12, 19, 5, 7, 1, 3, 18, 16],
                                [11, 9, 13, 16, 12, 33, 17, 14],
                                [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
        topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()

        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None

        distribute_buffer = MoeDistributeBuffer(ep_group)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, _ = distribute_buffer.npu_low_latency_dispatch(
            x=x,
            topk_idx=topk_idx,
            shared_expert_num=0,
            shared_expert_rank_num=shared_expert_rank_num,
            num_experts=num_experts,
            quant_mode=quant_mode,
            num_max_dispatch_tokens_per_rank=globalBS,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num)

        if is_quant:
            expand_x = expand_x.to(input_dtype)

        x = distribute_buffer.npu_low_latency_combine(x=expand_x,
                                                topk_idx=topk_idx,
                                                assist_info_for_combine=assist_info_for_combine,
                                                ep_send_counts=ep_recv_counts,
                                                topk_weights=topk_weights,
                                                shared_expert_num=0,
                                                shared_expert_rank_num=shared_expert_rank_num,
                                                num_experts=num_experts,
                                                num_max_dispatch_tokens_per_rank=globalBS,
                                                ori_x=x,
                                                const_expert_alpha_1=const_expert_alpha_1,
                                                const_expert_alpha_2=const_expert_alpha_2,
                                                const_expert_v=const_expert_v,
                                                zero_expert_num=zero_expert_num,
                                                copy_expert_num=copy_expert_num,
                                                const_expert_num=const_expert_num)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)
        buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(ep_world_size, bs, h, num_experts, k)
        os.environ['HCCL_BUFFSIZE'] = f'buffer_size'

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)
        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)
        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)
        if num_experts % moe_rank_num != 0:
            print("num_experts 必须是 moe_rank_num 的整数倍")
            exit(0)
        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```

-   图模式调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time
    from npu_ops_transformer.ops import MoeDistributeBuffer

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    num_experts = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0


    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self, group_ep):
            super().__init__()
            self.group = group_ep
            self.distribute_buffer = MoeDistributeBuffer(group_ep)

        def forward(self, x, topk_idx, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, num_experts,
                    scales, quant_mode, num_max_dispatch_tokens_per_rank, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = self.distribute_buffer.npu_low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)

            output_combine_npu = self.distribute_buffer.npu_low_latency_combine(
                x=expand_x_npu,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]


    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1


    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2


    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v


    def get_new_group(rank):
        for i in range(tp_world_size):
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info


    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()

        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'scales': None,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 1,
        }


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()

        topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        elastic_info = None
        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None

        model = MOE_DISTRIBUTE_GRAPH_Model(ep_group)
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, topk_idx, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
            rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, num_experts, scales,
            quant_mode, globalBS, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

        time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)
        buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(ep_world_size, bs, h, num_experts, k)
        os.environ['HCCL_BUFFSIZE'] = f'buffer_size'

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)

        if num_experts % moe_rank_num != 0:
            print("num_experts 必须是 moe_rank_num 的整数倍")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```