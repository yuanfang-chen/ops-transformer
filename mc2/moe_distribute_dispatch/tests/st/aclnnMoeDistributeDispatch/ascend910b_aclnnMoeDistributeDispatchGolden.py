 # -----------------------------------------------------------------------------------------------------------
 	 # Copyright (c) 2025 Huawei Technologies Co., Ltd.
 	 # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 	 # CANN Open Software License Agreement Version 2.0 (the "License").
 	 # Please refer to the License for details. You may not use this file except in compliance with the License.
 	 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 	 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 	 # See LICENSE in the root of the software repository for the full text of the License.
 	 # -----------------------------------------------------------------------------------------------------------

import os
import torch
import torch.nn.functional as F

class MoeDistributeDispatchGolden:
    """
    原先EP_TEST的标杆生成逻辑，标杆稳定性肯定是最高的，建议CPU优先使用
    """
    def __init__(self, bs_list, x_world, expert_ids_world, moe_expert_num, ep_world_size, quant_mode=0, expert_token_nums_type=1, scales=None, expert_scales_world=None, is_layered=None):
        self.bs_list = bs_list.cpu().tolist()
        self.global_bs = sum(self.bs_list)
        self.x_world = x_world.cpu().split(self.bs_list)
        self.expert_ids_world = expert_ids_world.cpu().split(self.bs_list)
        self.moe_expert_num = moe_expert_num
        self.ep_world_size = ep_world_size
        self.quant_mode = quant_mode
        self.expert_token_nums_type = expert_token_nums_type
        self.k = expert_ids_world.shape[-1]
        self.h = x_world.shape[-1]
        self.dtype = x_world.dtype
        self.local_moe_expert_num = moe_expert_num // ep_world_size
        self.A = self.global_bs * min(self.local_moe_expert_num, self.k)
        self.scales = scales
        self.has_scale = False
        if scales is not None:
            self.scales = self.scales.cpu()
            self.has_scale = True
        self.expert_scales_world = expert_scales_world
        if expert_scales_world is not None:
            self.expert_scales_world = expert_scales_world.cpu().split(self.bs_list)
        self.is_layered = is_layered

    def _is_layered(self):
        if self.is_layered is not None:
            return self.is_layered
        is_layered = False
        if (os.getenv('HCCL_INTRA_PCIE_ENABLE') == "1" and os.getenv('HCCL_INTRA_ROCE_ENABLE') == "0"):
            is_layered = True
        return is_layered

    def _get_empty_tensor(self):
        return [torch.tensor([], dtype=torch.int32) for _ in range(self.ep_world_size)]

    def process(self, is_golden=False):
        if not is_golden:
            expand_idx_world = self._init_expand_idx()
            ep_send_counts_world = self._init_ep_send_counts()
            expand_x_world, expand_scales_world = self._init_expand_x_and_expand_scales(expand_idx_world, ep_send_counts_world)
            if self._is_layered():
                ep_send_counts_world = self._init_hierarchy_inner_outer(ep_send_counts_world)
        expand_x_world, dynamic_scales_world, expert_token_nums_world, _ = self._cal_dispatch_golden(self._is_layered(), is_golden=is_golden)
        tp_recv_counts_world = self._get_empty_tensor()
        if is_golden:
            expand_idx_world = self._get_empty_tensor()
            expert_token_nums_world = self._get_empty_tensor()
            ep_send_counts_world = self._get_empty_tensor()
            expand_scales_world = self._get_empty_tensor()
            expand_x_world = self._get_empty_tensor()
        return expand_x_world, dynamic_scales_world, expand_idx_world, expert_token_nums_world, ep_send_counts_world, tp_recv_counts_world, expand_scales_world

    def process_for_combine(self):
        expand_idx_world = self._init_expand_idx()
        ep_send_counts_world = self._init_ep_send_counts()
        expand_x_world, expand_scales_world = self._init_expand_x_and_expand_scales(expand_idx_world, ep_send_counts_world)
        if self._is_layered():
            ep_send_counts_world = self._init_hierarchy_inner_outer(ep_send_counts_world)
        tp_recv_counts_world = [torch.tensor([], dtype=torch.int32) for _ in range(self.ep_world_size)]
        return expand_x_world, expand_idx_world, ep_send_counts_world, tp_recv_counts_world, expand_scales_world

    def _init_expand_idx(self):
        expand_idx_world = torch.zeros((self.global_bs, self.k), dtype=torch.int32).split(self.bs_list)
        for rank_id in range(self.ep_world_size):
            expert_ids_flattened = self.expert_ids_world[rank_id].flatten()
            expand_idx = expand_idx_world[rank_id].view(-1)
            count_dict = {}
            for i in range(len(expert_ids_flattened)):
                value = expert_ids_flattened[i].item()
                count_dict[value] = count_dict.get(value, -1) + 1
                expand_idx[i] = count_dict[value]
        return expand_idx_world

    def _init_expand_x_and_expand_scales(self, expand_idx_world, ep_send_counts_world):
        expand_x_world = torch.zeros((self.A * self.ep_world_size, self.h), dtype=self.dtype)
        expand_scales_world = torch.zeros((self.A * self.ep_world_size, 1), dtype=torch.float32)
        for rank_id in range(self.ep_world_size):
            x = self.x_world[rank_id]
            expert_ids = self.expert_ids_world[rank_id]
            expand_idx = expand_idx_world[rank_id]
            for i in range(self.bs_list[rank_id]):
                for j in range(self.k):
                    expert_id = expert_ids[i][j].item()
                    dst_rank_id = expert_id // self.local_moe_expert_num
                    expert_id_in_rank = expert_id % self.local_moe_expert_num
                    if expert_id_in_rank == 0 and rank_id == 0:
                        base_offset = 0
                    else:
                        base_offset = ep_send_counts_world[dst_rank_id][expert_id_in_rank * self.ep_world_size + rank_id - 1].item()
                    inner_offset = expand_idx[i][j].item()
                    expand_x_world[dst_rank_id * self.A + base_offset + inner_offset] = x[i]
                    expand_scales_world[dst_rank_id * self.A + base_offset + inner_offset] = self.expert_scales_world[rank_id][i][j]
        expand_scales_world = expand_scales_world.view(self.ep_world_size, self.A)
        expand_x_world = expand_x_world.chunk(self.ep_world_size)
        expand_scales_world = expand_scales_world.chunk(self.ep_world_size)
        return expand_x_world, expand_scales_world

    def _init_ep_send_counts(self):
        send_counts = torch.zeros((self.ep_world_size, self.moe_expert_num), dtype=torch.int32)
        for rank_id in range(self.ep_world_size):
            expert_ids = self.expert_ids_world[rank_id].flatten()
            send_counts[rank_id] = torch.bincount(expert_ids, minlength=self.moe_expert_num)
        send_counts = send_counts.T.reshape(self.ep_world_size, self.moe_expert_num).cumsum(-1, dtype=torch.int32)
        ep_send_counts_world = tuple(send_counts[i] for i in range(self.ep_world_size))
        return ep_send_counts_world

    def _init_hierarchy_inner_outer(self, ep_send_counts_world, is_old_hierarchy_cann=False):
        if is_old_hierarchy_cann:
            return self._init_hierarchy_inner_outer_930(ep_send_counts_world)
        else:
            return self._init_hierarchy_inner_outer_newest(ep_send_counts_world)

    def _init_hierarchy_inner_outer_930(self, ep_send_counts_world):
        k, ep_world_size, moe_expert_num = self.k, self.ep_world_size, self.moe_expert_num
        local_moe_expert_num, global_bs = self.local_moe_expert_num, self.global_bs
        rank_num_per_server = 8
        server_num = self.ep_world_size // rank_num_per_server

        # send_counts_outer
        def outer_count_offsets(listA, bs):
            # 初始化结果字典
            result = {}
            # 遍历每个 token_id 从 0 到 bs-1
            for token_id in range(bs):
                count = 0
                offsets = []
                for outer_index, group in enumerate(listA):
                    # 遍历组内的每个 token_id
                    for inner_index, tid in enumerate(group):
                        if tid == token_id:
                            count += 1
                            # 计算偏移量
                            offset = outer_index * bs + inner_index
                            offsets.append(offset)
                result[token_id] = {'count': count, 'offsets': offsets}
            offset_inner = []
            count_inner = []
            for token_id, data in result.items():
                offset_inner.extend(data['offsets'])
                count_inner.append(data['count'])
            return count_inner, offset_inner

        def inner_count_offsets(listA, bs, step):
            # 初始化结果字典
            result = {}
            token_sort_in_data = []
            for token_l in listA:
                for token_sub in token_l:
                    if token_sub not in token_sort_in_data:
                        token_sort_in_data.append(token_sub)
            # 遍历每个 token_id 从 0 到 bs-1
            for token_id in token_sort_in_data:
                count = 0
                offsets = []
                # 遍历 listA 中的每个组
                for outer_index, group in enumerate(listA):
                    # 遍历组内的每个 token_id
                    for inner_index, tid in enumerate(group):
                        if tid == token_id:
                            count += 1
                            # 计算偏移量
                            offset = outer_index * step + inner_index
                            offsets.append(offset)
                # 将结果存储在字典中
                result[token_id] = {'count': count, 'offsets': offsets}
            offset_inner = []
            count_inner = []
            for token_id, data in result.items():
                offset_inner.extend(data['offsets'])
                count_inner.append(data['count'])
            return count_inner, offset_inner

        send_counts_outer = [torch.zeros((self.bs_list[p], ), dtype=torch.int32) for p in range(ep_world_size)]
        offset_outer = [torch.zeros((self.bs_list[p] * server_num, ), dtype=torch.int32) for p in rangeep_world_size]
        send_counts_inner = torch.zeros((ep_world_size, global_bs * server_num), dtype=torch.int16)
        offset_inner = torch.zeros((ep_world_size, global_bs * k * server_num), dtype=torch.int32)
        rank_token_2_expert_total = [[] for _ in range(ep_world_size)] # 存每个rank发给了每个expert哪些token
        for rank_id in range(ep_world_size):
            tokens_2_expert = [[] for _ in range(moe_expert_num)]
            # tokens_2_expert存了moe_expert_num个list，每个list代表这个expert收到了这个rank的哪些token
            expert_ids = self.expert_ids_world[rank_id]
            for token_id in range(len(expert_ids)):
                for tpk in range(k):
                    ep_id = expert_ids[token_id][tpk]
                    tokens_2_expert[ep_id].append(token_id)
            rank_token_2_expert_total[rank_id] = tokens_2_expert
            # 单机专家数为moe_expert_num/server_num,以此为粒度取list融合，然后去重,就可以算出单卡的outer
            token_2_server = [[] for _ in range(server_num)]
            for expert_id in range(moe_expert_num):
                server_id = int(expert_id / (moe_expert_num / server_num))
                token_2_server[server_id].extend(tokens_2_expert[expert_id])
            def remove_duplicates_preserve_order(listA):
                # 创建一个新的列表来存储去重后的结果
                result = []
                for sublist in listA:
                    unique_sublist = list(dict.fromkeys(sublist))
                    result.append(unique_sublist)
                return result
            token_2_server_ = remove_duplicates_preserve_order(token_2_server)
            count_outer_np, offset_outer_np = outer_count_offsets(token_2_server_, self.bs_list[rank_id])
            for token in range(self.bs_list[rank_id]):
                send_counts_outer[rank_id][token] = count_outer_np[token]
            for offset in range(len(offset_outer_np)):
                offset_outer[rank_id][offset] = offset_outer_np[offset]
        # inner计算,要等tokens_2_expert_total先拿到，然后计算本卡上收到所有同号卡的在本机expert数据
        for rank_id in range(ep_world_size):
            local_node_id = int(rank_id / self.rank_num_per_server)
            local_rank_id = rank_id % self.rank_num_per_server
            expert_node_start = int(local_node_id * local_moe_expert_num * self.rank_num_per_server)
            for sv in range(server_num):
                tokens_2_expert_same_id = []
                for rk in range(self.rank_num_per_server):
                    rank_data = []
                    expert_node_start_ = expert_node_start + rk * local_moe_expert_num
                    for ep in range(local_moe_expert_num):
                        tmp_data = rank_token_2_expert_total[sv * self.rank_num_per_server + local_rank_id][expert_node_start_ + ep]
                        rank_data.extend(tmp_data)
                    tokens_2_expert_same_id.append(rank_data)
                count_inner_np, offset_inner_np = inner_count_offsets(tokens_2_expert_same_id, self.bs_list[rank_id],
                                                                      global_bs * k)
                for offset_ in offset_inner_np:
                    offset_ += global_bs * k * self.rank_num_per_server * sv
                for count in range(len(count_inner_np)):
                    send_counts_inner[rank_id][sv * global_bs + count] = count_inner_np[count]
                for offset in range(len(offset_inner_np)):
                    offset_inner[rank_id][sv * global_bs * k + offset] = offset_inner_np[offset]
        send_counts_outer = [torch.cumsum(p, dim=-1, dtype=torch.int32) for p in send_counts_outer]
        send_counts_inner = send_counts_inner.reshape((ep_world_size, server_num, global_bs))
        send_counts_inner = torch.cumsum(send_counts_inner, dim=-1, dtype=torch.int16)
        send_counts_inner = send_counts_inner.reshape(ep_world_size, global_bs * server_num)
        send_counts_inner = send_counts_inner.view(torch.int32)
        ep_send_counts_world = tuple(torch.cat((ep_send_counts_world[i], send_counts_inner[i], offset_inner[i], send_counts_outer[i], offset_outer[i])) for i in range(ep_world_size))
        return ep_send_counts_world
    def _init_hierarchy_inner_outer_newest(self, ep_send_counts_world):
        k, ep_world_size, moe_expert_num = self.k, self.ep_world_size, self.moe_expert_num
        local_moe_expert_num, global_bs = self.local_moe_expert_num, self.global_bs
        rank_num_per_server = 8
        server_num = ep_world_size // rank_num_per_server
        def inner_reduce_info(expert_ids_world, bs_list, k, ep_world_size, moe_expert_num, rank_num_per_server, global_bs):
            server_num = ep_world_size // rank_num_per_server
            offset_num_per_expert = global_bs
            send_counts_inner = torch.zeros((ep_world_size, server_num, global_bs), dtype=torch.int16)
            offset_inner = torch.full((ep_world_size, server_num, global_bs * k), -1, dtype=torch.int32)
            moe_expert_num_per_server = moe_expert_num // server_num
            exp_cnt_map = torch.zeros((ep_world_size, server_num, moe_expert_num_per_server), dtype=torch.int32)
            inner_offset_cnt = torch.zeros((ep_world_size, server_num), dtype=torch.int32)
            BS_BLOCK_SIZE = 1
            for rank_id in range(ep_world_size):
                src_server_id = rank_id // rank_num_per_server
                src_rank = rank_id % rank_num_per_server
                expert_ids_local = expert_ids_world[rank_id]

                # 提前在InnerCnt表开头写入BS信息
                for server_id in range(server_num):
                    target_rank = server_id * rank_num_per_server + src_rank
                    send_counts_inner[target_rank][src_server_id][0] = len(expert_ids_local)

                for token_id in range(len(expert_ids_local)):
                    have_tpks = torch.zeros((ep_world_size, server_num), dtype=torch.int32)
                    for tpk in range(k):
                        ep_id = expert_ids_local[token_id][tpk]
                        if ep_id.item() >= moe_expert_num:
                            continue
                        server_id = ep_id // moe_expert_num_per_server
                        local_tpk = ep_id % moe_expert_num_per_server
                        target_rank = server_id * rank_num_per_server + src_rank
                        send_counts_inner[target_rank][src_server_id][token_id + BS_BLOCK_SIZE] += 1
                        offset_inner[target_rank][src_server_id][token_id * k + have_tpks[target_rank][src_server_id]] = exp_cnt_map[rank_id][server_id][local_tpk] + local_tpk * offset_num_per_expert
                        index = int(have_tpks[target_rank][src_server_id])
                        while index > 0 and offset_inner[target_rank][src_server_id][token_id * k + index] < offset_inner[target_rank][src_server_id][token_id * k + index - 1]:
                            temp_value = offset_inner[target_rank][src_server_id][token_id * k + index - 1].clone()
                            offset_inner[target_rank][src_server_id][token_id * k + index - 1] = offset_inner[target_rank][src_server_id][token_id * k + index].clone()
                            offset_inner[target_rank][src_server_id][token_id * k + index] = temp_value.clone()
                            index -= 1
                        have_tpks[target_rank][src_server_id] += 1
                        exp_cnt_map[rank_id][server_id][local_tpk] += 1
                        inner_offset_cnt[target_rank][src_server_id] += 1
            send_counts_inner = send_counts_inner.reshape(ep_world_size, server_num * global_bs)
            send_counts_inner = send_counts_inner.view(torch.int32)
            offset_inner = offset_inner.reshape(ep_world_size, server_num * global_bs * k)
            return send_counts_inner, offset_inner

        def outer_reduce_info(expert_ids_world, bs_list, k, ep_world_size, moe_expert_num, rank_num_per_server, global_bs):
            server_num = ep_world_size // rank_num_per_server
            send_counts_outer = [torch.zeros((bs), dtype=torch.int32) for bs in bs_list]
            offset_outer = [torch.full((server_num * bs,), -1, dtype=torch.int32) for bs in bs_list]
            moe_expert_num_per_server = moe_expert_num // server_num
            for rank_id in range(ep_world_size):
                offset_num_per_server = bs_list[rank_id]
                cnt_server = torch.zeros((server_num, ), dtype=torch.int32)
                expert_ids_local = expert_ids_world[rank_id]
                for token_id in range(bs_list[rank_id]):
                    is_send_server = [False for _ in range(server_num)]
                    for tpk in range(k):
                        ep_id = expert_ids_local[token_id][tpk]
                        server_id = ep_id // moe_expert_num_per_server
                        is_send_server[server_id] = True
                    count = 0
                    for server_id, is_send in enumerate(is_send_server):
                        if is_send:
                            offset_outer[rank_id][token_id * server_num + count] = offset_num_per_server * server_id + cnt_server[server_id]
                            count += 1
                            cnt_server[server_id] += 1
                    send_counts_outer[rank_id][token_id] = count
            return send_counts_outer, offset_outer
        ##################
        # #inner计算
        send_counts_inner, offset_inner = inner_reduce_info(self.expert_ids_world, self.bs_list, k, ep_world_size, moe_expert_num,
            rank_num_per_server, global_bs)
        # #outer计算
        send_counts_outer, offset_outer = outer_reduce_info(self.expert_ids_world, self.bs_list, k, ep_world_size, moe_expert_num,
            rank_num_per_server, global_bs)
        # rt_ep_send_counts_world = tuple(torch.cat((ep_send_counts_world[i], send_counts_inner[i], offset_inner[i], send_counts_outer[i], offset_outer[i])) for i in range(ep_world_size))
        # 首先计算每个i对应的张量拼接后的长度
        tensor_lengths = []
        for i in range(ep_world_size):
            total_length = (ep_send_counts_world[i].numel() + send_counts_inner[i].numel() + 
                        offset_inner[i].numel() + send_counts_outer[i].numel() + offset_outer[i].numel())
            tensor_lengths.append(total_length)

        # 找到最大的长度
        max_length = max(tensor_lengths)

        # 创建对齐后的元组
        rt_ep_send_counts_world = tuple(
            torch.cat((
                ep_send_counts_world[i], 
                send_counts_inner[i], 
                offset_inner[i], 
                send_counts_outer[i], 
                offset_outer[i],
                # 如果当前长度小于最大长度，填充零使其对齐
                torch.zeros(max_length - tensor_lengths[i], dtype=ep_send_counts_world[i].dtype, device=ep_send_counts_world[i].device)
            )) for i in range(ep_world_size)
        )

        return rt_ep_send_counts_world

    def _get_final_golden(self, vaild_expert_token_nums_world, dynamic_scales_world_sorted, expand_x_world_sorted):
        expert_token_nums_world = F.pad(vaild_expert_token_nums_world, (0, self.moe_expert_num - vaild_expert_token_nums_world.size(0)), 'constant', 0)
        expert_token_nums_world_cumsum = []
        expert_token_nums_world_normal = []
        expert_num_per_rank = self.moe_expert_num // self.ep_world_size
        for rank_id in range(self.ep_world_size):
            count = torch.cumsum(expert_token_nums_world[rank_id * expert_num_per_rank : (rank_id + 1) * expert_num_per_rank], dim=0)
            expert_token_nums_world_cumsum.append(count)
            expert_token_nums_world_normal.append(expert_token_nums_world[rank_id * expert_num_per_rank : (rank_id + 1) * expert_num_per_rank])
        actual_tokens = []
        for rank_id in range(self.ep_world_size):
            count = torch.sum(expert_token_nums_world[rank_id * expert_num_per_rank : (rank_id + 1) * expert_num_per_rank]).item()
            actual_tokens.append(count)
        actual_tokens = tuple(actual_tokens)

        expand_x_world = []
        dynamic_scales_world = []
        count = 0
        for rank_id in range(self.ep_world_size):
            start = count
            end = start + actual_tokens[rank_id]
            count = end
            expand_x_world.append(expand_x_world_sorted[start:end])
            if self.quant_mode == 2:
                dynamic_scales_world.append(dynamic_scales_world_sorted[start:end])
            else:
                dynamic_scales_world.append(torch.tensor([], dtype=torch.float32))
        expand_x_world = tuple(expand_x_world)
        dynamic_scales_world = tuple(dynamic_scales_world)
        if self.expert_token_nums_type == 1:
            expert_token_nums_world = tuple(expert_token_nums_world_normal)
        else:
            expert_token_nums_world = tuple(expert_token_nums_world_cumsum)

        return expand_x_world, dynamic_scales_world, expert_token_nums_world, actual_tokens

    def _cal_dispatch_golden(self, is_layered, is_golden=False):
        x_world = torch.cat(self.x_world)
        expert_ids_world = torch.cat(self.expert_ids_world).view(-1)
        expand_x_world = torch.repeat_interleave(x_world, self.k, dim=0)
        if is_golden:
            dtype = torch.float64
        else:
            dtype = torch.float32

        if self.has_scale and self.quant_mode == 2 and not is_layered:
            scales = self.scales.to(dtype)
            expand_x_world = expand_x_world.to(dtype)
            indices = expert_ids_world.unsqueeze(1).expand(-1, self.h).to(torch.int64)
            scales_gather = torch.gather(scales, 0, indices)
            expand_x_world = torch.mul(expand_x_world, scales_gather)

        dynamic_scales_world = None
        if self.quant_mode == 2:
            expand_x_world = expand_x_world.to(dtype)
            max_value, _ = torch.max(torch.abs(expand_x_world), dim=1)
            dynamic_scales_world = (torch.tensor([127.0]).to(dtype) / max_value).view(-1, 1).to(dtype)
            expand_x_world = expand_x_world * dynamic_scales_world
            expand_x_world = expand_x_world.to(torch.int8)
        else:
            expand_x_world = expand_x_world.to(self.dtype)

        expert_ids_world_sorted, sorted_idx = torch.sort(expert_ids_world, stable=True)
        torch.sort(sorted_idx)
        expand_x_world_sorted = expand_x_world[sorted_idx]

        dynamic_scales_world_sorted = None
        if self.quant_mode == 2:
            dynamic_scales_world_sorted = 1 / dynamic_scales_world[sorted_idx].view(-1)
        vaild_expert_token_nums_world = torch.bincount(expert_ids_world).to(torch.int32)
        return self._get_final_golden(vaild_expert_token_nums_world, dynamic_scales_world_sorted, expand_x_world_sorted)
