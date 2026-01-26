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
import json
import torch
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass

from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.configs.nodes_config import Node
from atk.configs.nodetype_config import NodeType
from atk.configs.results_config import AccuracyConfig

from atk.common.utils import TimeTracker
from atk.common.connection import RemoteManager
from atk.tasks.post_process.equal_compare import Compare as equal_compare

from atk.tasks.post_process.base_compare import BaseAccuracyCompare
from atk.tasks.post_process.cv_fused_double_benchmark_compare import CvFusedDoubleBenchmarkAccuracyCompare

class CvFusedDoubleBenchmarkAccuracyCompareMC2(CvFusedDoubleBenchmarkAccuracyCompare):
    def _get_bm_data(self, data_file):
        if self.remote_manager.main_node.backend == NodeType.DIST or self.remote_manager.main_node.is_dist:
            bm_dir_list = self._get_dist_bm_dir(self.bm_path)
            bm_data = self.get_concat_data(data_file, bm_dir_list)
            return bm_data

        bm_dir = os.path.join(self.bm_path, data_file)
        if Path(data_file).suffix == '.bin':
            dtype = self._get_output_dtype(data_file, self.bm_path)
            bm_data = self.load_output_numpy(bm_dir, dtype)
        else:
            bm_data = BaseAccuracyCompare.load_output_tensor(
                bm_dir, is_npu=self.is_npu, device_id=self.device_id
            )
        return bm_data

    def compare_for_data(self, data_file, time_tracker, accuracy_result_list):
        if Path(data_file).suffix not in {'.pt', '.bin'}:
            return
        output_index = os.path.splitext(data_file)[0]
        if self.remote_manager.main_node.backend == NodeType.DIST or self.remote_manager.main_node.is_dist:
            local_output, remote_output, acc_result_flag, acc_result = self.get_dist_output(data_file)
            time_tracker.record(f"{data_file}: download_data")
            if self.need_md5 and acc_result_flag:
                accuracy_result_list.append(acc_result)
                return
            self.executor.download_benchmark_data()
            time_tracker.record(f"{data_file}: download_benchmark_data")
        time_tracker.record(f"{data_file}: load_output_tensor")
        acc_result = self.compute_accuracy_result(local_output, remote_output, data_file)
        if acc_result:
            accuracy_result_list.append(acc_result)
        time_tracker.record(f"{data_file}: compute_accuracy_result")

def compare_innerOrOuter(npu, golden):
    result = (npu - golden).abs().sum().item()
    result &= torch.equal(npu, golden)
    return result

class MoeDistributeDispatchCompare:
    @staticmethod
    def valid_expand_scales(rank, expand_scales, expand_scales_golden, actual_tokens):
        _expand_scales_golden = expand_scales_golden.view(-1)[:actual_tokens]
        _expand_scales = expand_scales[:actual_tokens]
        result = torch.equal(_expand_scales_golden, _expand_scales)
        return result

    @staticmethod
    def valid_reduce_info(rank, ep_recv_counts, ep_recv_counts_golden, moe_expert_num, global_bs, server_num, k, bs):
        results = []
        offset_ep_recv_counts = moe_expert_num
        offset_inner_counts = offset_ep_recv_counts + (global_bs * server_num) // 2
        offset_inner_offset = offset_inner_counts + global_bs * k * server_num
        offset_outer_counts = offset_inner_offset + bs
        offset_outer_offset = offset_outer_counts + bs * server_num
        inner_counts_golden = ep_recv_counts_golden[offset_ep_recv_counts : offset_inner_counts]
        inner_counts = ep_recv_counts[offset_ep_recv_counts : offset_inner_counts]
        inner_offsets_golden = ep_recv_counts_golden[offset_inner_counts : offset_inner_offset].view(torch.int32)
        inner_offsets = ep_recv_counts[offset_inner_counts : offset_inner_offset].view(torch.int32)

        inner_counts_golden = inner_counts_golden.view(torch.int16)
        inner_counts = inner_counts.view(torch.int16)
        res_counts_inner = []
        res_offset_inner = []
        BS_BLOCK_SIZE = 1
        for i in range(server_num):
            begin = i * global_bs
            real_bs_golden = inner_counts_golden[begin].to(torch.int32)
            real_bs = inner_counts[begin].to(torch.int32)
            if real_bs_golden != real_bs:
                res_counts_inner.append(False)
                res_offset_inner.append(False)
                break
            end = begin + BS_BLOCK_SIZE + real_bs_golden
            inner_counts_result = compare_innerOrOuter(inner_counts_golden[begin:end], inner_counts[begin:end])
            res_counts_inner.append(inner_counts_result)
            begin = i * global_bs * k
            end = begin + real_bs_golden * k
            inner_offset_result = compare_innerOrOuter(inner_offsets_golden[begin:end], inner_offsets[begin:end])
            res_offset_inner.append(inner_offset_result)
        if False in res_counts_inner:
            results.append(False)
        else:
            results.append(True)

        if False in res_offset_inner:
            results.append(False)
        else:
            results.append(True)

        outer_counts_golden = ep_recv_counts_golden[offset_inner_offset : offset_outer_counts].view(torch.int32)
        outer_counts = ep_recv_counts[offset_inner_offset : offset_outer_counts].view(torch.int32)
        outer_offsets_golden = ep_recv_counts_golden[offset_outer_counts : offset_outer_offset].view(torch.int32)
        outer_offsets = ep_recv_counts[offset_outer_counts : offset_outer_offset].view(torch.int32)
        outer_counts_result = compare_innerOrOuter(
            outer_counts_golden, outer_counts)
        if not outer_counts_result:
            results.append(False)
        else:
            results.append(True)

        outer_offset_result = compare_innerOrOuter(outer_offsets_golden, outer_offsets)
        if not outer_offset_result:
            results.append(False)
        else:
            results.append(True)

        return results


@dataclass
class StoreDataInfo:
    nums: int = -1 # 需要对比的数量，可能是int，可能是List
    element_length: int = 1 # h or 1
    output: torch.tensor = None # 保存的算子输出
    golden: torch.tensor = None # 保存的算子标杆
    need_change: bool = False # 对比的内容尺寸是否需要改变, 与is_need_load_cut只能一个满足
    need_save: bool = False # 是否需要保存
    save_reversed: bool = False # 保存nums前还是后
    is_need_load_cut: bool = False # 加载时就剪切
    need_default_check: bool = True # 走默认ATK对比

@ACCURACY_REGISTRY.register("cv_fused_double_benchmark_MoeDistributeDispatch")
class AccuracyStandardV2(CvFusedDoubleBenchmarkAccuracyCompareMC2):
    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)

    def precision_compare_for_case(
        self, main_node: Node, remote_node: Node, is_npu=False, device_id=0
    ) -> List[AccuracyConfig]:
        self.is_npu = is_npu
        self.device_id = device_id
        time_tracker = TimeTracker()
        self.remote_manager = RemoteManager(self.case_config, main_node, remote_node)
        time_tracker.record("RemoteManager init")
        if main_node.backend == NodeType.DIST or main_node.is_dist:
            local_dir, remote_dir = self.remote_manager.get_dist_local_and_remote_dir()
        else:
            local_dir, remote_dir = self.remote_manager.get_local_and_remote_dir()

        acc_result = self.compare_file_count(local_dir, remote_dir)
        time_tracker.record("compare_file_count")
        if acc_result:
            return [acc_result]
        
        # update data
        if isinstance(local_dir, list):
            local_dir_tmp = local_dir[0]
        else:
            local_dir_tmp = local_dir
        self.ep_world_size = len(local_dir)
        self.get_base_info()
        # 默认 local_dir[0] 在本地 
        data_files = []
        for data_file in sorted(os.listdir(local_dir_tmp)):
            if Path(data_file).suffix not in {'.pt', '.bin'}:
                continue
            data_files.append(data_file)

        self.name_to_data_file_map = {
            "expandX": data_files[0],
            "dynamicScales": data_files[1],
            "expandIdx": data_files[2],
            "expertTokenNum": data_files[3],
            "epRecvCounts": data_files[4],
            "expandScales": data_files[6],
            "bsList": data_files[7],
        }
        self.data_file_to_name_map = {value: key for key, value in self.name_to_data_file_map.items()}
        self.data_infos = {
            "expertTokenNum": StoreDataInfo(need_save=True),
            "bsList": StoreDataInfo(need_save=True),
            "dynamicScales": StoreDataInfo(is_need_load_cut=True),
            "expandX": StoreDataInfo(element_length=self.h, is_need_load_cut=True),
            "expandScales": StoreDataInfo(need_save=True, is_need_load_cut=True, need_default_check=False),
            "expandIdx": StoreDataInfo(element_length=self.k, is_need_load_cut=True),
            "epRecvCounts": StoreDataInfo(nums=self.moe_expert_num, need_change=True, need_save=True, save_reversed=True),
        }

        accuracy_result_list = []
        # 0-expandX, 1-dynamicScales, 2-expandIdx, 3-expertTokenNum, 4-epRecvCounts, 5-tpRecvCounts, 6-expandScales,
        # expertTokenNum, 二进制一致，其次保留expertTokenNum，后续获取actualTokens
        self.data_infos["expertTokenNum"] = StoreDataInfo(need_save=True)
        self.compare_for_data(self.name_to_data_file_map['expertTokenNum'], time_tracker, accuracy_result_list)
        expert_token_num_world = self.data_infos["expertTokenNum"].golden.view(self.ep_world_size, -1)
        actual_tokens = expert_token_num_world[:, -1].tolist() if self.expert_token_nums_type == 0 else expert_token_num_world.sum(dim=-1).tolist()
        self.data_infos.pop('expertTokenNum', None)

        self.data_infos["expandX"].nums = actual_tokens
        self.data_infos["dynamicScales"].nums = actual_tokens
        if self.quant_mode != 2:
            self.data_infos.pop('dynamicScales', None)
            # 非量化，不走默认的对比方案，需要二进制一致，不这么做，会进入双标杆
            self.data_infos["expandX"].need_default_check = False
            self.data_infos["expandX"].need_save = True
        else:
            self.data_infos["dynamicScales"].nums = actual_tokens
        is_layered = self._is_layered()
        if not is_layered:
            self.data_infos.pop('expandScales', None)
        else:
            self.data_infos['expandIdx'].need_default_check = False # 分层不比较expandIdx
            self.data_infos['expandScales'].nums = actual_tokens

        # bsList，辅助V2使用，需要用来对比ExpandIdx
        self.compare_for_data(self.name_to_data_file_map['bsList'], time_tracker, accuracy_result_list)
        bs_list = self.data_infos["bsList"].golden.tolist()
        self.data_infos["expandIdx"].nums = bs_list # 兼容V2
        self.data_infos.pop('bsList', None)
        self.global_bs = sum(bs_list) if self.global_bs == 0 else self.global_bs
        self.executor.download_benchmark_data()
        for key in self.data_infos.keys():
            self.compare_for_data(self.name_to_data_file_map[key], time_tracker, accuracy_result_list)
        if is_layered:
            # Inner Outer
            self.data_infos["epRecvCounts"].output = self.data_infos["epRecvCounts"].output.view(self.ep_world_size, -1)
            self.data_infos["epRecvCounts"].golden = self.data_infos["epRecvCounts"].golden.view(self.ep_world_size, -1)
            for rank_id in range(self.ep_world_size):
                bs = bs_list[rank_id]
                self.compare_innerOrOuter(rank_id, bs, self.data_infos["epRecvCounts"].output[rank_id], self.data_infos["epRecvCounts"].golden[rank_id],
                    self.name_to_data_file_map["epRecvCounts"], accuracy_result_list)
            # expandScales, 二进制一致
            instance_equal_compare = equal_compare(self.case_config)
            acc_ret = instance_equal_compare.compute_accuracy_result(self.data_infos["expandScales"].output, self.data_infos["expandScales"].output, self.name_to_data_file_map["expandScales"])
            if acc_ret:
                accuracy_result_list.append(acc_ret)

        if self.quant_mode != 2:
            # 非量化，expandX要求二进制一直，否则走默认的Int量化
            instance_equal_compare = equal_compare(self.case_config)
            acc_ret = instance_equal_compare.compute_accuracy_result(self.data_infos["expandX"].output, self.data_infos["expandX"].output, self.name_to_data_file_map["expandX"])
            accuracy_result_list.append(acc_ret)

        time_tracker.record(f"{data_file}: compute_accuracy_result")
        logging.debug(f"accuracy_result_list: %s", accuracy_result_list)
        logging.debug(time_tracker)
        return accuracy_result_list

    def _is_layered(self):
        is_layered = False
        if (os.getenv('HCCL_INTRA_PCIE_ENABLE') == "1" and os.getenv('HCCL_INTRA_ROCE_ENABLE') == "0"):
            is_layered = True
        return is_layered

    def compare_innerOrOuter(self, rank_id, bs, ep_recv_counts, ep_recv_counts_golden, data_file, accuracy_result_list):
        server_num = self.ep_world_size // 8
        result = MoeDistributeDispatchCompare.valid_reduce_info(rank_id,
            ep_recv_counts, ep_recv_counts_golden, 0, self.global_bs, server_num, self.k, bs)
        if False in result:
            ret = None
            error_info = (
                f"[Rank:{rank_id}] InnerOrOuter, output != golden"
            )
            ret = AccuracyConfig(
                result=False,
                error_info=error_info,
                file_name=data_file
            )
            accuracy_result_list.append(ret)

    def get_base_info(self):
        name_location_map = {}
        case_config = self.case_config
        for i in range(len(case_config.inputs)):
            name_location_map[case_config.inputs[i].name] = i
        _, self.h = self.case_config.inputs[name_location_map['x']].shape
        self.k = self.case_config.inputs[name_location_map['expertIds']].shape[1]
        self.moe_expert_num = self.case_config.inputs[name_location_map['moeExpertNum']].range_values
        self.quant_mode = self.case_config.inputs[name_location_map['quantMode']].range_values
        self.global_bs = self.case_config.inputs[name_location_map['globalBs']].range_values
        self.expert_token_nums_type = self.case_config.inputs[name_location_map['expertTokenNumsType']].range_values

    # 只拼接每张卡的有效数据
    def get_concat_data(self, data_file, dir_list):
        data_list = []
        dir_num = len(dir_list)
        if dir_num == 0:
            raise Exception("output_data dir list length is zero, please check operator's output!")
        need_cut = False
        element_length = None
        tokens = None
        if data_file in self.data_file_to_name_map.keys() and self.data_file_to_name_map[data_file] in self.data_infos.keys():
            name = self.data_file_to_name_map[data_file]
            if self.data_infos[name].is_need_load_cut and type(self.data_infos[name].nums) == list and len(self.data_infos[name].nums) == self.ep_world_size:
                need_cut = True
                element_length = self.data_infos[name].element_length
                tokens = self.data_infos[name].nums
        for dir_idx in range(dir_num):
            local_data_path = os.path.join(dir_list[dir_idx], data_file)
            if need_cut:
                data_list.append(self.load_output_tensor(
                    local_data_path, is_npu=self.is_npu, device_id=self.device_id
                )[:tokens[dir_idx] * element_length])
            else:
                data_list.append(self.load_output_tensor(
                    local_data_path, is_npu=self.is_npu, device_id=self.device_id
                ))
        return torch.concat(data_list)

    def compute_accuracy_result(
            self, local_output, remote_output, data_file
    ) -> AccuracyConfig:
        local_output, remote_output = self.store_data(local_output, remote_output, data_file)
        need_default_check = True
        if data_file in self.data_file_to_name_map.keys() and self.data_file_to_name_map[data_file] in self.data_infos.keys():
            name = self.data_file_to_name_map[data_file]
            need_default_check = self.data_infos[name].need_default_check
        acc_ret = None
        if need_default_check:
            acc_ret = super().compute_accuracy_result(local_output, remote_output, data_file)
        return acc_ret

    def deal_data(self, output, data_file):
        if data_file in self.data_file_to_name_map.keys() and self.data_file_to_name_map[data_file] in self.data_infos.keys():
            name = self.data_file_to_name_map[data_file]
            if self.data_infos[name].is_need_load_cut: # 已剪
                return output, output
            if type(self.data_infos[name].nums) == int and self.data_infos[name].nums >= 0:
                _output = output.view(self.ep_world_size, -1)
                nums = self.data_infos[name].nums * self.data_infos[name].element_length
                _output_front = _output[:, :nums].contiguous().view(-1)
                _output_rear = _output[:, nums:].contiguous().view(-1)
            elif type(self.data_infos[name].nums) == list and len(self.data_infos[name].nums) == self.ep_world_size:
                _output = output.view(self.ep_world_size, -1)
                # 创建掩码来提取每行对应的元素
                nums = torch.tensor(self.data_infos[name].nums).unsqueeze(1) * self.data_infos[name].element_length
                # 创建前部分和后部分的掩码
                front_mask = cols < nums
                back_mask = cols >= nums
                # 提取后部分元素
                _output_front = _output[front_mask].contiguous().view(-1)
                _output_rear = _output[back_mask].contiguous().view(-1)
            else:
                _output_front = output
                _output_rear = output
            return _output_front, _output_rear
        return output, output

    def store_data(self, local_output, remote_output, data_file):
        _local_output_front, _local_output_rear = self.deal_data(local_output, data_file)
        _remote_output_front, _remote_output_rear = self.deal_data(remote_output, data_file)
        if data_file in self.data_file_to_name_map.keys() and self.data_file_to_name_map[data_file] in self.data_infos.keys():
            name = self.data_file_to_name_map[data_file]
            if self.data_infos[name].need_save:
                # expertTokenNum: 全部保留，且全部比较
                # epRecvCount: 尾部保留，只比较开头MoeExpertNum个数
                self.data_infos[name].output = _local_output_rear if self.data_infos[name].save_reversed else _local_output_front
                self.data_infos[name].golden = _remote_output_rear if self.data_infos[name].save_reversed else _remote_output_front

            if self.data_infos[name].need_change:
                local_output = _local_output_front
                remote_output = _remote_output_front
        return local_output, remote_output