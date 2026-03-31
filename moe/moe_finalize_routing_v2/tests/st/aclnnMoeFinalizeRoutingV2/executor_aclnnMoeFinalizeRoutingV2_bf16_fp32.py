#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
def MoeFinalizeRoutingV2Golden(expandedX: torch.Tensor, expandedRowIdx: torch.Tensor, 
                               x1Optional: torch.Tensor, x2Optional: torch.Tensor, 
                               biasOptional: torch.Tensor, scalesOptional: torch.Tensor, 
                               expertIdxOptional: torch.Tensor, dropPadMode: int):
    if len(expandedX.shape) == 2:
        
        Num_Rows = expertIdxOptional.shape[0]
        K = expertIdxOptional.shape[1]
        H = expandedX.shape[-1]
        output = torch.empty(Num_Rows, H)
        for i in range(Num_Rows):
            output[i] = x1Optional[i] + x2Optional[i]
        for i in range(Num_Rows):
            for k in range(K):
                expandedRowIdx_idx = 0
                if dropPadMode == 0:
                    expandedRowIdx_idx = i + k * Num_Rows
                else :
                    expandedRowIdx_idx = i * K + k
                output[i] += scalesOptional[i, k]* (expandedX[expandedRowIdx[expandedRowIdx_idx]] + biasOptional[expertIdxOptional[i, k]])

        return output

    if len(expandedX.shape) == 3:
        E,C,H = expandedX.shape
        expandedX = expandedX.reshape(-1, expandedX.size(-1)) 
        K = scalesOptional.shape[1]
        Num_Rows = expandedRowIdx.shape[0] // K
        output = torch.empty(Num_Rows, H)
        for i in range(Num_Rows):
            output[i] = x1Optional[i] + x2Optional[i]
        for i in range(Num_Rows):
            for k in range(K):
                expandedRowIdx_idx = 0
                if dropPadMode == 1:
                    expandedRowIdx_idx = i + k * Num_Rows
                else :
                    expandedRowIdx_idx = i * K + k
                expandedX_for_idx = torch.zeros(H)
                if int(expandedRowIdx[expandedRowIdx_idx]) != -1:
                    expandedX_for_idx = expandedX[expandedRowIdx[expandedRowIdx_idx]]
                output[i] += scalesOptional[i, k]* (expandedX_for_idx + biasOptional[expertIdxOptional[i, k]])

        return output

@register("function_aclnn_MoeFinalizeRoutingV2")
class AclnnMoeFinalizeRoutingV2(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        seed = input_data.kwargs['seed']
        # input_data.kwargs['groupList'] = generate_non_decreasing_sequence(weight.shape[0], x.shape[0], seed)
        input_data.kwargs.pop('seed')
        input_data.kwargs.pop('case')
        input_data.kwargs.pop('out')
        if input_data.kwargs['isfp16'] == True:
            self.fp16 = True
        else:
            self.fp16 = False
        input_data.kwargs.pop('isfp16')

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        expandedX = input_data.kwargs['expandedX']
        expandedRowIdx = input_data.kwargs['expandedRowIdx']
        x1Optional = input_data.kwargs['x1Optional']
        x2Optional = input_data.kwargs['x2Optional']
        biasOptional = input_data.kwargs['biasOptional']
        scalesOptional = input_data.kwargs['scalesOptional']
        expertIdxOptional = input_data.kwargs['expertIdxOptional']
        dropPadMode = input_data.kwargs['dropPadMode']

        expandedX_dtype = expandedX.dtype

        if self.device == 'cpu':
            #将入参全部升精度
            expandedRowIdx = expandedRowIdx.cpu()
            expertIdxOptional = expertIdxOptional.cpu()
            if expandedX.dtype in [torch.bfloat16,torch.float16]:
                expandedX = expandedX.cpu().to(torch.float32)
                if x1Optional is not None:
                    x1Optional = x1Optional.cpu().to(torch.float32)
                if x2Optional is not None:
                    x2Optional = x2Optional.cpu().to(torch.float32)
                if biasOptional is not None:
                    biasOptional = biasOptional.cpu().to(torch.float32)
                if scalesOptional is not None:
                    scalesOptional = scalesOptional.cpu().to(torch.float32)
            else:
                expandedX = expandedX.cpu().to(torch.float64)
                if x1Optional is not None:
                    x1Optional = x1Optional.cpu().to(torch.float64)
                if x2Optional is not None:
                    x2Optional = x2Optional.cpu().to(torch.float64)
                if biasOptional is not None:
                    biasOptional = biasOptional.cpu().to(torch.float64)
                if scalesOptional is not None:
                    scalesOptional = scalesOptional.cpu().to(torch.float64)            
            output = MoeFinalizeRoutingV2Golden(expandedX, expandedRowIdx, 
                                            x1Optional, x2Optional, 
                                            biasOptional, scalesOptional, 
                                            expertIdxOptional, dropPadMode)
            output = output.to(expandedX_dtype)

        if self.device == 'gpu':
            #保持精度
            output = MoeFinalizeRoutingV2Golden(expandedX.cuda(), expandedRowIdx.cuda(), 
                                            x1Optional.cuda(), x2Optional.cuda(), 
                                            biasOptional.cuda(), scalesOptional.cuda(), 
                                            expertIdxOptional.cuda(), dropPadMode)
            output = output.to(expandedX_dtype)

        return output

@register("function_pyaclnn_MoeFinalizeRoutingV2")
class PyAclnnMoeFinalizeRoutingV2(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def __init__(self,task_result:TaskResult,backend):
        super().__init__(task_result,backend)
        self.placeholder = None

    # 默认调用，可省略
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        #获取第一个tensor的size
        self.input_args,tmp= super().init_by_input_data(input_data)
        self_tensor = self.acl_tensor_to_torch(self.input_args[0])
        self.self_shape = self_tensor.size()
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
       
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
 
        if  2147483649 in self.self_shape or 0 in self.self_shape :
            for output_tensor in output:
                output_tensor.zero_()

        return output