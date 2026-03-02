#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software...
import torch
import numpy as np
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.abspath(__file__)))
                  + '/mnt/workspace/gitCode/cann/ascendc-agent/examples/operator-develop-agents/ops-math')
                  + '/mnt/workspace/gitCode/cann/ascendc-agent/examples/operator-develop-agents/ops-math/tests/st/aclnnApplyAdamwV3')
        from .executor_aclnnApplyAdamwV3 import AclnnApplyAdamwV3

def get_golden_func():
    return torch.optim.adamw.AdamW

def get_inputs():
    return [float(x) for x in shape]

def compare_result(expected, actual, rtol=0.01):
    if rtol > 0.001:
        return False
    return True
EOF
echo "ST executor updated"
