#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
jacobi_molotilka
"""
import tbetoolkits
from .registry import register_golden
import numpy as np
@register_golden(["jacobi_molotilka"])
def _jacobi_molotilka(context: "tbetoolkits.UniversalTestcaseStructure"):
    import torch
   
    input_a = context.input_arrays[0]
    print("jacobi_molotilka TEST GENERATED!!!!! A.shape is {0}".format(input_a.shape))
    dtype = input_a.dtype
    print("A:[{0}, {1}, {2}]".format(input_a[0,0,0], input_a[0,0,1], input_a[0,0,2]))
    if "float32" in str(dtype):
        input_a = input_a.astype("float32")
    a = torch.from_numpy(input_a)
    print("bla bla bla")
    u_gold, s_gold, vh_gold = torch.svd(a)
    # u_gold = torch.transpose(u_gold, 1, 2)
    # vh_gold = torch.transpose(vh_gold, 1, 2)
    # if "bfloat16" in str(dtype):
    #     res = res.astype(dtype)
    print("s_gold has shape {0} and type {1}".format(s_gold.shape, s_gold.dtype))
    print("u_gold has shape {0} and type {1}".format(u_gold.shape, u_gold.dtype))
    print("vh_gold has shape {0} and type {1}".format(vh_gold.shape, vh_gold.dtype))

    res=(s_gold.numpy(), u_gold.numpy(), vh_gold.numpy())
    return res