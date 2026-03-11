# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import ascendc_extension

PACKAGE_NAME = 'torch_pfa'
VERSION = '0.0.1'
PFA_LIB = os.getenv('PFA_LIB', '')  # 2 options: 'custom'-restricted package containintg only PFA kernel; 'ops_transformer' - full ops_transformer package

# Libararies and Includes paths\
if PFA_LIB == 'ops_transformer':
    OPS_TRANSFORMER_LIB_PATH = os.getenv("ASCEND_TOOLKIT_HOME", "") + '/ops_transformer/lib64'
    OPS_TRANSFORMER_INC_PATH = os.getenv("ASCEND_TOOLKIT_HOME", "") + '/ops_transformer/include'
    OPS_TRANSFORMER_LIB_NAME = 'opapi_transformer'
elif PFA_LIB == 'custom':
    OPS_TRANSFORMER_LIB_PATH = os.getenv("ASCEND_OPP_PATH", "") + '/vendors/custom_transformer/op_api/lib'
    OPS_TRANSFORMER_INC_PATH = os.getenv("ASCEND_OPP_PATH", "") + '/vendors/custom_transformer/op_api/include'
    OPS_TRANSFORMER_LIB_NAME = 'cust_opapi'
else:
    raise ValueError(f'wrong PFA_LIB option given {PFA_LIB} supported only: custom or ops_transformer')

CANN_LIB_PATH = '/usr/local/Ascend/latest/lib64'

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    ext_modules=[
        ascendc_extension(
            name=PACKAGE_NAME,
            sources=['torch_interface.cpp'],
            extra_include_dirs=[OPS_TRANSFORMER_INC_PATH],
            extra_library_dirs=[OPS_TRANSFORMER_LIB_PATH, CANN_LIB_PATH],
            extra_libraries=[OPS_TRANSFORMER_LIB_NAME, 'opapi'],
            extra_link_args=[f'{OPS_TRANSFORMER_LIB_PATH}/lib{OPS_TRANSFORMER_LIB_NAME}.so', 
                             f'-Wl,-rpath,{OPS_TRANSFORMER_LIB_PATH}'],
            runtime_library_dirs=[OPS_TRANSFORMER_LIB_PATH, CANN_LIB_PATH],
            extra_compile_args=[]
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
