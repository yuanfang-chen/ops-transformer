# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.

import importlib
import os

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
TORCH_NPU_PATH = os.path.dirname(importlib.import_module('torch_npu').__file__)

setup(
    name='mhc_res_ext',
    ext_modules=[
        CppExtension(
            name='mhc_res_ext',
            sources=['csrc/mhc_res_torch.cpp'],
            include_dirs=[
                os.path.join(ASCEND_HOME, 'include'),
                os.path.join(TORCH_NPU_PATH, 'include'),
            ],
            library_dirs=[
                os.path.join(os.path.dirname(__file__), 'build/lib'),
                os.path.join(ASCEND_HOME, 'lib64'),
                os.path.join(TORCH_NPU_PATH, 'lib'),
            ],
            libraries=['mhc_res_kernel', 'ascendcl', 'torch_npu'],
            extra_compile_args=[
                '-D__FILENAME__="mhc_res_torch.cpp"',
            ],
            runtime_library_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build/lib'),
                os.path.join(TORCH_NPU_PATH, 'lib'),
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
