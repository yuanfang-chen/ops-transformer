# MIT License
#
# Copyright (c) 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
