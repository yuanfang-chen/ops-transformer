import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import ascendc_extension

PACKAGE_NAME = 'torch_pfa'
VERSION = '0.0.1'

# Library paths
OPS_TRANSFORMER_LIB_PATH = '/usr/local/Ascend/8.5.0.alpha001/ops_transformer/built-in/op_impl/ai_core/tbe/op_api/lib/linux/x86_64'
CANN_LIB_PATH = '/usr/local/Ascend/latest/x86_64-linux/lib64'
CANN_STUB_PATH = '/usr/local/Ascend/latest/x86_64-linux/lib64/stub'

# All required library directories
extra_library_dirs = [
    OPS_TRANSFORMER_LIB_PATH,
    CANN_LIB_PATH,
    CANN_STUB_PATH
]

# All required libraries (including dependencies)
extra_libraries = [
    'opapi_transformer',
    'opapi'
]

# Link arguments to ensure proper linking order
extra_link_args = [
    '-L' + OPS_TRANSFORMER_LIB_PATH,
    '-L' + CANN_LIB_PATH,
    '-lopapi_transformer',
    '-lopapi'
]

# Runtime library directories for dynamic loading
runtime_library_dirs = [
    OPS_TRANSFORMER_LIB_PATH,
    CANN_LIB_PATH,
    CANN_STUB_PATH
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    ext_modules=[
        ascendc_extension(
            name=PACKAGE_NAME,
            sources=['torch_interface.cpp'],
            extra_library_dirs=extra_library_dirs,
            extra_libraries=extra_libraries,
            extra_link_args=extra_link_args,
            runtime_library_dirs=runtime_library_dirs,
            extra_compile_args=[]
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
