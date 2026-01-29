import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import ascendc_extension

PACKAGE_NAME = 'torch_pfa'
VERSION = '0.0.1'

# Libararies and Includes paths
OPS_TRANSFORMER_LIB_PATH = '/usr/local/Ascend/8.5.0.alpha001/ops_transformer/built-in/op_impl/ai_core/tbe/op_api/lib/linux/x86_64'
OPS_TRANSFORMER_INC_PATH = '/usr/local/Ascend/8.5.0.alpha001/ops_transformer/include'
CANN_LIB_PATH = '/usr/local/Ascend/latest/x86_64-linux/lib64'

# Link arguments to ensure proper linking order
extra_link_args = [
    f'{OPS_TRANSFORMER_LIB_PATH}/libopapi_transformer.so',  # Direct path
    f'-L{CANN_LIB_PATH}',
    f'-Wl,-rpath,{OPS_TRANSFORMER_LIB_PATH}',
    f'-Wl,-rpath,{CANN_LIB_PATH}',
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    ext_modules=[
        ascendc_extension(
            name=PACKAGE_NAME,
            sources=['torch_interface.cpp'],
            extra_include_dirs=[OPS_TRANSFORMER_INC_PATH],
            extra_library_dirs=[OPS_TRANSFORMER_LIB_PATH, CANN_LIB_PATH],
            extra_libraries=['opapi_transformer','opapi'],
            extra_link_args=extra_link_args,
            runtime_library_dirs=[OPS_TRANSFORMER_LIB_PATH, CANN_LIB_PATH],
            extra_compile_args=[]
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
