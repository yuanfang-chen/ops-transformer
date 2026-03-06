import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import ascendc_extension

PACKAGE_NAME = 'torch_pfa'
VERSION = '0.0.1'
PFA_LIB = os.getenv('PFA_LIB','')  # 2 options: 'custom'-restricted package containintg only PFA kernel; 'ops_transformer' - full ops_transformer package

# Libararies and Includes paths\
if PFA_LIB == 'ops_transformer':
    OPS_TRANSFORMER_LIB_PATH = '/usr/local/Ascend/8.5.0.alpha001/ops_transformer/lib64'
    OPS_TRANSFORMER_INC_PATH = '/usr/local/Ascend/8.5.0.alpha001/ops_transformer/include'
    OPS_TRANSFORMER_LIB_NAME = 'opapi_transformer'
elif PFA_LIB == 'custom':
    OPS_TRANSFORMER_LIB_PATH = '/usr/local/Ascend/latest/opp/vendors/custom_transformer/op_api/lib'
    OPS_TRANSFORMER_INC_PATH = '/usr/local/Ascend/latest/opp/vendors/custom_transformer/op_api/include'
    OPS_TRANSFORMER_LIB_NAME = 'cust_opapi'
else:
    assert False, 'wrong PFA_LIB option'

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
            extra_libraries=[OPS_TRANSFORMER_LIB_NAME,'opapi'],
            extra_link_args=[f'{OPS_TRANSFORMER_LIB_PATH}/lib{OPS_TRANSFORMER_LIB_NAME}.so', 
                             f'-Wl,-rpath,{OPS_TRANSFORMER_LIB_PATH}'],
            runtime_library_dirs=[OPS_TRANSFORMER_LIB_PATH, CANN_LIB_PATH],
            extra_compile_args=[]
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
