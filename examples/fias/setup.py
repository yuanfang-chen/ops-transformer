import os
import subprocess
import sys
import shutil
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_type = "Debug" if self.debug else "Release"
        build_temp = os.path.join(self.build_temp, ext.name)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]

        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."], cwd=build_temp
        )

        so_files = list(Path(extdir).glob("*.so"))
        if so_files:
            src_so = so_files[0]
            dst_so = self.get_ext_fullpath(ext.name)
            if str(src_so.resolve()) != str(Path(dst_so).resolve()):
                shutil.copy(str(src_so), dst_so)

            if self.inplace:
                inplace_so = os.path.join(os.getcwd(), os.path.basename(dst_so))
                shutil.copy(dst_so, inplace_so)


class PostInstall(install):
    def run(self):
        install.run(self)


setup(
    name="fias",
    version="1.0.0",
    author="Huawei Technologies Co., Ltd.",
    description="Fused Infer Attention Score operator for Ascend NPU",
    ext_modules=[CMakeExtension("ascendc_ops", sourcedir=".")],
    cmdclass={
        "build_ext": CMakeBuild,
        "install": PostInstall,
    },
    zip_safe=False,
    python_requires=">=3.8",
)
