# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize

"""
include_dirs=[np.get_include()]: 访问numpy头文件
"""

extensions = [
    Extension(
        "speedup",
        ["LRender/CyModule/speedup.pyx"],
        # define_macros=[("CYTHON_TRACE", "1")],
        include_dirs=[np.get_include()],
        # libraries=["m"],
    )
]

"""
annotate=True: 这会生成Cython注释文件，允许你查看Python和Cython代码的映射，用于性能分析和调试。
linetrace 启用了行级跟踪，用于生成性能分析数据
binding 允许生成Cython与C代码的绑定。
"""
setup(
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={"linetrace": True, "binding": True},
    ),
    script_args=["build_ext", "-b", "LRender/CyModule", "-t", "Compiled_Temps"],
)

"""
命令为: python setup.py build_ext --inplace
"""