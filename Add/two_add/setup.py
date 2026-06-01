from setuptools import setup
from torch.utils import cpp_extension
setup(
    name='add2',  # 安装包名称
    ext_modules=[  # 扩展模块列表
        cpp_extension.CppExtension(
            'add2', ['add2_bind.cpp', 'add2.cu'] 
        )
    ],
    cmdclass={						       # 执行编译命令设置
        'build_ext': cpp_extension.BuildExtension
    }
)