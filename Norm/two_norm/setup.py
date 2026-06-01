from setuptools import setup
from torch.utils import cpp_extension
setup(
    name='norm',  # 安装包名称
    ext_modules=[  # 扩展模块列表
        cpp_extension.CppExtension(
            'norm', ['two_norm_bind.cpp', 'two_norm_kernel.cu'] 
        )
    ],
    cmdclass={						       # 执行编译命令设置
        'build_ext': cpp_extension.BuildExtension
    }
)