from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extension_mod = CUDAExtension(
        name='fastAgg', 
        sources=[   
                    'cuda_operators/operators.cpp', 
                    'cuda_operators/aggregate_acc_kernel.cu',
                    'cuda_operators/aggregate_shd_kernel.cu',
                    'cuda_operators/edge_softmax.cu',
                    'cuda_operators/fused_attention.cu',
                    'cuda_operators/utils.cu'
                ],
        extra_compile_args={'cxx': ['-g','-O3'],
                                'nvcc': ['-g','-O3','-lineinfo']},
        extra_link_args=['-lcurand']
        )

# Specify the CUDA architecture flags
CUDA_ARCH_FLAGS = ['-gencode', 'arch=compute_80,code=sm_80']
# extension_mod.extra_compile_args['nvcc'] = CUDA_ARCH_FLAGS
extension_mod.extra_compile_args['nvcc'].extend(CUDA_ARCH_FLAGS)

setup(
    name='fastAgg',
    version='0.1.0',
    ext_modules=[
        extension_mod
    ],
    cmdclass={
        'build_ext': BuildExtension
    })