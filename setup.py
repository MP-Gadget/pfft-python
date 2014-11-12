from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension
import numpy
import mpi4py

def myext(*args):
    return Extension(*args, 
        compiler=mpi4py.get_config()['mpicc'],
        include_dirs=["./", 
        mpi4py.get_include(),
        numpy.get_include(),
        ],
        libraries=["fftw3", "fftw3_mpi", "pfft"])

extensions = [
        myext("pfft.core", ["src/core.pyx"]),
        myext("pfft.failure", ["src/failure.pyx"]),
        ]

setup(
    name="pfft-python", version="0.1",
    author="Yu Feng",
    description="python binding of PFFT, a massively parallel FFT library",
    package_dir = {'pfft': 'src'},
    install_requires=['cython', 'numpy'],
    packages= ['pfft'],
    requires=['numpy'],
    ext_modules = cythonize(extensions,
        include_path=[mpi4py.get_include()])
)

