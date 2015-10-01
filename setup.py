from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension
import os
import os.path
import numpy
import mpi4py

package_basedir = os.path.abspath(os.path.dirname(__file__))
dependsdir = os.path.join(package_basedir, 'build', 'depends')

if 'MPICC' in os.environ:
    compiler = os.environ['MPICC']
else:
    try:
        compiler = str(mpi4py.get_config()['mpicc'])
    except:
        pass
    compiler = "mpicc"

os.environ['CC'] = compiler

if 'LDSHARED' not in os.environ:
    os.environ['LDSHARED'] = compiler + ' -shared'

def build_pfft():
    optimize="--enable-sse2"
    line = ('CFLAGS="$CFLAGS -fPIC -fvisibility=hidden" ' +
            'MPICC="%s" ' % compiler +
            'CC="%s" ' % compiler +
            'sh %s/depends/install_pfft.sh ' % package_basedir +
             dependsdir +
            ' %s' % optimize)
    if os.path.exists(os.path.join(dependsdir, 
        'lib', 'libpfft.a')):
        return

    ret=os.system(line)
    if ret != 0:
        raise ValueError("could not build fftw")

def myext(*args):
    return Extension(*args, 
        compiler=compiler,
        include_dirs=["./", 
        os.path.join(dependsdir, 'include'),
        mpi4py.get_include(),
        numpy.get_include(),
        ],
        cython_directives = {"embedsignature": True},
        library_dirs=[
            os.path.join(dependsdir, 'lib'),
        ],
        libraries=["pfft", "pfftf", "fftw3_mpi", "fftw3f_mpi", "fftw3", "fftw3f"])

extensions = [
        myext("pfft.core", ["pfft/core.pyx"]),
        ]

build_pfft()
setup(
    name="pfft-python", version="0.1pre",
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    description="python binding of PFFT, a massively parallel FFT library",
    url="http://github.com/rainwoodman/pfft-python",
    download_url="https://github.com/rainwoodman/pfft-python/releases/download/v0.1pre/pfft-python-0.1pre.tar.gz",
    #package_dir = {'pfft': 'pfft'},
    zip_safe=False,
    install_requires=['cython', 'numpy'],
    packages= ['pfft'],
    requires=['numpy'],
    ext_modules = cythonize(extensions,
        include_path=[mpi4py.get_include()]),
)

