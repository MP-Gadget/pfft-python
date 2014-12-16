from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension
import os
import os.path
import numpy
import mpi4py

package_basedir = os.path.abspath(os.path.dirname(__file__))
dependsdir = os.path.join(package_basedir, 'build', 'depends')
compiler = mpi4py.get_config()['mpicc']
# how otherwise do I set the compiler cython uses?
os.environ['CC'] = compiler
os.environ['LDSHARED'] = compiler + " -shared"
print mpi4py.get_include()
def build_fftw():
    line = ('CFLAGS="$CFLAGS -fPIC -fvisibility=hidden -I%s/include" ' % dependsdir +
            'LDFLAGS="$LDFLAGS -L%s/lib" ' % dependsdir +
            'MPICC="%s" ' % compiler +
            'CC="%s" ' % compiler +
            'sh depends/install_fftw.sh ' +
             dependsdir)
    if os.path.exists(os.path.join(dependsdir, 
        'lib', 'libfftw3.a')):
        return
    ret=os.system(line)
    if ret != 0:
        raise ValueError("could not build fftw")

def build_pfft():
    line = ('CFLAGS="$CFLAGS -fPIC -fvisibility=hidden -I%s/include" ' % dependsdir +
            'LDFLAGS="$LDFLAGS -L%s/lib" ' % dependsdir +
            'MPICC="%s" ' % compiler +
            'CC="%s" ' % compiler +
            'sh depends/install_pfft.sh ' +
             dependsdir)
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
        library_dirs=[
            os.path.join(dependsdir, 'lib'),
        ],
        libraries=["pfft", "fftw3_mpi", "fftw3"])

extensions = [
        myext("pfft.core", ["pfft/core.pyx"]),
        ]

build_fftw()
build_pfft()
setup(
    name="pfft-python", version="0.1",
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    description="python binding of PFFT, a massively parallel FFT library",
    url="http://github.com/rainwoodman/pfft-python",
    #package_dir = {'pfft': 'pfft'},
    zip_safe=False,
    install_requires=['cython', 'numpy'],
    packages= ['pfft'],
    requires=['numpy'],
    ext_modules = cythonize(extensions,
        include_path=[mpi4py.get_include()]),
)

