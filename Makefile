MPICC=mpicc -g -O0 
LDSHARED=$(MPICC) -shared
CFLAGS=-fPIC -Idepends/include
LDFLAGS=-Ldepends/lib
all:
	LDSHARED="$(LDSHARED)" LDFLAGS="$(LDFLAGS)" CC="$(MPICC)" python setup.py build_ext --inplace
fftw:
	CFLAGS="$(CFLAGS)" sh depends/install_fftw-3.3.3_gcc.sh $(PWD)/depends/
