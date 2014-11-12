MPICC=mpicc -g -O0 
LDSHARED=$(MPICC) -shared
CFLAGS=-fPIC -I$(PWD)/depends/include
LDFLAGS=-L$(PWD)/depends/lib
all:
	LDSHARED="$(LDSHARED)" LDFLAGS="$(LDFLAGS)" CC="$(MPICC)" python setup.py build_ext --inplace
dep-fftw:
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" sh depends/install_fftw.sh $(PWD)/depends/
dep-pfft:
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" sh depends/install_pfft.sh $(PWD)/depends/
