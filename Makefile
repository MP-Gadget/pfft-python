MPICC=mpicc
all:
	CC=$(MPICC) python setup.py build
