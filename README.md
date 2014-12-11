pfft-python
===========

Python binding of PFFT. (github.com/mpip/pfft)

PFFT is a massively parallel Fast Fourier Transform library. For its
performance, see:

https://www-user.tu-chemnitz.de/~mpip/software.php.en

This is the python binding of PFFT.

It requires mpi4py for installation. The mpicc configuration of mpi4py
is used.

To install:

```
    python setup.py install --user
```

PFFT, patched FFTW, and the binding are linked into one giant (6MB) shared
object files. Some systems may not like it much.

To use:

  see example.py

  also see tests/roundtrip.py


Current ISSUES:
  seems to be not broken on Cray XE Edison.nersc.gov.

