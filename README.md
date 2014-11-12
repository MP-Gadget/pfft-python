pfft-python
=====

Python binding of PFFT. (github.com/mpip/pfft)

To install:

```
    make dep-fftw
    make dep-pfft
    make 
    mpirun -np 1 python tests/roundtrip.py
    mpirun -np 9 python tests/roundtrip.py
    make install
```

To use:

  see example.py

  also see tests/roundtrip.py
