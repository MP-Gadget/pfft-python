pfft-python
===========

Python binding of PFFT. (github.com/mpip/pfft)

PFFT is a massively parallel Fast Fourier Transform library. For its
performance, see:

    https://www-user.tu-chemnitz.de/~potts/workgroup/pippig/software.php.en

This is the python binding of PFFT. 
The API document is at

    http://rainwoodman.github.io/pfft-python/index.html

.. image:: https://api.travis-ci.org/rainwoodman/pfft-python.svg
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/pfft-python/


DOI of pfft-python:

.. image:: https://zenodo.org/badge/26140163.svg
   :target: https://zenodo.org/badge/latestdoi/26140163
   
PFFT is a FFT library with excellent scaling at large number of processors.
We have been routinely running 10,000 ** 3 transforms on 81,000 MPI ranks as 
a component of the BlueTides simulation at National Center for Supercomputing
Applications. This is beyond our knowledge of the limits of FFTW.

This Python binding of course cannot yet operate at such a large scale. Due
to the limitations of Python packaging and moduling system. 
We nevertheless feel it is important to develop a python binding of PFFT to
allow early exploration of a migration into scripting languages in super computing.


For example, we have build a particle-mesh solver at

    http://github.com/rainwoodman/pmesh

For some leverage of the python import problem, see `python-mpi-bcast` at 

    http://github.com/rainwoodman/python-mpi-bcast

pfft-python requires mpi4py for installation. 

To install from PyPI:

.. code::

    pip --user pfft-python

To install from git source

.. code::

    python setup.py install --user

PFFT, patched FFTW, and the binding are linked into one giant (6MB) shared
object file.  We use `-fvisibility=hidden` to hide the PFFT/FFTW symbols.

For Macs with Anaconda, due to this bug https://github.com/conda/conda/issues/2277
one needs to make a symlink from the anaconda installation directory to
/opt/anaconda1anaconda2anaconda3 .

The mental model of PFFT is similar to FFTW. We plan ahead such that the code
runs and runs fast. 4 objects are involved in a FFT:

- ProcMesh : The topology / geometry of the MPI ranks. For example 4x2 or 2x4 for 8
  ranks, or 250x200 for 500000 ranks.

- Partition : The partition of the FFT input / output array onto the ranks.
  local_i_slice, local_i_start, local_ni describes the relative offset
  of the input. replacing 'i' with 'o' for the output.

- LocalBuffer : The place holder of the local data storage (allocated by PFFT).
  use view_input() view_output() to obtain the correct numpy array of the
  correct shape and strides suited for either the input or the output.
  -- always indexed in (x, y, z) ordering.

- Plan : The PFFT plan. execute the plan to obtain the results in the output array.

A fairly complex example (testing agreement with numpy.fft) is at tests/roundtrip.py .
A simpler example is example.py.

The documentation is sparse and in the source code (pfft/core.pyx), 
hopefully the guide here can get you started:

1. create a ProcMesh object for the communication geometry

2. create a Partition object for the data partition of the FFT mesh,
   in real and fourier space, both

3. allocate LocalBuffer objects for input and output. A LocalBuffer can be
   reused for inplace transforms. 

4. create Plan objects for the transforms, with the LocalBuffer objects as
   scratch

5. optionally, free the scratch buffers, and create new LocalBuffer objects.

6. view the LocalBuffer objects via view_input / view_output 

7. fill the LocalBuffer objects, making use of 
   Partition.local_i_start, local_o_start which marks the offset of the local
   mesh.
   A useful function is numpy.indices. numpy.meshgrid and numpy.ogrid are also useful.

8. Apply the plans via Plan.execute with LocalBuffer objects as arguments.


Yu Feng
