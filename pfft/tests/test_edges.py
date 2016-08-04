from __future__ import absolute_import
from mpi4py import MPI

import pfft

import numpy
from numpy.testing import assert_array_equal
from numpy.testing.decorators import skipif
from numpy.testing.decorators import knownfailureif

def MPIWorld(NTask):
    if MPI.COMM_WORLD.size < NTask:
        return knownfailureif(True, "Test Failed because the world is too small")

    color = MPI.COMM_WORLD.rank >= NTask
    comm = MPI.COMM_WORLD.Split(color)

    if(color > 0):
        return skipif(True, "Idling ranks")

    def dec(func):
        def wrapped(*args, **kwargs):
            kwargs['comm'] = comm
            return func(*args, **kwargs)
        wrapped.__name__ = func.__name__
        return wrapped
    return dec

@MPIWorld(NTask=3)
def test_edges(comm):
    print('test_edges', MPI.COMM_WORLD.rank)
    procmesh = pfft.ProcMesh(np=[3,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C,
            [4, 4], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT)

    assert_array_equal(partition.i_edges[0], [0, 2, 4, 4])
    assert_array_equal(partition.i_edges[1], [0, 4])

