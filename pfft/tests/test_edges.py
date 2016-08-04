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
    procmesh = pfft.ProcMesh(np=[3,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C,
            [4, 4], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT)

    assert_array_equal(partition.i_edges[0], [0, 2, 4, 4])
    assert_array_equal(partition.i_edges[1], [0, 4])

@MPIWorld(NTask=1)
def test_correct_single(comm):
    procmesh = pfft.ProcMesh(np=[1,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, [2, 2],
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE)

    buffer1 = pfft.LocalBuffer(partition)
    buffer2 = pfft.LocalBuffer(partition)

    plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)
    buffer1.view_input()[:] = numpy.arange(4).reshape(2, 2)
    correct = numpy.fft.fftn(buffer1.view_input())
    plan.execute(buffer1, buffer2)

    assert_array_equal(correct, buffer2.view_output())

@MPIWorld(NTask=4)
def test_correct_multi(comm):
    procmesh = pfft.ProcMesh(np=[4,], comm=comm)

    data = numpy.arange(16, dtype='complex128').reshape(4, 4)
    correct = numpy.fft.fftn(data)
    result = numpy.zeros_like(data)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, [4, 4],
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE)

    buffer1 = pfft.LocalBuffer(partition)
    buffer2 = pfft.LocalBuffer(partition)

    plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)

    buffer1.view_input()[:] = data[partition.local_i_slice]
    plan.execute(buffer1, buffer2)

    result[partition.local_o_slice] = buffer2.view_output()
    result = comm.allreduce(result)

    assert_array_equal(correct, result)
