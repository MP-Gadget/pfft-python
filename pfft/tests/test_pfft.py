from __future__ import absolute_import
import pfft

import numpy
from numpy.testing import assert_array_equal, assert_almost_equal


from runtests.mpi import MPITest
from mpi4py import MPI

def test_world():
    world = MPI.COMM_WORLD

    procmesh = pfft.ProcMesh(np=[world.size,], comm=world)
    assert procmesh.comm == world
    procmesh = pfft.ProcMesh(np=[world.size,], comm=None)
    assert procmesh.comm == world

    assert_array_equal(pfft.ProcMesh.split(2, None), pfft.ProcMesh.split(2, world))
    assert_array_equal(pfft.ProcMesh.split(1, None), pfft.ProcMesh.split(1, world))

@MPITest(3)
def test_edges(comm):
    procmesh = pfft.ProcMesh(np=[comm.size,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C,
            [4, 4], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT
        )

    assert_array_equal(partition.i_edges[0], [0, 2, 4, 4])
    assert_array_equal(partition.i_edges[1], [0, 4])

    assert_array_equal(partition.o_edges[1], [0, 2, 4, 4])
    assert_array_equal(partition.o_edges[0], [0, 4])

@MPITest(1)
def test_edges_padded(comm):
    procmesh = pfft.ProcMesh(np=[comm.size,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_R2C,
            [16, 8], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT |
            pfft.Flags.PFFT_PADDED_R2C
    )

    assert_array_equal(partition.i_edges[0], [0, 16])
    assert_array_equal(partition.i_edges[1], [0, 8])

    assert_array_equal(partition.o_edges[0], [0, 16])
    assert_array_equal(partition.o_edges[1], [0, 5])

@MPITest(3)
def test_nino(comm):
    procmesh = pfft.ProcMesh(np=[comm.size,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C,
            [4, 8], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT)

    assert_array_equal(partition.ni, [4, 8])
    assert_array_equal(partition.no, [4, 8])

@MPITest(1)
def test_transposed(comm):
    procmesh = pfft.ProcMesh(np=[1,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C,
            [4, 8], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT)

    buffer = pfft.LocalBuffer(partition)
    o = buffer.view_output()
    i = buffer.view_input()

    assert_array_equal(i.shape, (4, 8))
    assert_array_equal(i.strides, (128, 16))
    assert_array_equal(o.shape, (4, 8))
    assert_array_equal(o.strides, (16, 64))

    assert o.dtype == numpy.dtype('complex128')
    assert i.dtype == numpy.dtype('complex128')

@MPITest(1)
def test_padded(comm):
    procmesh = pfft.ProcMesh(np=[1,], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_R2C,
            [4, 8], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_PADDED_R2C)

    buffer = pfft.LocalBuffer(partition)
    i = buffer.view_input()
    o = buffer.view_output()

    assert_array_equal(i.shape, (4, 8))
    assert_array_equal(i.strides, (80, 8))
    assert_array_equal(o.shape, (4, 5))
    assert_array_equal(o.strides, (16, 64))

    assert i.dtype == numpy.dtype('float64')
    assert o.dtype == numpy.dtype('complex128')

@MPITest(1)
def test_correct_single(comm):
    procmesh = pfft.ProcMesh(np=[1], comm=comm)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, [2, 2],
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE)

    buffer1 = pfft.LocalBuffer(partition)
    buffer2 = pfft.LocalBuffer(partition)

    plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)
    buffer1.view_input()[:] = numpy.arange(4).reshape(2, 2)
    correct = numpy.fft.fftn(buffer1.view_input())
    plan.execute(buffer1, buffer2)

    assert_array_equal(correct, buffer2.view_output())

@MPITest(1)
def test_transpose_1d_decom(comm):
    procmesh = pfft.ProcMesh(np=[1,], comm=comm)
    N = (1, 2, 3, 4)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, N,
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE | pfft.Flags.PFFT_TRANSPOSED_OUT)

    buffer = pfft.LocalBuffer(partition)
    i = buffer.view_input()
    assert_array_equal(i.strides, [384, 192, 64, 16])
    o = buffer.view_output()
    assert_array_equal(o.strides, [192, 192, 64, 16])

@MPITest(1)
def test_transpose_2d_decom(comm):
    procmesh = pfft.ProcMesh(np=[1,1], comm=comm)
    N = (1, 2, 3, 4)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, N,
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE | pfft.Flags.PFFT_TRANSPOSED_OUT)

    buffer = pfft.LocalBuffer(partition)
    i = buffer.view_input()
    assert_array_equal(i.strides, [384, 192, 64, 16])
    o = buffer.view_output()
    assert_array_equal(o.strides, [64, 192, 64, 16])

@MPITest(1)
def test_transpose_3d_decom(comm):
    procmesh = pfft.ProcMesh(np=[1,1,1], comm=comm)
    N = (1, 2, 3, 4, 5)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, N,
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE | pfft.Flags.PFFT_TRANSPOSED_OUT)

    buffer = pfft.LocalBuffer(partition)
    #FIXME: check with @mpip if this is correct.
    i = buffer.view_input()
    assert_array_equal(i.strides, [1920, 960, 320, 80, 16])
    o = buffer.view_output()
    assert_array_equal(o.strides, [80, 960, 320, 80, 16])

@MPITest((1, 4))
def test_correct_multi(comm):
    procmesh = pfft.ProcMesh(np=[comm.size,], comm=comm)
    N = (2, 3)
    data = numpy.arange(numpy.prod(N), dtype='complex128').reshape(N)
    correct = numpy.fft.fftn(data)
    result = numpy.zeros_like(data)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, N,
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE)

    buffer1 = pfft.LocalBuffer(partition)
    buffer2 = pfft.LocalBuffer(partition)

    plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)

    buffer1.view_input()[:] = data[partition.local_i_slice]
    plan.execute(buffer1, buffer2)

    result[partition.local_o_slice] = buffer2.view_output()
    result = comm.allreduce(result)
    assert_almost_equal(correct, result)

@MPITest(commsize=1)
def test_leak(comm):
    for i in range(1024):
        procmesh = pfft.ProcMesh(np=[1,1], comm=comm)

        partition = pfft.Partition(pfft.Type.PFFT_C2C,
                [128, 128, 128], procmesh,
                pfft.Flags.PFFT_TRANSPOSED_OUT)

        buffer = pfft.LocalBuffer(partition)
        #FIXME: check with @mpip if this is correct.
        i = buffer.view_input()

@MPITest([4])
def test_2d_on_2d_c2c(comm):
    procmesh = pfft.ProcMesh(np=[2, 2], comm=comm)
    N = (8, 8)

    data = numpy.arange(numpy.prod(N), dtype='complex128').reshape(N)

    correct = numpy.fft.fftn(data.copy())
    result = numpy.zeros_like(correct)

    partition = pfft.Partition(pfft.Type.PFFT_C2C, N,
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE
          | pfft.Flags.PFFT_TRANSPOSED_OUT
#          | pfft.Flags.PFFT_DESTROY_INPUT
          | pfft.Flags.PFFT_PRESERVE_INPUT
        )

    buffer1 = pfft.LocalBuffer(partition)
    buffer2 = pfft.LocalBuffer(partition)

    plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)

    buffer1.view_input()[:] = data[partition.local_i_slice]
    plan.execute(buffer1, buffer2)

    result[partition.local_o_slice] = buffer2.view_output()
    result = comm.allreduce(result)
    assert_almost_equal(correct, result)

@MPITest([1, 4])
def test_2d_on_2d_r2c(comm):
    if comm.size == 1:
        procmesh = pfft.ProcMesh(np=[1, 1], comm=comm)
    else:
        procmesh = pfft.ProcMesh(np=[2, 2], comm=comm)
    N = (8, 8)

    data = numpy.arange(numpy.prod(N), dtype='f8').reshape(N)

    correct = numpy.fft.rfftn(data.copy())
    result = numpy.zeros_like(correct)

    partition = pfft.Partition(pfft.Type.PFFT_R2C, N,
        procmesh, flags=pfft.Flags.PFFT_ESTIMATE
          | pfft.Flags.PFFT_TRANSPOSED_OUT
#          | pfft.Flags.PFFT_DESTROY_INPUT
          | pfft.Flags.PFFT_PRESERVE_INPUT
#          | pfft.Flags.PADDED_R2C # doesn't work yet
        )

    buffer1 = pfft.LocalBuffer(partition)
    buffer2 = pfft.LocalBuffer(partition)

    plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)

    buffer1.view_input()[:] = data[partition.local_i_slice]
    plan.execute(buffer1, buffer2)

    result[partition.local_o_slice] = buffer2.view_output()
    result = comm.allreduce(result)
    assert_almost_equal(correct, result)
