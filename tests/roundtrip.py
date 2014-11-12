"""
   this is the standard tests for pfft-python

   Tests are performed on a 3d grid of [29, 30, 31].

   tested features are:

   regular transform (r2c + c2r, c2c)
   transposed in / out, 
   padded in / out, 
   destroy input,
   inplace transform 

   * for single-rank numpy aggrement test(single), run with

   [mpirun -np 1] python roundtrip.py

   * for multi-rank roundtrip tests, run with 
   
   mpirun -np n python roundtrip.py

   n can be any number. procmeshes tested are:
   np = [n], [1, n], [n, 1], [a, d], [d, a]
   where a * d == n and a d are closest to n** 0.5
   
"""
from mpi4py import MPI
from sys import path
import os.path
path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pfft import *
import traceback

def test_roundtrip_3d(procmesh, type, flags, inplace):
    if numpy.product(procmesh.np) > 1:
        single = False
    else:
        single = True

    Nmesh = [29, 30, 31]
#    Nmesh = [2, 3, 2]
    
    partition = Partition(type, Nmesh, procmesh, flags)
    for rank in range(MPI.COMM_WORLD.size):
        MPI.COMM_WORLD.barrier()
        if rank != procmesh.rank:
            continue
        print procmesh.rank, 'roundtrip test, np=', procmesh.np, 'Nmesh = ', Nmesh, 'inplace = ', inplace
        print repr(partition)

    buf1 = LocalBuffer(partition)
    if inplace:
        buf2 = buf1
    else:
        buf2 = LocalBuffer(partition)

    input = buf1.view_input() 
    output = buf2.view_output()
#    print 'output', output.shape
#    print 'input', input.shape
    forward = Plan(
            partition,
            Direction.PFFT_FORWARD, 
            buf1,
            buf2,
            type=type,
            flags=flags)
    if procmesh.rank == 0:
        print repr(forward)

    # find the inverse plan
    if type == Type.PFFT_R2C:
        btype = Type.PFFT_C2R
        bflags = flags
        # the following lines are just good looking
        # PFFT_PADDED_R2C and PFFT_PADDED_C2R
        # are identical
        bflags &= ~Flags.PFFT_PADDED_R2C
        bflags &= ~Flags.PFFT_PADDED_C2R
        if flags & Flags.PFFT_PADDED_R2C:
            bflags |= Flags.PFFT_PADDED_C2R

    elif type == Type.PFFT_C2C:
        btype = Type.PFFT_C2C
        bflags = flags
    else:
        raise Exception("only r2c and c2c roundtrip are tested")

    bflags &= ~Flags.PFFT_TRANSPOSED_IN
    bflags &= ~Flags.PFFT_TRANSPOSED_OUT
    if flags & Flags.PFFT_TRANSPOSED_IN:
        bflags |= Flags.PFFT_TRANSPOSED_OUT
    if flags & Flags.PFFT_TRANSPOSED_OUT:
        bflags |= Flags.PFFT_TRANSPOSED_IN


    backward = Plan(
            partition,
            Direction.PFFT_BACKWARD, 
            buf2,
            buf1,
            type=btype, 
            flags=bflags,
            )
    if procmesh.rank == 0:
        print repr(backward)

    i = numpy.array(buf1.buffer, copy=False)
    numpy.random.seed(9999)
    i[:] = numpy.random.normal(size=i.shape)
    original = input.copy()

    if single:
        if type == Type.PFFT_R2C:
            correct = numpy.fft.rfftn(original)
        elif type == Type.PFFT_C2C:
            correct = numpy.fft.fftn(original)

    original *= numpy.product(Nmesh) # fftw vs numpy 

    if not inplace:
        output[:] = 0

    forward.execute(buf1, buf2)

    if single:
        if False:
            print output.shape
            print correct.shape
            print output
            print correct
            print i

        r2cerr = numpy.abs(output - correct).std(dtype='f8')
        print repr(forward.type), "error = ", r2cerr
        i[:] = 0
        output[:] = correct

    if not inplace:
        input[:] = 0
    backward.execute(buf2, buf1)

    if input.size > 0:
        c2rerr = numpy.abs(original - input).std(dtype='f8')
    else:
        c2rerr = 0.0

    for rank in range(MPI.COMM_WORLD.size):
        MPI.COMM_WORLD.barrier()
        if rank != procmesh.rank:
            continue
        print rank, repr(backward.type), "error = ", c2rerr
        if False:
            print original
            print input
            print i
        MPI.COMM_WORLD.barrier()

    if single:
        assert (r2cerr < 1e-5)
    assert (c2rerr < 1e-5) 

if MPI.COMM_WORLD.size == 1: 
    nplist = [
            [1],
            [1, 1],
            ]
else:
    s = MPI.COMM_WORLD.size
    a = int(s ** 0.5)
    while a > 1:
        if s % a == 0:
            d = s // a
            break
        a = a - 1 
    nplist = [
            [s],
            [1, s],
            [s, 1],
            ]
    if a > 1:
        nplist += [
            [a, d],
            [d, a],
            ]

try:
    for np in nplist:
        procmesh = ProcMesh(np)
        for flags in [
            Flags.PFFT_ESTIMATE | Flags.PFFT_DESTROY_INPUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_PADDED_R2C | Flags.PFFT_DESTROY_INPUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_PADDED_R2C,
            Flags.PFFT_ESTIMATE | Flags.PFFT_TRANSPOSED_OUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_TRANSPOSED_OUT | Flags.PFFT_DESTROY_INPUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_PADDED_R2C | Flags.PFFT_TRANSPOSED_OUT,
            ]:
            test_roundtrip_3d(procmesh, Type.PFFT_R2C, flags, True)
            test_roundtrip_3d(procmesh, Type.PFFT_R2C, flags, False)
            test_roundtrip_3d(procmesh, Type.PFFT_C2C, flags, True)
            test_roundtrip_3d(procmesh, Type.PFFT_C2C, flags, False)
except Exception as e:
    print traceback.format_exc()
    MPI.COMM_WORLD.Abort()
