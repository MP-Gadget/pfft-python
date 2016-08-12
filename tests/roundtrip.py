"""
   This is the standard tests for pfft-python.

   Roundtrip (Backward + Forward) tests are performed on a 3d grid,
   given by -Nmesh. Default is [29, 30, 31].
   Tested features are:
       regular transform (r2c + c2r, c2c)
       transposed in / out, 
       padded in / out, 
       destroy input,
       inplace transform 

   Examples:
   * for single-rank numpy agreement test, run with
       mpirun -np 1 python roundtrip.py -Nmesh 32 32 32 -Nmesh 3 3 3 -tree -verbose

   * for multi-rank tests, run with 
       mpirun -np n python roundtrip.py -Nmesh 32 32 32 -Nmesh 3 3 3 -tree -verbose

   n can be any number. procmeshes tested are:
       np = [n], [1, n], [n, 1], [a, d], [d, a]
    where a * d == n and a d are closest to n** 0.5
"""
from mpi4py import MPI
import itertools
import traceback
import numpy
import argparse

import os.path
from sys import path

parser = argparse.ArgumentParser(description='Roundtrip testing of pfft', 
        epilog=__doc__,
       formatter_class=argparse.RawDescriptionHelpFormatter 
        )

parser.add_argument('-Nmesh', nargs=3, type=int,
        action='append', metavar=('Nx', 'Ny', 'Nz'), 
        help='size of FFT mesh, default is 29 30 31',
        default=[])
parser.add_argument('-Nproc', nargs=2, type=int,
        action='append', metavar=('Nx', 'Ny'), 
        help='proc mesh',
        default=[])
parser.add_argument('-tree', action='store_true', default=False,
        help='Use pfft from source tree, ' +
        'built with setup.py build_ext --inplace')
parser.add_argument('-diag', action='store_true', default=False,
        help='show which one failed and which one passed')
parser.add_argument('-verbose', action='store_true', default=False,
        help='print which test will be ran')

ns = parser.parse_args()
Nmesh = ns.Nmesh
if len(Nmesh) == 0:
    # default 
    Nmesh = [[29, 30, 31]]
if ns.tree:
    # prefers to use the locally built pfft in source tree, in case there is an
    # installation
    path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pfft import *

class LargeError(Exception):
    pass

def test_roundtrip_3d(procmesh, type, flags, inplace, Nmesh):
    partition = Partition(type, Nmesh, procmesh, flags)
    for rank in range(MPI.COMM_WORLD.size):
        MPI.COMM_WORLD.barrier()
        if rank != procmesh.rank:
            continue
        #print procmesh.rank, 'roundtrip test, np=', procmesh.np, 'Nmesh = ', Nmesh, 'inplace = ', inplace
        #print repr(partition)

    buf1 = LocalBuffer(partition)
    if inplace:
        buf2 = buf1
    else:
        buf2 = LocalBuffer(partition)

    input = buf1.view_input() 
    output = buf2.view_output()

    assert input.base == buf1
    assert output.base == buf2

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
        #print repr(forward)
        pass

    # find the inverse plan
    typemap = {
        Type.PFFT_R2C: Type.PFFT_C2R,
        Type.PFFT_C2C: Type.PFFT_C2C,
        Type.PFFTF_R2C: Type.PFFTF_C2R,
        Type.PFFTF_C2C: Type.PFFTF_C2C
    }
    btype = typemap[type]
    if type == Type.PFFT_R2C or type == Type.PFFTF_R2C:
        bflags = flags
        # the following lines are just good looking
        # PFFT_PADDED_R2C and PFFT_PADDED_C2R
        # are identical
        bflags &= ~Flags.PFFT_PADDED_R2C
        bflags &= ~Flags.PFFT_PADDED_C2R
        if flags & Flags.PFFT_PADDED_R2C:
            bflags |= Flags.PFFT_PADDED_C2R

    elif type == Type.PFFT_C2C or type == Type.PFFTF_C2C:
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
        #print repr(backward)
        pass

    numpy.random.seed(9999)

    fullinput = numpy.random.normal(size=Nmesh)
    if type == Type.PFFT_R2C or type == Type.PFFTF_R2C:
        correct = numpy.fft.rfftn(fullinput)
    elif type == Type.PFFT_C2C or type == Type.PFFTF_C2C:
        correct = numpy.fft.fftn(fullinput)


    input[:] = fullinput[partition.local_i_slice]
    correct = correct[partition.local_o_slice].copy()

    original = input.copy()

    if not inplace:
        output[:] = 0

    forward.execute(buf1, buf2)

    if output.size > 0:
        r2cerr = numpy.abs(output - correct).max()
    else:
        r2cerr = 0.0
    # now test the backward transformation
    input[:] = 0
    output[:] = correct

    backward.execute(buf2, buf1)

    if input.size > 0:
        input[:] /= numpy.product(Nmesh)
        # some distributions have no input value
        c2rerr = numpy.abs(original - input).max()
    else:
        c2rerr = 0.0

    for rank in range(MPI.COMM_WORLD.size):
        MPI.COMM_WORLD.barrier()
        if rank != procmesh.rank:
            continue
        if False:
            print('error', original - input)
        MPI.COMM_WORLD.barrier()
    if False:
        print(repr(forward.type), 'forward', "error = ", r2cerr)
        print(repr(forward.type), 'backward', "error = ", c2rerr)

    r2cerr = MPI.COMM_WORLD.allreduce(r2cerr, MPI.MAX)
    c2rerr = MPI.COMM_WORLD.allreduce(c2rerr, MPI.MAX)
    if (r2cerr > 5e-4):
        raise LargeError("r2c: %g" % r2cerr)

    if (c2rerr > 5e-4):
        raise LargeError("c2r: %g" % c2rerr)

if MPI.COMM_WORLD.size == 1: 
    nplist = [
            [1],
            [1, 1],
            ]
else:
    nplist = ns.Nproc
            

try:
    flags = [
            Flags.PFFT_ESTIMATE | Flags.PFFT_DESTROY_INPUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_PADDED_R2C | Flags.PFFT_DESTROY_INPUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_PADDED_R2C,
            Flags.PFFT_ESTIMATE | Flags.PFFT_TRANSPOSED_OUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_TRANSPOSED_OUT | Flags.PFFT_DESTROY_INPUT,
            Flags.PFFT_ESTIMATE | Flags.PFFT_PADDED_R2C | Flags.PFFT_TRANSPOSED_OUT,
            ]
    params = list(itertools.product(
            nplist, [Type.PFFT_C2C, Type.PFFT_R2C, Type.PFFTF_C2C, Type.PFFTF_R2C], flags, [True, False],
            Nmesh,
            ))

    PASS = []
    FAIL = []
    for param in params:
        if MPI.COMM_WORLD.rank == 0:
            if ns.verbose:
                f = param
                print("NP", f[0], repr(Type(f[1])), repr(Flags(f[2])), "InPlace", f[3], "Nmesh", f[4])
        np = param[0]
        procmesh = ProcMesh(np=np)
        try:
            test_roundtrip_3d(procmesh, *(param[1:]))
            PASS.append(param)
        except LargeError as e:
            if ns.verbose:
                f = param
                print("Failed", e)
            FAIL.append((param, e))

    if MPI.COMM_WORLD.rank == 0:
        print("PASS", len(PASS), '/', len(params))
        if ns.diag:
            for f in PASS:
                print("NP", f[0], repr(Type(f[1])), repr(Flags(f[2])), "InPlace", f[3], "Nmesh", f[4])
        print("FAIL", len(FAIL), '/', len(params))
        if ns.diag:
            for f, e in FAIL:
                print("NP", f[0], repr(Type(f[1])), repr(Flags(f[2])), "InPlace", f[3], "Nmesh", f[4], e)
        assert len(FAIL) == 0
except Exception as e:
    print(traceback.format_exc())
    MPI.COMM_WORLD.Abort()
