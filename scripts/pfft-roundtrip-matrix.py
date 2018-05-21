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

   * to run in source code, first get a shell with
       python runtests.py --shell

   * for single-rank numpy agreement test, run with
       mpirun -np 1 python roundtrip.py -Nmesh 32 32 32 -Nmesh 3 3 3 -verbose

   * for multi-rank tests, run with 
       mpirun -np 4 python roundtrip.py -Nmesh 32 32 32 -Nmesh 3 3 3 --verbose

   n can be any number. procmeshes tested are:
       np = [n], [1, n], [n, 1], [a, d], [d, a]
    where a * d == n and a d are closest to n** 0.5
"""
from __future__ import print_function

from mpi4py import MPI
import itertools
import traceback
import numpy
import argparse

import os.path

parser = argparse.ArgumentParser(description='Roundtrip testing of pfft', 
        epilog=__doc__,
       formatter_class=argparse.RawDescriptionHelpFormatter 
        )

from pfft import *

oldprint = print
def print(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        oldprint(*args, **kwargs)

parser.add_argument('-Nmesh', nargs='+', type=int,
        action='append',
        help='size of FFT mesh, default is 29 30 31',
        default=[])
parser.add_argument('-Nproc', nargs='+', type=int,
        action='append',
        help='proc mesh',
        default=[])
parser.add_argument('-diag', action='store_true', default=False,
        help='show which one failed and which one passed')
parser.add_argument('-rigor', default="estimate", choices=['estimate', 'measure', 'patient', 'exhaustive'],
        help='the level of rigor in planning. ')
parser.add_argument('-verbose', action='store_true', default=False,
        help='print which test will be ran')

class LargeError(Exception):
    pass

def test_roundtrip_3d(procmesh, type, flags, inplace, Nmesh):

    partition = Partition(type, Nmesh, procmesh, flags)
    for rank in range(MPI.COMM_WORLD.size):
        MPI.COMM_WORLD.barrier()
        if rank != procmesh.rank:
            continue
        #oldprint(procmesh.rank, 'roundtrip test, np=', procmesh.np, 'Nmesh = ', Nmesh, 'inplace = ', inplace)
        #oldprint(repr(partition))

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
    # print(repr(forward))

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
    #print(repr(backward))

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
        # oldprint('error', original - input)
        MPI.COMM_WORLD.barrier()
    if False:
        print(repr(forward.type), 'forward', "error = ", r2cerr)
        print(repr(forward.type), 'backward', "error = ", c2rerr)

    r2cerr = MPI.COMM_WORLD.allreduce(r2cerr, MPI.MAX)
    c2rerr = MPI.COMM_WORLD.allreduce(c2rerr, MPI.MAX)
    if (r2cerr > 5e-4):
        raise LargeError("forward: %g" % r2cerr)

    if (c2rerr > 5e-4):
        raise LargeError("backward: %g" % c2rerr)

def main():

    ns = parser.parse_args()
    Nmesh = ns.Nmesh

    if len(Nmesh) == 0:
        # default 
        Nmesh = [[29, 30, 31]]

    if MPI.COMM_WORLD.size == 1 and len(ns.Nproc) == 0:
        nplist = [ [1], [1, 1], ]
    else:
        nplist = ns.Nproc

    rigor = {
            'exhaustive': Flags.PFFT_EXHAUSTIVE,
            'patient' : Flags.PFFT_PATIENT,
            'estimate' : Flags.PFFT_ESTIMATE,
            'measure' : Flags.PFFT_MEASURE,
            }[ns.rigor]
    import itertools
    import functools

    flags = []
    matrix = Flags.PFFT_DESTROY_INPUT, Flags.PFFT_PADDED_R2C, Flags.PFFT_TRANSPOSED_OUT
    print_flags = functools.reduce(lambda x, y: x | y, matrix, rigor)

    matrix2 = [[0, i] for i in matrix]
    for row in itertools.product(*matrix2):
        flag = functools.reduce(lambda x, y: x | y, row, rigor)
        flags.append(flag)

    params = list(itertools.product(
            nplist, [Type.PFFT_C2C, Type.PFFT_R2C, Type.PFFTF_C2C, Type.PFFTF_R2C], flags, [True, False],
            Nmesh,
            ))

    PASS = []
    FAIL = []
    IMPL = []
    for param in params:
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
                print("Failed", f, e)
            FAIL.append((param, e))
        except NotImplementedError as e:
            if ns.verbose:
                f = param
                print("notsupported", f, e)
            IMPL.append((param, e))

    N = len(PASS) + len(FAIL) + len(IMPL)

    print("PASS", len(PASS), '/', N)

    if ns.diag:
        printcase("", "", print_flags, header=True)
        for f in PASS:
            printcase(f, "", print_flags, )

    print("UNIMPL", len(IMPL), '/', N)
    if ns.diag:
        printcase("", "", print_flags, header=True)
        for f, e in IMPL:
            printcase(f, e, print_flags)

    print("FAIL", len(FAIL), '/', N)
    if ns.diag:
        printcase("", "", print_flags, header=True)
        for f, e in FAIL:
            printcase(f, e, print_flags)

    if len(FAIL) != 0:
        return 1

    return 0

def printcase(f, e, flags, header=False):
    if header:
        inplace = "INPLACE"
        np = "NP"
        flags = "FLAGS"
        type = "TYPE"
        nmesh = "NMESH"
        error = "ERROR"
    else:
        inplace = "INPL" if f[3] else "OUTP"
        np = str(f[0])
        flags = Flags(f[2]).format(flags)
        type = repr(Type(f[1]))
        nmesh = str(f[4])
        error = str(e)
    print("%(np)-6s %(nmesh)-8s %(type)-6s %(inplace)-6s %(flags)-80s %(error)-s" % locals())

# use unbuffered stdout
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

if __name__ == '__main__':

    try:
        sys.exit(main())
    except Exception as e:
        print(traceback.format_exc())
        MPI.COMM_WORLD.Abort()

