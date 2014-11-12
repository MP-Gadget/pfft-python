from mpi4py import MPI
from core import *

procmesh = ProcMesh([2, 1]) #, MPI.COMM_WORLD)

def test(flag, type):
    partition = Partition([29, 30, 31], procmesh, flag, type)
    lb = LocalBuffer(partition)
    for i in range(MPI.COMM_WORLD.size):
        MPI.COMM_WORLD.barrier()
        if i != MPI.COMM_WORLD.rank: continue
        if i == 0:
            print 'flags', partition.flags
            print 'type', partition.type
        print 'rank', i
        print 'local_ni', partition.local_ni
        print 'local_no', partition.local_no
        print 'buffer', len(lb.buffer)
        print 'input'
        try:
            print lb.view_input().dtype
            print lb.view_input().shape
        except Exception as e:
            print e
        print 'output'
        try:
            print lb.view_output().dtype
            print lb.view_output().shape
        except Exception as e:
            print e

test(PFFT_TRANSPOSED_OUT, PFFT_R2C)
#test(PFFT_TRANSPOSED_IN, PFFT_C2R)
