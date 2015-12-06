from __future__ import absolute_import
from mpi4py import MPI

import pfft
from numpy.testing import assert_array_equal

def main():
    comm = MPI.COMM_WORLD
    # this must run with comm.size == 3
    assert comm.size == 3
    procmesh = pfft.ProcMesh(np=[3,])
    partition = pfft.Partition(pfft.Type.PFFT_C2C,
            [4, 4], procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT)
    
    assert_array_equal(partition.i_edges[0], [0, 2, 4, 4])
    assert_array_equal(partition.i_edges[1], [0, 4])

main()
