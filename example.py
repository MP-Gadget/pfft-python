from mpi4py import MPI

import pfft

procmesh = pfft.ProcMesh([4])
partition = pfft.Partition(
        pfft.Type.PFFT_C2C, 
        [8, 8], 
        procmesh, 
        pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT
        )

buffer = pfft.LocalBuffer(partition)

plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer)
iplan = pfft.Plan(partition, pfft.Direction.PFFT_BACKWARD, buffer, 
        flags=pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT,
        )

input = buffer.view_input()
input[...] = 1.0
plan.execute(buffer)
output = buffer.view_output()
iplan.execute(buffer)
print input
