from mpi4py import MPI
import numpy
import pfft

if MPI.COMM_WORLD.rank == 0:
    print \
        """
This example performs a in-place transform, with a naive slab decomposition.

In place transform is achieved by providing a single buffer object to pfft.Plan.
Consequently, calls to plan.execute we also provide only a single buffer object.
"""


procmesh = pfft.ProcMesh([4], comm=MPI.COMM_WORLD)
partition = pfft.Partition(
        pfft.Type.PFFT_C2C,
        [8, 8],
        procmesh,
        pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT
        )
for irank  in range(4):
    MPI.COMM_WORLD.barrier()
    if irank != procmesh.rank:
        continue
    print 'My rank is', procmesh.this
    print 'local_i_start', partition.local_i_start
    print 'local_o_start', partition.local_o_start
    print 'i_edges', partition.i_edges
    print 'o_edges', partition.o_edges

buffer = pfft.LocalBuffer(partition)

plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer)
iplan = pfft.Plan(partition, pfft.Direction.PFFT_BACKWARD, buffer,
        flags=pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT,
        )

input = buffer.view_input()
print input.base
# now lets fill the input array in a funny way
# a[i, j] = i * 10 + j
# we will do a tranform roundtrip and print this out
indices = numpy.array(numpy.indices(input.shape))
indices += partition.local_i_start[:, None, None]
i, j = indices
input[...] = i * 10 + j

plan.execute(buffer)

output = buffer.view_output()

# denormalize the forward transform
output /= numpy.prod(partition.n)

iplan.execute(buffer)

ginput = MPI.COMM_WORLD.gather(input)
if MPI.COMM_WORLD.rank == 0:
    print 'You shall see an array of form i * 10 + j'
    for item in ginput:
        print item
