from mpi4py import MPI

import pfft
import numpy

def main(comm):
    Nmesh = [8, 8]
    if len(Nmesh) == 3:
        procmesh = pfft.ProcMesh(pfft.split_size_2d(comm.size), comm=comm)
    else:
        procmesh = pfft.ProcMesh((comm.size,), comm=comm)

    partition = pfft.Partition(
        pfft.Type.PFFT_R2C,
        Nmesh,
        procmesh,
        pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT
        )

    # generate the coordinate support.

    k = [None] * partition.ndim
    x = [None] * parititon.ndim
    for d in range(partition.ndim):
        k[d] = numpy.arange(partition.no[d])[partition.local_o_slice[d]]
        k[d][k[d] > partition.no[d] // 2] -= partition.no[d]
        # set to the right numpy broadcast shape
        k[d] = k[d].reshape([-1 if i == d else 1 for i in range(partition.ndim)])

        x[d] = numpy.arange(partition.ni[d])[partition.local_i_slice[d]]
        # set to the right numpy broadcast shape
        x[d] = x[d].reshape([-1 if i == d else 1 for i in range(partition.ndim)])

    # allocate memory
    buffer1 = pfft.LocalBuffer(partition)
    phi_disp = buffer1.view_input()

    buffer2 = pfft.LocalBuffer(partition)
    phi_spec = buffer2.view_output()

    # forward plan
    disp_to_spec = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer1, buffer2)

    buffer3 = pfft.LocalBuffer(partition)
    grad_spec = buffer3.view_output()

    buffer4 = pfft.LocalBuffer(partition)
    grad_disp = buffer4.view_input()

    # backward plan
    spec_to_disp = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer3, buffer4)

    # to do : fill in initial value
    dx = x[0] - Nmesh[0] * 0.5 + 0.5
    dy = x[1] - Nmesh[1] * 0.5 + 0.5
    phi_disp[...] = dx ** 2 + dx * dy + dy ** 2

    cprint('phi =', gather(partition, phi_disp).round(2), comm=comm)

    disp_to_spec.execute(phi_disp.base, phi_spec.base)

    all_grad_disp = numpy.zeros([partition.ndim] + list(phi_disp.shape), dtype=grad_disp.dtype)

    cprint('phi_k =', gather(partition, phi_spec, mode='output').round(2), comm=comm)

    for d in range(partition.ndim):
        grad_spec[...] = phi_spec[...] * (k[d] * 1j)
        cprint('grad_k =', gather(partition, grad_spec, mode='output').round(2), comm=comm)

        disp_to_spec.execute(grad_spec.base, grad_disp.base)
        # copy the gradient along d th direction
        all_grad_disp[d] = grad_disp

    # now do your thing.

    for d in range(partition.ndim):
        cprint('dim =', gather(partition, all_grad_disp[d]).round(2), comm=comm)

def cprint(*args, comm):
    if comm.rank == 0:
        print(*args)

def gather(partition, data, mode='input'):
    if mode == 'input':
        full = numpy.zeros(partition.ni, data.dtype)
        full[partition.local_i_slice] = data
    else:
        full = numpy.zeros(partition.no, data.dtype)
        full[partition.local_o_slice] = data
    partition.procmesh.comm.Allreduce(MPI.IN_PLACE, full)
    return full

main(MPI.COMM_WORLD)
