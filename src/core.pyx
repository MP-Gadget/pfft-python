from mpi4py cimport MPI
cdef extern from 'compat.h':
    pass
import numpy
cimport numpy

cdef extern from 'pfft.h':
    int PFFT_FORWARD
    int PFFT_BACKWARD
    int PFFT_TRANSPOSED_NONE
    int PFFT_TRANSPOSED_IN
    int PFFT_TRANSPOSED_OUT
    int PFFT_SHIFTED_NONE
    int PFFT_SHIFTED_IN
    int PFFT_SHIFTED_OUT
    int PFFT_MEASURE
    int PFFT_ESTIMATE
    int PFFT_PATIENT
    int PFFT_EXHAUSTIVE
    int PFFT_NO_TUNE
    int PFFT_TUNE
    int PFFT_PRESERVE_INPUT
    int PFFT_DESTROY_INPUT
    int PFFT_BUFFERED_INPLACE
    int PFFT_PADDED_R2C
    int PFFT_PADDED_C2R

    void pfft_init()
    void pfft_cleanup()
    struct pfft_plan:
        pass
    struct pfft_complex:
        pass
    void pfft_execute(pfft_plan * plan)
    void pfft_execute_dft(pfft_plan * plan, void * input, void * output)
    void pfft_destroy_plan(pfft_plan * plan)
    void pfft_plan_dft(
            int rnk_n, int *n, void * input, void * output, 
            MPI.MPI_Comm comm_cart,
            int sign, unsigned pfft_flags)
    int pfft_create_procmesh(int rnk_n, MPI.MPI_Comm comm, numpy.intp_t *np, 
            MPI.MPI_Comm * comm_cart)
    numpy.intp_t pfft_local_size_dft(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    double * pfft_alloc_real(size_t size)
    pfft_complex * pfft_alloc_complex(size_t size)
    void pfft_free(void * ptr)

cdef class ProcMesh(object):
    cdef MPI.MPI_Comm comm_cart
    cdef readonly numpy.ndarray np
    def __init__(self, np, MPI.Comm comm):
        cdef numpy.intp_t[::1] np_ = numpy.array(np, 'intp')
        rt = pfft_create_procmesh(np_.shape[0], comm.ob_mpi, &np_[0], &self.comm_cart)
        if rt != 0:
            raise Exception("failed to create proc mesh")
        self.np = numpy.array(np_)
        
cdef class LocalSize(object):
    cdef readonly size_t alloc_local
    cdef readonly numpy.ndarray n
    cdef readonly numpy.ndarray local_ni
    cdef readonly numpy.ndarray local_i_start
    cdef readonly numpy.ndarray local_no
    cdef readonly numpy.ndarray local_o_start

    def __init__(self, n, ProcMesh procmesh, int flags):
        cdef numpy.intp_t[::1] n_ = numpy.array(n, 'intp')
        cdef numpy.intp_t[::1] local_ni
        cdef numpy.intp_t[::1] local_no
        cdef numpy.intp_t[::1] local_i_start
        cdef numpy.intp_t[::1] local_o_start
        local_ni, local_no, local_i_start, local_o_start = \
                numpy.empty((4, n_.shape[0]), 'intp')
        rt = pfft_local_size_dft(n_.shape[0], 
                &n_[0], 
                procmesh.comm_cart,
                flags,
                &local_ni[0],
                &local_i_start[0],
                &local_no[0],
                &local_o_start[0])
        if rt <= 0:
            raise Exception("failed local size")
        self.alloc_local = rt

        self.local_ni = numpy.array(local_ni)
        self.local_no = numpy.array(local_no)
        self.local_i_start = numpy.array(local_i_start)
        self.local_o_start = numpy.array(local_o_start)
        self.n = numpy.array(n_)

cdef class LocalBuffer(object):
    cdef void * ptr
    cdef readonly LocalSize localsize
    cdef double [::1] buffer
    def __init__(self, LocalSize localsize):
        alloc_local = localsize.alloc_local
        self.ptr = pfft_alloc_complex(alloc_local)
        self.localsize = localsize

        self.buffer = <double [:alloc_local:1]> self.ptr

    def as(self, dtype):
        a = numpy.array(self.buffer)
        return a.view(dtype=dtype)

    def __dealloc__(self):
        self.buffer = None
        pfft_free(self.ptr)

cdef class Plan(object):
    def __init__(self, procmesh, input, output, int flags):
        pass

