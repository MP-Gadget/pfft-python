#from mpi4py cimport MPI
cimport libmpi as MPI
cdef extern from 'compat.h':
    pass
import numpy
cimport numpy

cdef extern from 'pfft.h':
    int _PFFT_FORWARD "PFFT_FORWARD"
    int _PFFT_BACKWARD "PFFT_BACKWARD"
    int _PFFT_TRANSPOSED_NONE "PFFT_TRANSPOSED_NONE"
    int _PFFT_TRANSPOSED_IN "PFFT_TRANSPOSED_IN"
    int _PFFT_TRANSPOSED_OUT "PFFT_TRANSPOSED_OUT"
    int _PFFT_SHIFTED_NONE "PFFT_SHIFTED_NONE"
    int _PFFT_SHIFTED_IN "PFFT_SHIFTED_IN"
    int _PFFT_SHIFTED_OUT "PFFT_SHIFTED_OUT"
    int _PFFT_MEASURE "PFFT_MEASURE"
    int _PFFT_ESTIMATE "PFFT_ESTIMATE"
    int _PFFT_PATIENT "PFFT_PATIENT"
    int _PFFT_EXHAUSTIVE "PFFT_EXHAUSTIVE"
    int _PFFT_NO_TUNE "PFFT_NO_TUNE"
    int _PFFT_TUNE "PFFT_TUNE"
    int _PFFT_PRESERVE_INPUT "PFFT_PRESERVE_INPUT"
    int _PFFT_DESTROY_INPUT "PFFT_DESTROY_INPUT"
    int _PFFT_BUFFERED_INPLACE "PFFT_BUFFERED_INPLACE"
    int _PFFT_PADDED_R2C "PFFT_PADDED_R2C"
    int _PFFT_PADDED_C2R "PFFT_PADDED_C2R"

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

    int pfft_create_procmesh(int rnk_n, MPI.MPI_Comm comm, int *np, 
            MPI.MPI_Comm * comm_cart)

    numpy.intp_t pfft_local_size_dft(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfft_local_size_dft_r2c(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfft_local_size_dft_c2r(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfft_local_size_r2r(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    double * pfft_alloc_real(size_t size)
    pfft_complex * pfft_alloc_complex(size_t size)
    void pfft_free(void * ptr)

PFFT_FORWARD = _PFFT_FORWARD
PFFT_BACKWARD = _PFFT_BACKWARD
PFFT_TRANSPOSED_NONE = _PFFT_TRANSPOSED_NONE
PFFT_TRANSPOSED_IN = _PFFT_TRANSPOSED_IN
PFFT_TRANSPOSED_OUT = _PFFT_TRANSPOSED_OUT
PFFT_SHIFTED_NONE = _PFFT_SHIFTED_NONE
PFFT_SHIFTED_IN = _PFFT_SHIFTED_IN
PFFT_SHIFTED_OUT = _PFFT_SHIFTED_OUT
PFFT_MEASURE = _PFFT_MEASURE
PFFT_ESTIMATE = _PFFT_ESTIMATE
PFFT_PATIENT = _PFFT_PATIENT
PFFT_EXHAUSTIVE = _PFFT_EXHAUSTIVE
PFFT_NO_TUNE = _PFFT_NO_TUNE
PFFT_TUNE = _PFFT_TUNE
PFFT_PRESERVE_INPUT = _PFFT_PRESERVE_INPUT
PFFT_DESTROY_INPUT = _PFFT_DESTROY_INPUT
PFFT_BUFFERED_INPLACE = _PFFT_BUFFERED_INPLACE
PFFT_PADDED_R2C = _PFFT_PADDED_R2C
PFFT_PADDED_C2R = _PFFT_PADDED_C2R

PFFT_C2C = 0
PFFT_R2C = 1
PFFT_C2R = 2
PFFT_R2R = 3

ctypedef numpy.intp_t (*pfft_local_size_func)(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

cdef pfft_local_size_func PFFT_LOCAL_SIZE_FUNC [4]

PFFT_LOCAL_SIZE_FUNC[:] = [
    pfft_local_size_dft,
    pfft_local_size_dft_r2c,
    pfft_local_size_dft_c2r,
    pfft_local_size_r2r
        ]

cdef class ProcMesh(object):
    cdef MPI.MPI_Comm comm_cart
    cdef readonly numpy.ndarray np
    def __init__(self, np, comm = None):
        cdef MPI.MPI_Comm mpicomm
        if comm == "world" or comm is None:
            mpicomm = MPI.MPI_COMM_WORLD
        else:
            raise Exception("hahahah")
        cdef int [::1] np_ = numpy.array(np, 'int32')
        rt = pfft_create_procmesh(np_.shape[0], mpicomm, &np_[0], &self.comm_cart)
        print np_.shape[0], np_[0], np_[1]
        if rt != 0:
            raise Exception("failed to create proc mesh")
        self.np = numpy.array(np_)
        
cdef class Partition(object):
    cdef readonly size_t alloc_local
    cdef readonly numpy.ndarray n
    cdef readonly numpy.ndarray local_ni
    cdef readonly numpy.ndarray local_i_start
    cdef readonly numpy.ndarray local_no
    cdef readonly numpy.ndarray local_o_start
    cdef readonly int type
    cdef readonly int flags
    def __init__(self, n, ProcMesh procmesh, int flags, int type):
        cdef numpy.intp_t[::1] n_ = numpy.array(n, 'intp')
        cdef numpy.intp_t[::1] local_ni
        cdef numpy.intp_t[::1] local_no
        cdef numpy.intp_t[::1] local_i_start
        cdef numpy.intp_t[::1] local_o_start
        local_ni, local_no, local_i_start, local_o_start = \
                numpy.empty((4, n_.shape[0]), 'intp')

        cdef pfft_local_size_func func = PFFT_LOCAL_SIZE_FUNC[type]

        self.type = type
        self.flags = flags

        rt = func(n_.shape[0], 
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

cdef class LocalBuffer:
    cdef void * ptr
    cdef readonly double [::1] buffer
    cdef readonly Partition partition

    def __init__(self, Partition partition):
        self.partition = partition
        self.ptr = pfft_alloc_complex(partition.alloc_local)
        self.buffer = <double [:partition.alloc_local * 2:1]> self.ptr

    def _view(self, dtype, local_n, local_start, roll):
        shape = local_n.copy()

        a = numpy.array(self.buffer, copy=False)
        a = a.view(dtype=dtype)
        if numpy.iscomplexobj(a):
            # complex array needs no padding
            shape2 = shape
            pass
        else:
            # real array needs to worry about padding
            shape2 = shape.copy()
            global_end = local_start + local_n
            if global_end[-1] == self.partition.n[-1]:
                # This chunk is the last one along the conjugated axis
                shape[-1] = 2 * (shape[-1] // 2 + 1)

        if roll:
            # The shape of the input array differ from local_ni
            # FIXME: this works for 3d and 2d, not sure if it is 
            # correct in 4d!
            shape = numpy.roll(shape, -1)
            shape2 = numpy.roll(shape2, -1)

        sel = [slice(0, s) for s in shape2]
        a = a[:numpy.product(shape)]
        a = a.reshape(shape)
        a = a[sel]
        return a

    def view_input(self):
        dtypes = ['complex128', 'float64', 'complex128', 'float64']
        return self._view(dtypes[self.partition.type],
                self.partition.local_ni,
                self.partition.local_i_start,
                self.partition.flags | PFFT_TRANSPOSED_IN)

    def view_output(self):
        dtypes = ['complex128', 'complex128', 'float64', 'float64']
        return self._view(dtypes[self.partition.type],
                self.partition.local_no,
                self.partition.local_o_start,
                self.partition.flags | PFFT_TRANSPOSED_OUT)

    def __dealloc__(self):
        self.buffer = None
        pfft_free(self.ptr)

cdef class Plan(object):
    def __init__(self, procmesh, LocalBuffer i, LocalBuffer o, int flags, int type):
        pass

pfft_init()
print 'init pfft'
