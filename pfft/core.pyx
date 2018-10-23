#cython: embedsignature=True
#cython: boundscheck=False
"""
    pfft-python: python binding of PFFT.

    Author: Yu Feng (yfeng1@berkeley.edu), 
          University of California Berkeley (2014)

"""
from mpi4py import MPI
cimport libmpi as cMPI
import numpy
cimport numpy
from libc.stdlib cimport free, calloc
from libc.string cimport memset

numpy.import_array()

def split_size_2d(s):
    """ Split `s` into two integers, 
        a and d, such that a * d == s and a <= d

        returns:  a, d
    """
    a = int(s** 0.5) + 1
    d = s
    while a > 1:
        if s % a == 0:
            d = s // a
            break
        a = a - 1 
    return a, d

####
#  import those pfft functions
#####
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

    int  _FFTW_R2HC "FFTW_R2HC"
    int  _FFTW_HC2R "FFTW_HC2R"

    void pfft_init()
    void pfftf_init()
    void pfft_cleanup()

    ctypedef void * pfft_plan

    struct pfft_complex:
        pass
    struct pfftf_complex:
        pass

    void pfft_execute_dft(pfft_plan plan, void * input, void * output)
    void pfft_execute_dft_r2c(pfft_plan plan, void * input, void * output)
    void pfft_execute_dft_c2r(pfft_plan plan, void * input, void * output)
    void pfft_execute_r2r(pfft_plan plan, void * input, void * output)

    void pfftf_execute_dft(pfft_plan plan, void * input, void * output)
    void pfftf_execute_dft_r2c(pfft_plan plan, void * input, void * output)
    void pfftf_execute_dft_c2r(pfft_plan plan, void * input, void * output)
    void pfftf_execute_r2r(pfft_plan plan, void * input, void * output)

    void pfft_destroy_plan(pfft_plan plan)
    void pfftf_destroy_plan(pfft_plan plan)

    pfft_plan pfft_plan_dft(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

    pfft_plan pfft_plan_dft_r2c(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

    pfft_plan pfft_plan_dft_c2r(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

    pfft_plan pfft_plan_r2r(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int * kinds, unsigned pfft_flags)

    pfft_plan pfftf_plan_dft(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

    pfft_plan pfftf_plan_dft_r2c(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

    pfft_plan pfftf_plan_dft_c2r(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

    pfft_plan pfftf_plan_r2r(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int * kinds, unsigned pfft_flags)

    int pfft_create_procmesh(int rnk_n, cMPI.MPI_Comm comm, int *np, 
            cMPI.MPI_Comm * ccart)

    int pfftf_create_procmesh(int rnk_n, cMPI.MPI_Comm comm, int *np, 
            cMPI.MPI_Comm * ccart)

    numpy.intp_t pfft_local_size_dft(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfft_local_size_dft_r2c(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfft_local_size_dft_c2r(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfft_local_size_r2r(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfftf_local_size_dft(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfftf_local_size_dft_r2c(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfftf_local_size_dft_c2r(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    numpy.intp_t pfftf_local_size_r2r(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

    double * pfft_alloc_real(size_t size)
    pfft_complex * pfft_alloc_complex(size_t size)
    pfftf_complex * pfftf_alloc_complex(size_t size)
    void pfft_free(void * ptr)

#######
#  wrap Flags, Direction
#####

class Flags(int):
    """
    PFFT Transformation Flags
    """

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
    TRANSPOSED_NONE = _PFFT_TRANSPOSED_NONE
    TRANSPOSED_IN = _PFFT_TRANSPOSED_IN
    TRANSPOSED_OUT = _PFFT_TRANSPOSED_OUT
    SHIFTED_NONE = _PFFT_SHIFTED_NONE
    SHIFTED_IN = _PFFT_SHIFTED_IN
    SHIFTED_OUT = _PFFT_SHIFTED_OUT
    MEASURE = _PFFT_MEASURE
    ESTIMATE = _PFFT_ESTIMATE
    PATIENT = _PFFT_PATIENT
    EXHAUSTIVE = _PFFT_EXHAUSTIVE
    NO_TUNE = _PFFT_NO_TUNE
    TUNE = _PFFT_TUNE
    PRESERVE_INPUT = _PFFT_PRESERVE_INPUT
    DESTROY_INPUT = _PFFT_DESTROY_INPUT
    BUFFERED_INPLACE = _PFFT_BUFFERED_INPLACE
    PADDED_R2C = _PFFT_PADDED_R2C
    PADDED_C2R = _PFFT_PADDED_C2R

    def __new__(cls, value):
        self = int.__new__(cls, value)
        return self

    def __repr__(self):
        d = self.__class__.__dict__
        keys = sorted([k for k in d.keys() if k.isupper() and not k.startswith('PFFT')])
        return '|'.join([k for k in keys if (d[k] & self)])

    def format(self, flags=None):
        d = self.__class__.__dict__
        keys = sorted([k for k in d.keys() if k.isupper() and not k.startswith('PFFT')])
        s = []
        for key in keys:
            if flags is not None and not (d[key] & flags): continue
            if d[key] & self:
                s.append(key)
            else:
                s.append(" " * len(key))
        return ' '.join(s)

class Direction(int):
    """ 
    PFFT Transformation Directions 
    """
    PFFT_FORWARD = _PFFT_FORWARD
    PFFT_BACKWARD = _PFFT_BACKWARD
    FORWARD = _PFFT_FORWARD
    BACKWARD = _PFFT_BACKWARD
    def __new__(cls, value):
        self = int.__new__(cls, value)
        return self

    def __repr__(self):
        d = self.__class__.__dict__
        keys = sorted([k for k in d.keys() if k.isupper() and not k.startswith('PFFT')])
        return 'and'.join([k for k in keys if (d[k] == self)])

######
# define Type as the transform type
# fill in the function tables as well.
##
class Type(int):
    """
    PFFT Transformation Types 
    Double precision is prefixed with PFFT
    Single precision is prefixed with PFFTF
    """
    PFFT_C2C = 0
    PFFT_R2C = 1
    PFFT_C2R = 2
    PFFT_R2R = 3
    PFFTF_C2C = 4
    PFFTF_R2C = 5
    PFFTF_C2R = 6
    PFFTF_R2R = 7
    C2C = 0
    R2C = 1
    C2R = 2
    R2R = 3
    C2CF = 4
    R2CF = 5
    C2RF = 6
    R2RF = 7
    def __new__(cls, value):
        self = int.__new__(cls, value)
        return self

    def __repr__(self):
        d = self.__class__.__dict__
        keys = sorted([k for k in d.keys() if k.isupper() and not k.startswith('PFFT')])
        return 'and'.join([k for k in keys if (d[k] == self)])

    def is_inverse_of(self, other):
        return self == other.inverse

    @property
    def inverse(self):
        inverses = { Type.C2C : Type.C2C,
                     Type.R2C : Type.C2R,
                     Type.C2R : Type.R2C,
                     Type.R2R : Type.R2R,
                     Type.C2CF : Type.C2CF,
                     Type.R2CF : Type.C2RF,
                     Type.C2RF : Type.R2CF,
                     Type.R2RF : Type.R2RF,
                    }
        return inverses[self]

ctypedef numpy.intp_t (*pfft_local_size_func)(int rnk_n, numpy.intp_t * n, cMPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)

cdef pfft_local_size_func PFFT_LOCAL_SIZE_FUNC [8]

PFFT_LOCAL_SIZE_FUNC[:] = [
    <pfft_local_size_func> pfft_local_size_dft,
    <pfft_local_size_func> pfft_local_size_dft_r2c,
    <pfft_local_size_func> pfft_local_size_dft_c2r,
    <pfft_local_size_func> pfft_local_size_r2r,
    <pfft_local_size_func> pfftf_local_size_dft,
    <pfft_local_size_func> pfftf_local_size_dft_r2c,
    <pfft_local_size_func> pfftf_local_size_dft_c2r,
    <pfft_local_size_func> pfftf_local_size_r2r,
        ]

ctypedef pfft_plan (*pfft_plan_func) (
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int sign, unsigned pfft_flags)

ctypedef pfft_plan (*pfft_plan_func_r2r) (
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            cMPI.MPI_Comm ccart,
            int * kinds, unsigned pfft_flags)

cdef void * PFFT_PLAN_FUNC [8]

PFFT_PLAN_FUNC[:] = [
    <void*> pfft_plan_dft,
    <void*> pfft_plan_dft_r2c,
    <void*> pfft_plan_dft_c2r,
    <void*> pfft_plan_r2r,
    <void*> pfftf_plan_dft,
    <void*> pfftf_plan_dft_r2c,
    <void*> pfftf_plan_dft_c2r,
    <void*> pfftf_plan_r2r,
        ]

ctypedef void (*pfft_free_plan_func) (void * plan)
cdef pfft_free_plan_func PFFT_FREE_PLAN_FUNC [8]

PFFT_FREE_PLAN_FUNC[:] = [
    <pfft_free_plan_func> pfft_destroy_plan,
    <pfft_free_plan_func> pfft_destroy_plan,
    <pfft_free_plan_func> pfft_destroy_plan,
    <pfft_free_plan_func> pfft_destroy_plan,
    <pfft_free_plan_func> pfftf_destroy_plan,
    <pfft_free_plan_func> pfftf_destroy_plan,
    <pfft_free_plan_func> pfftf_destroy_plan,
    <pfft_free_plan_func> pfftf_destroy_plan,
        ]

ctypedef void (*pfft_execute_func) ( pfft_plan plan, void * input, void * output)
cdef pfft_execute_func PFFT_EXECUTE_FUNC [8]

PFFT_EXECUTE_FUNC[:] = [
    <pfft_execute_func> pfft_execute_dft,
    <pfft_execute_func> pfft_execute_dft_r2c,
    <pfft_execute_func> pfft_execute_dft_c2r,
    <pfft_execute_func> pfft_execute_r2r,
    <pfft_execute_func> pfftf_execute_dft,
    <pfft_execute_func> pfftf_execute_dft_r2c,
    <pfft_execute_func> pfftf_execute_dft_c2r,
    <pfft_execute_func> pfftf_execute_r2r,
        ]
cdef int PFFT_NPY_TYPE[8]

PFFT_NPY_TYPE[:] = [
    numpy.NPY_DOUBLE,
    numpy.NPY_DOUBLE,
    numpy.NPY_DOUBLE,
    numpy.NPY_DOUBLE,
    numpy.NPY_FLOAT,
    numpy.NPY_FLOAT,
    numpy.NPY_FLOAT,
    numpy.NPY_FLOAT,
        ]



cdef class ProcMesh(object):
    """
    The topology of the MPI ranks. (procmesh)

    Attributes
    ==========
    comm   :   MPI.Comm
        MPI communicator the proc mesh is built for.
        Note that it does not have the 2D topology.

    this        : array_like 
        The rank of current process in the procmesh 
    np          : array_like
        The shape of the proc mesh. 
    ndim        : int
        size of the proc mesh
    rank        : int
        MPI rank
    """
    cdef readonly numpy.ndarray this # nd rank of the current process
    cdef readonly numpy.ndarray np
    cdef readonly int rank
    cdef readonly int ndim
    cdef readonly object comm

    cdef cMPI.MPI_Comm ccart
    cdef cMPI.MPI_Comm * ccol

    @classmethod
    def split(cls, ndim, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        if ndim == 2:
            np = split_size_2d(comm.size)
        elif ndim == 1:
            np = [comm.size]
        else:
            raise ValueError("only know how to split to upto 2d")
        return np

    def __init__(self, np, comm=None):
        """ A mesh of processes 
            np is the number of processes in each direction.

            example:
                procmesh = ProcMesh([2, 3]) # creates a 2 x 3 mesh.

            product(np) must equal to comm.size

            if the mpi4py version is recent (cMPI._addressof), comm can
            be any mpi4py Comm objects.
        """
        cdef cMPI.MPI_Comm ccomm
        self.ccol = NULL
        self.ccart = NULL

        if comm is None:
            comm = MPI.COMM_WORLD

        if isinstance(comm, MPI.Comm):
            if hasattr(MPI, '_addressof'):
                ccomm = (<cMPI.MPI_Comm*> (<numpy.intp_t>
                        MPI._addressof(comm))) [0]
            else:
                if comm == MPI.COMM_WORLD:
                    ccomm = cMPI.MPI_COMM_WORLD
                else:
                    raise ValueError("only comm=MPI.COMM_WORLD is supported, "
                            + " update mpi4py to 2.0, with MPI._addressof")
        cdef int [::1] np_ = numpy.array(np, 'int32')
        rt = pfft_create_procmesh(np_.shape[0], ccomm, &np_[0], &self.ccart)

        if rt != 0:
            self.ccart = NULL
            raise RuntimeError("Failed to create proc mesh")
        pycomm = comm.Create_cart(dims=np_,
                                      periods=[True] * len(np_),
                                      reorder=1)
        self.comm = pycomm
        self.rank = pycomm.rank

        self.np = numpy.array(np_)
        self.ndim = len(self.np)

        # a buffer used for various purposes 
        cdef int[::1] junk = numpy.empty(self.ndim, 'int32')

        # now fill `this'
        self.this = numpy.array(np, 'int32')
        cMPI.MPI_Cart_get(self.ccart, 2, 
                &junk[0], &junk[0],
                <int*>self.this.data);

        # build the ccol sub communicators
        self.ccol = <cMPI.MPI_Comm*>calloc(self.ndim, sizeof(cMPI.MPI_Comm))
        for i in range(self.ndim):
            junk[:] = 0
            junk[i] = 1
            if cMPI.MPI_SUCCESS != cMPI.MPI_Cart_sub(self.ccart, &junk[0],
                    &self.ccol[i]):
                self.ccol[i] = NULL
                raise RuntimeError("Failed to create sub communicators")

    def __dealloc__(self):
        if self.ccart:
            cMPI.MPI_Comm_free(&self.ccart)
            pass
        if self.ccol != NULL:
            for i in range(self.ndim):
                if self.ccol[i]:
                    cMPI.MPI_Comm_free(&self.ccol[i])
            free(self.ccol)

cdef class Partition(object):
    cdef readonly size_t alloc_local
    cdef readonly int ndim
    cdef readonly numpy.ndarray n
    cdef readonly numpy.ndarray ni
    cdef readonly numpy.ndarray no
    cdef readonly numpy.ndarray local_ni
    cdef readonly numpy.ndarray local_i_start
    cdef readonly numpy.ndarray local_no
    cdef readonly numpy.ndarray local_o_start
    cdef readonly numpy.ndarray local_i_strides
    cdef readonly numpy.ndarray local_i_shape
    cdef readonly numpy.ndarray local_o_strides
    cdef readonly numpy.ndarray local_o_shape
    cdef readonly object local_i_slice
    cdef readonly object local_o_slice
    cdef readonly object type
    cdef readonly object flags
    cdef readonly ProcMesh procmesh
    cdef readonly object i_edges
    cdef readonly object o_edges
    cdef readonly numpy.dtype i_dtype
    cdef readonly numpy.dtype o_dtype

    i_dtypes = [
            'complex128', 'float64', 'complex128', 'float64',
            'complex64', 'float32', 'complex64', 'float32',
             ]

    o_dtypes = [
                'complex128', 'complex128', 'float64', 'float64',
                'complex64', 'complex64', 'float32', 'float32',
                 ]

    def __init__(self, type, n, ProcMesh procmesh, flags):
        """ A data partition object 
            type is the type of the transform, r2c, c2r, c2c or r2r see Type.
            n is the size of the mesh.
            procmesh is a ProcMesh object
            flags, see Flags

            i_edges: the edges of the input mesh. This is identical on all
                     ranks. Notice that if the input is PFFT_TRANSPOSED_IN the edges
                     remain the ordering of the original array. The mapping to
                     the procmesh is somewhat complicated:
                         (I will write this when I figure it out)
            o_edges: the edges of the output mesh. similar to i_edges

            local_i_start: the start offset.
            local_o_start: the start offset.

            Example:
                Partition(Type.R2C, [32, 32, 32], procmesh, Flags.PFFT_TRANSPOSED_OUT)
        """
        self.procmesh = procmesh
        cdef numpy.intp_t[::1] n_ = numpy.array(n, 'intp')
        cdef numpy.intp_t[::1] local_ni
        cdef numpy.intp_t[::1] local_no
        cdef numpy.intp_t[::1] local_i_start
        cdef numpy.intp_t[::1] local_o_start
        cdef numpy.intp_t[::1] local_i_strides
        cdef numpy.intp_t[::1] local_o_strides

        local_ni, local_no, local_i_start, local_o_start = numpy.empty((4, n_.shape[0]), 'intp')

        self.type = Type(type)
        self.flags = Flags(flags)

        if len(n_) < len(procmesh.np):
            raise ValueError("ProcMesh (%d) shall have less dimentions than Mesh (%d)" % (len(procmesh.np), len(n_)))

        if len(n_) == len(procmesh.np):
            if len(n_) != 2 and len(n_) != 3: # https://github.com/mpip/pfft/issues/29
                raise NotImplementedError("Currently using the same ProcMesh (%d) dimentions with Mesh (%d) is not supported other than 2don2d or 3don3d" % (len(procmesh.np), len(n_)))

            if ( ((self.flags & Flags.PFFT_PADDED_R2C) | (self.flags & Flags.PFFT_PADDED_C2R))
             and ( self.type in (Type.R2C, Type.C2R, Type.R2CF, Type.C2RF))
               ):
                # https://github.com/mpip/pfft/pull/31
                raise NotImplementedError("Currently using the same ProcMesh (%d) dimentions with Mesh (%d) is not supported on padded transforms." % (len(procmesh.np), len(n_)))

        cdef pfft_local_size_func func = PFFT_LOCAL_SIZE_FUNC[self.type]


        rt = func(n_.shape[0], 
                &n_[0], 
                procmesh.ccart,
                flags,
                &local_ni[0],
                &local_i_start[0],
                &local_no[0],
                &local_o_start[0])

        if rt <= 0:
            raise RuntimeError("failed local size")

        self.alloc_local = rt
        self.local_ni = numpy.array(local_ni)
        self.local_no = numpy.array(local_no)
        self.local_i_start = numpy.array(local_i_start)
        self.local_o_start = numpy.array(local_o_start)
        self.n = numpy.array(n_)
        self.ndim = len(self.n)

        self.i_dtype = numpy.dtype(self.i_dtypes[self.type])
        self.o_dtype = numpy.dtype(self.o_dtypes[self.type])

        self.local_i_shape, self.local_i_strides = \
            self._build_shape_strides(
                self.local_i_start,
                self.local_ni,
                self.flags & Flags.PFFT_TRANSPOSED_IN)
        self.local_i_strides *= self.i_dtype.itemsize

        self.local_o_shape, self.local_o_strides =\
            self._build_shape_strides(
                self.local_o_start,
                self.local_no,
                self.flags & Flags.PFFT_TRANSPOSED_OUT)
        self.local_o_strides *= self.o_dtype.itemsize

        # Notice that local_i_start and i_edges can be different
        # due to https://github.com/mpip/pfft/issues/22
        #
        # i_edges are used for domain decomposition, thus
        # supposed to be non-decreasing, so we calculate
        # them from local_ni.

        self.i_edges = self._build_edges(self.local_ni,
                self.flags & Flags.PFFT_TRANSPOSED_IN,
                )
        self.o_edges = self._build_edges(self.local_no,
                self.flags & Flags.PFFT_TRANSPOSED_OUT
                )

        # it is alright to use the 'zero' local_i_start
        # in slices, since the local_ni is is also zero
        # and would give a zero size slice anyways.
        self.local_i_slice = tuple(
                [slice(start, start + n)
                for start, n in zip(
                    self.local_i_start, self.local_ni)])
        self.local_o_slice = tuple(
                [slice(start, start + n)
                for start, n in zip(
                    self.local_o_start, self.local_no)])

        self.ni = numpy.array([e[-1] for e in self.i_edges], dtype='intp')
        self.no = numpy.array([e[-1] for e in self.o_edges], dtype='intp')

    def _build_shape_strides(self, numpy.intp_t[::1] local_start, numpy.intp_t[::1] local_n, transposed):
        cdef int d
        cdef numpy.intp_t[::1] axismapping
        cdef numpy.intp_t[::1] strides
        cdef numpy.intp_t[::1] shape

        strides = numpy.empty(local_n.shape[0], dtype='intp')
        shape = numpy.empty(local_n.shape[0], dtype='intp')

        # invaxismapping[d] stores the untransposed axis for d
        axismapping = numpy.arange(self.ndim, dtype='intp')
        if transposed:
            first = axismapping[:self.procmesh.ndim + 1]
            first[:] = numpy.roll(first, -1)
        #    print numpy.array(axismapping)
        for d in range(self.ndim):
            shape[d] = local_n[d]

        # strides are transposed
        strides[axismapping[self.ndim - 1]] = 1

        #print 'local_n', numpy.array(local_n)
        for d in range(self.ndim - 2, -1, -1):
            d0 = axismapping[d]
            d1 = axismapping[d + 1]
            strides[d0] = local_n[d1] * strides[d1]
            #print d0, d1, local_n[d1], strides[d1], '=', strides[d0]

        # if shape[d] is too large, there is padding
        # we know it must be r2c here, so replace with n[d]
        for d in range(self.ndim):
            if shape[d] > self.n[d] - local_start[d]:
                shape[d] = self.n[d] - local_start[d]

        return numpy.array(shape), numpy.array(strides)

    def _build_edges(self, numpy.intp_t[::1] local_n, transposed):
        cdef numpy.intp_t[::1] start_dim
        cdef numpy.intp_t[::1] invaxismapping
        cdef numpy.intp_t tmp
        edges = []
        cdef int d
        cdef int d1

        # invaxismapping[d] stores the transposed axis for d
        invaxismapping = numpy.arange(self.ndim, dtype='intp')
        if transposed:
            first = invaxismapping[:self.procmesh.ndim + 1]
            first[:] = numpy.roll(first, 1)

        np = numpy.ones(self.ndim, dtype='int')
        np[:self.procmesh.ndim] = self.procmesh.np
        for d in range(self.ndim):
            d1 = invaxismapping[d]

            start_dim = numpy.empty((np[d1] + 1), dtype='intp')
            start_dim[0] = 0
            if d1 < self.procmesh.ndim:
                tmp = local_n[d]
                cMPI.MPI_Allgather(&tmp, sizeof(numpy.intp_t), cMPI.MPI_BYTE, 
                        &start_dim[1], sizeof(numpy.intp_t), cMPI.MPI_BYTE, 
                        self.procmesh.ccol[d1])
            else:
                # use the full axis, because it is not chopped
                start_dim[1] = local_n[d]
                # if local_n is too large, there is padding
                # we know it must be r2c here, so replace with n[d]
                if start_dim[1] > self.n[d]:
                    start_dim[1] = self.n[d]

            start_dim_a = numpy.array(start_dim, copy=False)
            start_dim_a[:] = numpy.cumsum(start_dim_a)

            edges.append(numpy.array(start_dim))
        return edges

    def __repr__(self):
        return 'Partition(' + \
                ','.join([
                    'n = %s' % str(self.n),
                    'local_ni = %s' % str(self.local_ni),
                    'local_no = %s' % str(self.local_no),
                    'local_i_start = %s' % str(self.local_i_start),
                    'local_o_start = %s' % str(self.local_o_start),
                    'flags = %s' % repr(self.flags),
                    'type = %s' % repr(self.type),
                    ]) + ')'

cdef class LocalBuffer:
    cdef void * ptr
    cdef readonly numpy.intp_t address
    cdef readonly Partition partition
    cdef readonly LocalBuffer base

    # keep track of base because the python object may be destroyed
    # before dealloc is called
    cdef int _has_base

    def __init__(self, partition, LocalBuffer base=None):
        """ The local portion of the distributed array used by PFFT 

            see the documents of view_input, view_output
        """
        self.partition = partition

        self.base = base

        if base is None:
            if PFFT_NPY_TYPE[self.partition.type] == numpy.NPY_DOUBLE:
                self.ptr = pfft_alloc_complex(partition.alloc_local)
            elif PFFT_NPY_TYPE[self.partition.type] == numpy.NPY_FLOAT:
                self.ptr = pfftf_alloc_complex(partition.alloc_local)
            self._has_base = 0
        else:
            assert base.partition.alloc_local == self.partition.alloc_local
            #FIXME: check procmesh
            self.ptr = base.ptr
            self._has_base = 1

        self.address = <numpy.intp_t> self.ptr

    def __contains__(self, LocalBuffer other):
        return self.address == other.address

    def view_raw(self, type=numpy.ndarray):
        cdef numpy.dtype dt
        if PFFT_NPY_TYPE[self.partition.type] == numpy.NPY_DOUBLE:
            dt = numpy.dtype('f8')
        else:
            dt = numpy.dtype('f4')
        cdef numpy.intp_t alloc_local = 2 * self.partition.alloc_local
        cdef numpy.intp_t strides = dt.itemsize
        cdef numpy.ndarray a = numpy.PyArray_New(type,
                1,
                <numpy.intp_t*>&alloc_local,
                dt.type_num,
                <numpy.intp_t*>&strides,
                self.ptr, dt.itemsize, numpy.NPY_BEHAVED, None)

        numpy.set_array_base(a, self)
        return a

    def view_input(self, type=numpy.ndarray):
        cdef numpy.dtype dt = self.partition.i_dtype
        cdef numpy.ndarray a = numpy.PyArray_New(type,
                self.partition.ndim,
                <numpy.intp_t*>self.partition.local_i_shape.data,
                dt.type_num,
                <numpy.intp_t*>self.partition.local_i_strides.data,
                self.ptr, dt.itemsize, numpy.NPY_BEHAVED, None)

        numpy.set_array_base(a, self)

        return a

    def view_output(self, type=numpy.ndarray):
        cdef numpy.dtype dt = self.partition.o_dtype

        cdef numpy.ndarray a = numpy.PyArray_New(type,
                self.partition.ndim,
                <numpy.intp_t*>self.partition.local_o_shape.data,
                dt.type_num, <numpy.intp_t*>
                    self.partition.local_o_strides.data,
                self.ptr, dt.itemsize, numpy.NPY_BEHAVED, None)

        numpy.set_array_base(a, self)

        return a


    def __dealloc__(self):
        if not self._has_base:
            pfft_free(self.ptr)

cdef class Plan(object):
    cdef pfft_plan plan
    cdef readonly object flags
    cdef readonly object type
    cdef readonly object direction
    cdef readonly int inplace
    cdef pfft_free_plan_func free_func

    def __init__(self, Partition partition, direction, 
            LocalBuffer i, LocalBuffer o=None, 
            type=None, flags=None):
        """ initialize a Plan.
            o defaults to i
            type defaults to parititon.type
            flags defaults to partition.flags

            The usually convention is:

                iDFT: Direction.PFFT_FORWARD
                DFT : Direction.PFFT_BACKWARD

                CFT = dx * iDFT = (L/N) * iDFT
                iCFT = dk * DFT = (2pi / L) * DFT

                We have these normalizations:
                iDFT(DFT) = N

                iCFT(CFT) = dx * dk * DFT(iDFT)
                          = L / N * (2pi / L) * N
                          = 2 pi

            example:
                plan = Plan(partition, Direction.PFFT_FORWARD, buf1, buf2)

        """
        self.direction = Direction(direction)

        if type is None:
            if self.direction == Direction.BACKWARD:
                type = partition.type.inverse
            else:
                type = partition.type

        self.type = Type(type)

        n = partition.n
        cdef ProcMesh procmesh = partition.procmesh

        if flags is None:
            flags = partition.flags

            if self.type.is_inverse_of(partition.type) and self.direction == Direction.BACKWARD:
                if partition.flags & Flags.PFFT_TRANSPOSED_IN:
                    flags = flags & ~Flags.PFFT_TRANSPOSED_IN
                    flags |= Flags.PFFT_TRANSPOSED_OUT

                if partition.flags & Flags.PFFT_TRANSPOSED_OUT:
                    flags = flags & ~Flags.PFFT_TRANSPOSED_OUT
                    flags |= Flags.PFFT_TRANSPOSED_IN

        self.flags = Flags(flags)

        cdef pfft_plan_func     plan_func = <pfft_plan_func>         PFFT_PLAN_FUNC[self.type]
        cdef pfft_plan_func_r2r plan_func_r2r = <pfft_plan_func_r2r> PFFT_PLAN_FUNC[self.type]

        cdef numpy.intp_t [::1] n_ = numpy.array(n, dtype='intp')
        if o is None:
            o = i
        if o.ptr == i.ptr:
            self.inplace = True
        else:
            self.inplace = False

        if ( (self.type in (Type.PFFT_R2C, Type.PFFT_C2R, Type.PFFTF_R2C, Type.PFFT_C2R) )
         and (self.flags & Flags.PFFT_PRESERVE_INPUT)
         and not (self.flags & Flags.PFFT_PADDED_R2C)
         and not self.inplace
        ):
            raise NotImplementedError("out place non-padded r2c / c2r does not preserve input.(%s) " % repr(self.flags)
                                    + "Provide PFFT_DESTROY_INPUT as a flag and deal with this quirk.")

        cdef int [::1] kinds = numpy.zeros(len(n), dtype='int32')

        if direction == Direction.FORWARD:
            kinds[...] = _FFTW_R2HC
        else:
            kinds[...] = _FFTW_HC2R

        if self.type in (Type.R2R, Type.R2RF):
            self.plan = plan_func_r2r(n_.shape[0], &n_[0], i.ptr, o.ptr,
                    procmesh.ccart,
                    &kinds[0],
                    flags)
        else:
            self.plan = plan_func(n_.shape[0], &n_[0], i.ptr, o.ptr,
                    procmesh.ccart,
                    self.direction,
                    flags)
        if not self.plan:
            raise ValueError("Plan is not created")

        self.free_func = PFFT_FREE_PLAN_FUNC[self.type]

    def execute(self, LocalBuffer i, LocalBuffer o=None):
        """ execute a plan.
            o and i must match the alignment (unchecked), 
            inplace status of the plan.
        """
        cdef pfft_execute_func func = PFFT_EXECUTE_FUNC[self.type]
        if o is None:
            o = i
        if o.ptr == i.ptr:
            inplace = True
        else:
            inplace = False
        if inplace != self.inplace:
            raise ValueError("inplace status mismatch with the plan")

        func(self.plan, i.ptr, o.ptr)

    def __repr__(self):
        return "Plan(" + \
                ','.join([
                'flags = %s' % repr(self.flags),
                'type = %s' % repr(self.type),
                'direction = %s' % repr(self.direction),
                'inplace = %s' % repr(self.inplace),
                ]) + ")"
    def __dealloc__(self):
        if self.plan:
            self.free_func(self.plan)
 
pfft_init()
pfftf_init()
#print 'init pfft'
