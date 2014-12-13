#from mpi4py cimport MPI
from mpi4py import MPI as pyMPI
cimport libmpi as MPI
import numpy
cimport numpy
from libc.stdlib cimport free, calloc
from libc.string cimport memset
numpy.import_array()
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

    void pfft_init()
    void pfft_cleanup()

    ctypedef void * pfft_plan

    struct pfft_complex:
        pass
    void pfft_execute(pfft_plan plan)

    void pfft_execute_dft(pfft_plan plan, void * input, void * output)
    void pfft_execute_dft_r2c(pfft_plan plan, void * input, void * output)
    void pfft_execute_dft_c2r(pfft_plan plan, void * input, void * output)
    void pfft_execute_r2r(pfft_plan plan, void * input, void * output)

    void pfft_destroy_plan(pfft_plan plan)

    pfft_plan pfft_plan_dft(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            MPI.MPI_Comm comm_cart,
            int sign, unsigned pfft_flags)

    pfft_plan pfft_plan_dft_r2c(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            MPI.MPI_Comm comm_cart,
            int sign, unsigned pfft_flags)

    pfft_plan pfft_plan_dft_c2r(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            MPI.MPI_Comm comm_cart,
            int sign, unsigned pfft_flags)

    pfft_plan pfft_plan_r2r(
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
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

#######
#  wrap Flags, Direction
#####

class Flags(int):
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
    def __new__(cls, value):
        self = int.__new__(cls, value)
        return self
    def __repr__(self):
        d = self.__class__.__dict__
        return '|'.join([k for k in d.keys() if k.startswith('PFFT') and (d[k] & self)])

class Direction(int):
    PFFT_FORWARD = _PFFT_FORWARD
    PFFT_BACKWARD = _PFFT_BACKWARD
    def __new__(cls, value):
        self = int.__new__(cls, value)
        return self
    def __repr__(self):
        d = self.__class__.__dict__
        return 'and'.join([k for k in d.keys() if k.startswith('PFFT') and (d[k] == self)])

######
# define Type as the transform type
# fill in the function tables as well.
##
class Type(int):
    PFFT_C2C = 0
    PFFT_R2C = 1
    PFFT_C2R = 2
    PFFT_R2R = 3
    def __new__(cls, value):
        self = int.__new__(cls, value)
        return self
    def __repr__(self):
        d = self.__class__.__dict__
        return 'and'.join([k for k in d.keys() if k.startswith('PFFT') and (d[k] == self)])

ctypedef numpy.intp_t (*pfft_local_size_func)(int rnk_n, numpy.intp_t * n, MPI.MPI_Comm comm, int
            pfft_flags, numpy.intp_t * local_ni, numpy.intp_t * local_i_start,
            numpy.intp_t* local_no, numpy.intp_t * local_o_start)
cdef pfft_local_size_func PFFT_LOCAL_SIZE_FUNC [4]

PFFT_LOCAL_SIZE_FUNC[:] = [
    <pfft_local_size_func> pfft_local_size_dft,
    <pfft_local_size_func> pfft_local_size_dft_r2c,
    <pfft_local_size_func> pfft_local_size_dft_c2r,
    <pfft_local_size_func> pfft_local_size_r2r
        ]

ctypedef pfft_plan (*pfft_plan_func) (
            int rnk_n, numpy.intp_t *n, void * input, void * output, 
            MPI.MPI_Comm comm_cart,
            int sign, unsigned pfft_flags)
cdef pfft_plan_func PFFT_PLAN_FUNC [4]

PFFT_PLAN_FUNC[:] = [
    <pfft_plan_func> pfft_plan_dft,
    <pfft_plan_func> pfft_plan_dft_r2c,
    <pfft_plan_func> pfft_plan_dft_c2r,
    <pfft_plan_func> pfft_plan_r2r
        ]

ctypedef void (*pfft_execute_func) ( pfft_plan plan, void * input, void * output)
cdef pfft_execute_func PFFT_EXECUTE_FUNC [4]

PFFT_EXECUTE_FUNC[:] = [
    <pfft_execute_func> pfft_execute_dft,
    <pfft_execute_func> pfft_execute_dft_r2c,
    <pfft_execute_func> pfft_execute_dft_c2r,
    <pfft_execute_func> pfft_execute_r2r
        ]

cdef class ProcMesh(object):
    cdef MPI.MPI_Comm comm_cart
    cdef readonly numpy.ndarray this # nd rank of the current process
    cdef readonly numpy.ndarray np
    cdef readonly int rank
    cdef readonly int Ndim
    cdef MPI.MPI_Comm * comm_col

    def __init__(self, np, comm=None):
        """ A mesh of processes 
            np is the number of processes in each direction.


            example:
                procmesh = ProcMesh([2, 3]) # creates a 2 x 3 mesh.

            product(np) must equal to comm.size
            
            if the mpi4py version is recent (MPI._addressof), comm can
            be any mpi4py Comm objects.
        """
        cdef MPI.MPI_Comm mpicomm
        self.comm_col = NULL
        self.comm_cart = NULL

        if comm is None:
            mpicomm = MPI.MPI_COMM_WORLD
        else:
            if isinstance(comm, pyMPI.Comm):
                if hasattr(pyMPI, '_addressof'):
                    mpicomm = (<MPI.MPI_Comm*> (<numpy.intp_t>
                            pyMPI._addressof(comm))) [0]
                else:
                    raise ValueError("only comm=None is supported, "
                            + " update mpi4py to a version with MPI._addressof")
            else:
                raise ValueError("only MPI.Comm objects are supported")

        MPI.MPI_Comm_rank(mpicomm, &self.rank)
        cdef int [::1] np_ = numpy.array(np, 'int32')
        rt = pfft_create_procmesh(np_.shape[0], mpicomm, &np_[0], &self.comm_cart)

        if rt != 0:
            self.comm_cart = NULL
            raise RuntimeError("Failed to create proc mesh")

        self.np = numpy.array(np_)
        self.Ndim = len(self.np)

        # a buffer used for various purposes 
        cdef int[::1] junk = numpy.empty(self.Ndim, 'int32')

        # now fill `this'
        self.this = numpy.array(np, 'int32')
        MPI.MPI_Cart_get(self.comm_cart, 2, 
                &junk[0], &junk[0],
                <int*>self.this.data);

        # build the comm_col sub communicators
        self.comm_col = <MPI.MPI_Comm*>calloc(self.Ndim, sizeof(MPI.MPI_Comm))
        for i in range(self.Ndim):
            junk[:] = 0
            junk[i] = 1
            if MPI.MPI_SUCCESS != MPI.MPI_Cart_sub(self.comm_cart, &junk[0],
                    &self.comm_col[i]):
                self.comm_col[i] = NULL
                raise RuntimeError("Failed to create sub communicators")

    def __dealloc__(self):
        if self.comm_cart:
            MPI.MPI_Comm_free(&self.comm_cart)
            pass
        if self.comm_col != NULL:
            for i in range(self.Ndim):
                if self.comm_col[i]:
                    MPI.MPI_Comm_free(&self.comm_col[i])
            free(self.comm_col)

cdef class Partition(object):
    cdef readonly size_t alloc_local
    cdef readonly int Ndim
    cdef readonly numpy.ndarray n
    cdef readonly numpy.ndarray local_ni
    cdef readonly numpy.ndarray local_i_start
    cdef readonly numpy.ndarray local_no
    cdef readonly numpy.ndarray local_o_start
    cdef readonly object type
    cdef readonly object flags
    cdef readonly ProcMesh procmesh
    cdef readonly object i_edges
    cdef readonly object o_edges
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

        local_ni, local_no, local_i_start, local_o_start = \
                numpy.empty((4, n_.shape[0]), 'intp')

        if len(n_) <= len(procmesh.np):
            raise ValueError("ProcMesh (%d) shall have less dimentions than Mesh (%d)" % (len(procmesh.np), len(n_)))

        self.type = Type(type)
        self.flags = Flags(flags)

        cdef pfft_local_size_func func = PFFT_LOCAL_SIZE_FUNC[self.type]


        rt = func(n_.shape[0], 
                &n_[0], 
                procmesh.comm_cart,
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
        self.Ndim = len(self.n)

        self.i_edges = self._build_edges(self.local_i_start,
                self.flags & Flags.PFFT_TRANSPOSED_IN 
                )
        self.o_edges = self._build_edges(self.local_o_start,
                self.flags & Flags.PFFT_TRANSPOSED_OUT
                )

    def _build_edges(self, numpy.intp_t[::1] local_start, transposed):
        cdef numpy.intp_t[::1] start_dim
        cdef numpy.intp_t tmp
        edges = []
        cdef int d
        np = numpy.ones(self.Ndim, dtype='int')
        np[:self.procmesh.Ndim] = self.procmesh.np
        for d in range(self.Ndim):
            if transposed:
                d1 = self.transpose_d(d)
            else:
                d1 = d
            start_dim = numpy.empty((np[d1] + 1), dtype='intp')
            start_dim[0] = 0
            start_dim[np[d1]] = self.n[d1]
            if d1 < self.procmesh.Ndim:
                tmp = local_start[d]
                MPI.MPI_Allgather(&tmp, sizeof(numpy.intp_t), MPI.MPI_BYTE, 
                        &start_dim[0], sizeof(numpy.intp_t), MPI.MPI_BYTE, 
                        self.procmesh.comm_col[d1])
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
                
    def transpose_d(self, d):
        r = len(self.procmesh.np)
        if d >= 1 and d < r + 1:
            return d - 1
        if d == 0:
            return r
        return d
    def transpose_list(self, list):
        """ migrate shape to the transposed ordering """
        assert len(list) == len(self.n)
        r = len(self.procmesh.np)
        n0 = list[0] 
        newlist = list[1:r+1]
        newlist.append(n0)
        newlist += list[r+1:]
        return newlist
    def transpose_shape(self, shape):
        """ migrate shape to the transposed ordering """
        assert len(shape) == len(self.n)
        r = len(self.procmesh.np)
        n0 = shape[0] 
        newshape = numpy.copy(shape)
        newshape[:r] = shape[1:r+1]
        newshape[r] = n0
        newshape[r+1:] = shape[r+1:]
        return newshape
    def restore_transposed_array(self, array):
        """ transposed array from the 'transposed ordering' to the ordinary
        ordering; used by LocalBuffer to return an ordinary looking ndarray"""
        oldaxes = numpy.arange(len(self.n))
        newaxes = self.transpose_shape(oldaxes)
        revert = oldaxes.copy()
        revert[newaxes] = numpy.arange(len(self.n))
        return array.transpose(revert)

cdef class LocalBuffer:
    cdef void * ptr
    property buffer:
        def __get__(self):
            cdef numpy.intp_t shape[1]
            shape[0] = self.partition.alloc_local * 2
            cdef numpy.ndarray buffer = numpy.PyArray_SimpleNewFromData(1, shape, numpy.NPY_DOUBLE, self.ptr)
            numpy.set_array_base(buffer, self)
            return buffer

    cdef readonly Partition partition

    def __init__(self, Partition partition):
        """ The local portion of the distributed array used by PFFT 

            see the documents of view_input, view_output
        """
        self.partition = partition
        self.ptr = pfft_alloc_complex(partition.alloc_local)

    def _view(self, dtype, local_n, local_start, roll, padded):

        cdef numpy.ndarray a = numpy.array(self.buffer, copy=False)

        shape = local_n.copy()
        a = a.view(dtype=dtype)
        if numpy.iscomplexobj(a):
            # complex array needs no padding
            shape2 = shape
            pass
        else:
            # real array needs to worry about padding
            shape2 = shape.copy()
            if padded:
                global_end = local_start + local_n
                if global_end[-1] > self.partition.n[-1]:
                    # This chunk is the last one along the conjugated axis
                    # cut it such that the array shape looks correct (padding is
                    # hidden from user)
                    shape2[-1] = self.partition.n[-1] - local_start[-1]
                #print shape, shape2
        if roll:
            #shift = len(self.partition.procmesh.np) - len(self.partition.n)
            shape = self.partition.transpose_shape(shape)
            shape2 = self.partition.transpose_shape(shape2)

        sel = [slice(0, s) for s in shape2]
        a = a[:numpy.product(shape)]
        a = a.reshape(shape)
        a = a[sel]
        #print <numpy.intp_t> a.data, <numpy.intp_t> self.ptr
        if roll:
            a = self.partition.restore_transposed_array(a)
        return a

    def view_input(self):
        """ return the buffer as a numpy array, the dtype and shape
            are for the input of the transform

            padding is opaque; the returned array has removed the padding column.
            PFFT_TRANSPOSED_IN does not affect the ordering of the axes in
            the returned array. (this is achieved via numpy.transpose)
        """
        dtypes = ['complex128', 'float64', 'complex128', 'float64']
        return self._view(dtypes[self.partition.type],
                self.partition.local_ni,
                self.partition.local_i_start,
                self.partition.flags & Flags.PFFT_TRANSPOSED_IN, 
                self.partition.flags & Flags.PFFT_PADDED_R2C, 
                )
    def view_output(self):
        """ return the buffer as a numpy array, the dtype and shape
            are for the output of the transform

            padding is opaque; the returned array has removed the padding column.
            PFFT_TRANSPOSED_OUT does not affect the ordering of the axes in
            the returned array. (this is achieved via numpy.transpose)
        """
        dtypes = ['complex128', 'complex128', 'float64', 'float64']
        return self._view(dtypes[self.partition.type],
                self.partition.local_no,
                self.partition.local_o_start,
                self.partition.flags & Flags.PFFT_TRANSPOSED_OUT, 
                self.partition.flags & Flags.PFFT_PADDED_C2R, 
                )

    def __dealloc__(self):
        pfft_free(self.ptr)

cdef class Plan(object):
    cdef pfft_plan plan
    cdef readonly object flags
    cdef readonly object type
    cdef readonly object direction
    cdef readonly int inplace
    def __init__(self, Partition partition, direction, 
            LocalBuffer i, LocalBuffer o=None, 
            type=None, flags=None):
        """ initialize a Plan.
            o defaults to i
            type defaults to parititon.type
            flags defaults to partition.flags

            example:
                plan = Plan(partition, Direction.PFFT_FORWARD, buf1, buf2)
        """
        if type is None:
            type = partition.type

        n = partition.n
        cdef ProcMesh procmesh = partition.procmesh
        if flags is None:
            flags = partition.flags

        self.flags = Flags(flags)
        self.type = Type(type)
        self.direction = Direction(direction)

        cdef pfft_plan_func func = PFFT_PLAN_FUNC[self.type]
        cdef numpy.intp_t [::1] n_ = numpy.array(n, dtype='intp')
        if o is None:
            o = i
        if o.ptr == i.ptr:
            self.inplace = True
        else:
            self.inplace = False
        self.plan = func(n_.shape[0], &n_[0], i.ptr, o.ptr,
                procmesh.comm_cart,
                self.direction,
                flags)

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
pfft_init()
#print 'init pfft'
