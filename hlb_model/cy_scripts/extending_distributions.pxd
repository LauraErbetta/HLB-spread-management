# structs_HLB.pxd
import numpy as np
cimport numpy as np
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from libc.stdint cimport uint16_t, uint64_t
from numpy.random cimport bitgen_t, BitGenerator
from numpy.random import PCG64
from numpy.random.c_distributions cimport (
      random_standard_uniform_fill, random_standard_uniform, random_exponential, random_standard_t)

np.import_array()

ctypedef fused NumType:
    np.int8_t
    np.int16_t
    np.int32_t
    np.uint8_t
    np.uint16_t
    np.uint32_t

cdef class SimulationRNG:
    cdef BitGenerator bg
    cdef bitgen_t *rng

    cdef double next_uniform(self)
    cdef void fill_uniform(self, double[::1] buf, Py_ssize_t n)
    cdef double next_rexp(self, double scale)
    cdef double next_t(self, double df)
    cpdef NumType[::1] random_choice_without_replacement(self, NumType[::1] population, int sample_size)
    cpdef NumType[::1] sample_items_from_weighted_groups(self, NumType[::1] groups, np.float64_t[::1] weights_per_group, NumType sample_size)

# cpdef NumType[::1] random_choice_weighted_without_replacement(NumType[::1] population, NumType[::1] weights, int sample_size, BitGenerator bg)