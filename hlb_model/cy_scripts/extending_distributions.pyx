# extending_distributions.pyx

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

cdef class SimulationRNG:
    
    def __cinit__(self, BitGenerator bg):
        self.bg = bg
        self.rng = <bitgen_t *> PyCapsule_GetPointer(bg.capsule, "BitGenerator")

    cdef inline double next_uniform(self):
        return random_standard_uniform(self.rng)

    cdef inline void fill_uniform(self, double[::1] buf, Py_ssize_t n):
        random_standard_uniform_fill(self.rng, n, &buf[0])

    cdef inline double next_rexp(self, double scale):
        return random_exponential(self.rng, scale)

    cdef inline double next_t(self, double df):
        return random_standard_t(self.rng, df)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef NumType[::1] random_choice_without_replacement(self, NumType[::1] population, int sample_size):
        """
        Select <sample_size> elements from <population> without replacement.
        """

        cdef int n = <int>population.shape[0]
        cdef np.int32_t[::1] indices = np.arange(n, dtype=np.int32)
        cdef int i, j, tmp
        cdef NumType[::1] result = np.empty_like(population, shape=(sample_size,))
        
        # Implement Fisher-Yates shuffle and take the first sample_size elements
        for i in range(sample_size):
            j = i + <int>(self.next_uniform() * (n - i))
            tmp = indices[i]
            indices[i] = indices[j]
            indices[j] = tmp

            result[i] = population[indices[i]]
            
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef NumType[::1] sample_items_from_weighted_groups(self, NumType[::1] groups, np.float64_t[::1] weights_per_group, NumType sample_size):
        """ Samples <sample_size> items from <groups> with <weights_per_group> as weights. """
        cdef Py_ssize_t i
        cdef NumType j
        cdef double total_weight, cumulative_weight, target_w
        cdef double[::1] random_values = np.empty(sample_size, dtype=np.float64)
        cdef NumType[::1] sampled_per_groups = np.zeros_like(groups)
        cdef NumType total_units = 0

        for i in range(groups.shape[0]):
            total_units += groups[i]
        
        if sample_size > total_units:
            raise ValueError(f"Units to be sampled are more than available units")
        elif sample_size == total_units:
            return groups

        random_standard_uniform_fill(self.rng, sample_size, &random_values[0])

        for j in range(sample_size):
            total_weight = 0.0
            for i in range(groups.shape[0]):
                total_weight += (groups[i] - sampled_per_groups[i]) * weights_per_group[i]
            
            target_w = random_values[j] * total_weight
            cumulative_weight = 0.0

            for i in range(groups.shape[0]):
                cumulative_weight += (groups[i] - sampled_per_groups[i]) * weights_per_group[i]
                if target_w < cumulative_weight:
                    sampled_per_groups[i] += 1
                    break
        
        return sampled_per_groups