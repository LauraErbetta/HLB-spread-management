# ratesStructs.pxd
# language_level=3, boundscheck=True, wraparound=True, initializedcheck=True, cdivision=True

cimport numpy as np

np.import_array()

cpdef enum DispType:
    short = 0,
    mid = 1,
    long = 2

cdef class SqrtBlocks_with_base:
    """ Stores values of rates and methods to use them.
     - Uses Sqrt-Decomposition to have O(1) updates and O(sqrt(N)) idx searches --> Best when batch updates are common (e.g., rates affected by neighbouring cells). 
     - Saves 'base' values of rates, to be multiplied by flushing values to obtain 'real' rates. """
    cdef public np.ndarray _rates, _blocks_totals
    cdef public np.float64_t[::1] rates, blocks_totals
    cdef public np.int32_t[::1] blocks_limits
    cdef public np.float64_t flush, total_rate, total_no_flush
    cdef np.int32_t nRates, nBlocks

    cpdef void recompute_totals(self)
    cpdef void change_flush(self, np.float64_t flush)
    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate)
    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate)
    cpdef void zeroRates(self)
    cpdef np.int32_t idxSearch(self, np.float64_t rate)
    cpdef np.ndarray get_rates_array(self)
    cpdef np.ndarray get_rates_array_noflush(self)

cdef class SqrtBlocks:
    """ Stores values of rates and methods to use them.
     - Uses Sqrt-Decomposition to have O(1) updates and O(sqrt(N)) idx searches --> Best when batch updates are common (e.g., rates affected by neighbouring cells). """
    cdef public np.ndarray _rates, _blocks_totals
    cdef public np.float64_t[::1] rates, blocks_totals
    cdef public np.int32_t[::1] blocks_limits
    cdef np.int32_t nRates, nBlocks
    cdef public np.float64_t total_rate
    
    cpdef void recompute_totals(self)
    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate)
    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate) 
    cpdef void zeroRates(self)
    cpdef np.int32_t idxSearch(self, np.float64_t rate)
    cpdef np.ndarray get_rates_array(self)

cdef class SegTree_with_base:
    """ Stores values of rates and methods to use them.
    - Uses a Segmented Tree to have O(log(N)) updates and O(log(N)) idx searches --> Best when individual updates are common (e.g., rates affected by changes in one cell only). 
    - Saves 'base' values of rates, to be multiplied by flushing values to obtain 'real' rates. """
    cdef public np.int32_t nRates, nPaddedLength, tree_length
    cdef public np.uint8_t nTreeLevels
    cdef public np.ndarray _rates, _changed_flags
    cdef public np.float64_t[::1] rates
    cdef public np.int8_t[::1] changed_flags
    cdef public np.int8_t update
    cdef public np.float64_t total_rate, flush

    cpdef void change_flush(self, np.float64_t flush)
    cpdef void zeroRates(self)
    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate)
    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate)
    cpdef void recompute_totals(self)
    cpdef np.int32_t idxSearch(self, np.float64_t rate)
    cpdef np.ndarray get_rates_array(self)
    cpdef np.ndarray get_rates_array_noflush(self)

cdef class SegTree:
    """ Stores values of rates and methods to use them.
    - Uses a Segmented Tree to have O(log(N)) updates and O(log(N)) idx searches --> Best when individual updates are common (e.g., rates affected by changes in one cell only). 
    """
    cdef public np.int32_t nRates, nPaddedLength, tree_length
    cdef public np.uint8_t nTreeLevels
    cdef public np.ndarray _rates, _changed_flags
    cdef public np.float64_t[::1] rates
    cdef public np.int8_t[::1] changed_flags
    cdef public np.int8_t update
    cdef public np.float64_t total_rate

    cpdef void zeroRates(self)
    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate)
    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate)
    cpdef void recompute_totals(self)
    cpdef np.int32_t idxSearch(self, np.float64_t rate)
    cpdef np.ndarray get_rates_array(self)