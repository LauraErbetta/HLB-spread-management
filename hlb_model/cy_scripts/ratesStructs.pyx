# ratesStructs.pyx

import numpy as np
cimport numpy as np

np.import_array()

# RateTree class
cdef class SqrtBlocks_with_base:
    """ Stores values of rates and methods to use them.
     - Uses Sqrt-Decomposition to have O(1) updates and O(sqrt(N)) idx searches --> Best when batch updates are common (e.g., rates affected by neighbouring cells). 
     - Saves 'base' values of rates, to be multiplied by flushing values to obtain 'real' rates. 
     However, updates tend to introduce approximation errors that may become relevant when rates are small. Generally, does not save total rate 0 when rates eventually are zeroed through simulation. """
        
    def __cinit__(self, np.int32_t size, np.int32_t n_blocks, np.int32_t[::1] blocks_limits):
        # Initialize class
        self.nRates = size
        self.nBlocks = n_blocks
        self.flush = 1.0
        self.total_rate = 0.0
        self.total_no_flush = 0.0
        self.blocks_limits = blocks_limits

        # Arrays
        self._rates = np.zeros(self.nRates, dtype = np.float64)
        self._blocks_totals = np.zeros(self.nBlocks, dtype = np.float64)

        # Memoryviews
        self.rates = self._rates
        self.blocks_totals = self._blocks_totals

    cpdef void recompute_totals(self):
        """ Recompute all block totals and global total. """
        cdef Py_ssize_t i, start, size
        cdef double block_sum, total = 0.0

        for i in range(self.nBlocks):
            start = self.blocks_limits[i]
            end = self.blocks_limits[i+1]
            block_sum = 0.0
            for j in range(start, end):
                block_sum += self.rates[j]
            self.blocks_totals[i] = block_sum
            total += block_sum

        self.total_no_flush = total
        self.total_rate = self.total_no_flush * self.flush

    cpdef void change_flush(self, np.float64_t flush):
        """ Change flush value"""
        self.flush = flush
        self.total_rate = self.total_no_flush * self.flush
    
    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx """
        cdef Py_ssize_t b = idx // ((self.nRates + self.nBlocks - 1) // self.nBlocks) # blocks it belongs to
        cdef np.float64_t delta
        delta = rate - self.rates[idx]
        self.rates[idx] = rate

        self.blocks_totals[b] += delta
        self.total_no_flush += delta
        self.total_rate = self.total_no_flush * self.flush

    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx but does not update total rate"""
        self.rates[idx] = rate
    
    cpdef void zeroRates(self):
        self._rates.fill(0.0)
        self._blocks_totals.fill(0.0)
        self.flush = 1.0
        self.total_rate = 0.0
        self.total_no_flush = 0.0

    cpdef np.int32_t idxSearch(self, np.float64_t rate):
        """ Search first idx where cum_rates[idx] > rate"""
        cdef np.int32_t b, j
        cdef double acc = 0.0

        # Binary search for block
        for b in range(self.nBlocks):
            if acc + self.blocks_totals[b] * self.flush > rate:
                for j in range(self.blocks_limits[b], self.blocks_limits[b+1]):
                    acc += self.rates[j] * self.flush
                    if acc > rate:
                        return j
            acc += self.blocks_totals[b] * self.flush

        raise IndexError("Target rate not in selected rate vector")
    
    cpdef np.ndarray get_rates_array(self):
        """ Return rates as a NumPy array"""
        return self._rates * self.flush

    cpdef np.ndarray get_rates_array_noflush(self):
        """ Return rates as a NumPy array (without flush effect)"""
        return self._rates

cdef class SqrtBlocks:
    """ Stores values of rates and methods to use them.
     - Uses Sqrt-Decomposition to have O(1) updates and O(sqrt(N)) idx searches --> Best when batch updates are common (e.g., rates affected by neighbouring cells). However, updates tend to introduce approximation errors that may become relevant when rates are small. Generally, does not save total rate 0 when rates eventually are zeroed through simulation.
     """

    def __cinit__(self, np.int32_t size, np.int32_t n_blocks, np.int32_t[::1] blocks_limits):
        # Initialize class
        self.nRates = size
        self.nBlocks = n_blocks
        self.total_rate = 0.0
        self.blocks_limits = blocks_limits

        # Arrays
        self._rates = np.zeros(self.nRates, dtype = np.float64)
        self._blocks_totals = np.zeros(self.nBlocks, dtype = np.float64)

        # Memoryviews
        self.rates = self._rates
        self.blocks_totals = self._blocks_totals

    cpdef void recompute_totals(self):
        """ Recompute all block totals and global total. """
        cdef Py_ssize_t i, start, size
        cdef double block_sum, total = 0.0

        for i in range(self.nBlocks):
            start = self.blocks_limits[i]
            end = self.blocks_limits[i+1]
            block_sum = 0.0
            for j in range(start, end):
                block_sum += self.rates[j]
            self.blocks_totals[i] = block_sum
            total += block_sum

        self.total_rate = total
    
    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx """
        cdef Py_ssize_t b = idx // ((self.nRates + self.nBlocks - 1) // self.nBlocks) # blocks it belongs to
        cdef np.float64_t delta

        delta = rate - self.rates[idx]
        self.rates[idx] = rate
        self.blocks_totals[b] += delta
        self.total_rate += delta

    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx but does not update total rate"""
        self.rates[idx] = rate
    
    cpdef void zeroRates(self):
        self._rates.fill(0.0)
        self._blocks_totals.fill(0.0)
        self.total_rate = 0.0

    cpdef np.int32_t idxSearch(self, np.float64_t rate):
        """ Search first idx where cum_rates[idx] > rate"""
        cdef np.int32_t b, j
        cdef double acc = 0.0

        # Binary search for block
        for b in range(self.nBlocks):
            if acc + self.blocks_totals[b] > rate:
                for j in range(self.blocks_limits[b], self.blocks_limits[b+1]):
                    acc += self.rates[j]
                    if acc > rate:
                        return j
            acc += self.blocks_totals[b]

        raise IndexError("Target rate not in selected rate vector")
    
    cpdef np.ndarray get_rates_array(self):
        """ Return rates as a NumPy array"""
        return self._rates

cdef class SegTree_with_base:
    """ Stores values of rates and methods to use them.
    - Uses a Segmented Tree to have O(log(N)) updates and O(log(N)) idx searches --> Best when individual updates are common (e.g., rates affected by changes in one cell only). 
    - Saves 'base' values of rates, to be multiplied by flushing values to obtain 'real' rates. """

    def __cinit__(self, long size):
        # Initialize tree structure
        self.nRates = size
        
        # Pad array to minimum length that is a power of 2
        self.nTreeLevels = 1
        self.nPaddedLength = 1
        while self.nPaddedLength < size:
            self.nPaddedLength *= 2
            self.nTreeLevels += 1
        
        # Allocate memory for rates array (note: rates_length is length of whole tree array)
        self.flush = 1.0
        self.tree_length = 2 * self.nPaddedLength - 1
        self._rates = np.zeros(self.tree_length, dtype = np.float64)    # whole tree
        self.rates = self._rates
        self.total_rate = 0.0

        # Flags for batch updates (sporadic - do not use often, it's generally slower than block updates - limit to management consequences)
        self._changed_flags = np.zeros(self.nPaddedLength, dtype=np.int8)
        self.changed_flags = self._changed_flags
        self.update = 0

    cpdef void change_flush(self, np.float64_t flush):
        self.flush = flush
        self.total_rate = self.rates[2 * self.nPaddedLength - 2] * self.flush

    cpdef void zeroRates(self):
        self._rates.fill(0.0)
        self.total_rate = 0.0
        self.flush = 1.0
        self._changed_flags.fill(0.0)
        self.update = 0

    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx and propagates change throughout the tree """
        cdef int iLevelStart = 0
        cdef int nLevelLength = self.nPaddedLength
        
        self.rates[idx] = rate

        # Standardize idx to always refer to the left child
        idx = (idx // 2) * 2
        while nLevelLength > 1:
            
            # Parent rate is sum of children real rates
            tree_change = (
                self.rates[iLevelStart + idx] + 
                self.rates[iLevelStart + idx + 1]
            )
            
            parent_idx = iLevelStart + nLevelLength + (idx // 2)
            self.rates[parent_idx] = tree_change
        
            # Move up the tree
            iLevelStart += nLevelLength
            nLevelLength //= 2
            idx //= 2
            idx = (idx // 2) * 2

        self.total_rate = self.rates[2 * self.nPaddedLength - 2] * self.flush

    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx but does not update total rate"""
        self.rates[idx] = rate
        self.changed_flags[idx] = 1
        self.update = 1

    cpdef void recompute_totals(self):
        """ Recomputes the tree by propagate only the internal nodes affected by changed leaves.
            self.changed_flags[i] = 1 if leaf i changed, else 0
        """
        cdef np.int32_t level_len = self.nPaddedLength
        cdef np.int32_t start = 0
        cdef np.int32_t i, parent_idx
        cdef np.int8_t[:] next_flags
        cdef np.int8_t[:] flags = self.changed_flags

        if self.update:
            while level_len > 1:
                next_flags = np.zeros(level_len // 2, dtype=np.int8)
                for i in range(0, level_len, 2):    # check every two children
                    # Only update parent if either child changed
                    if flags[i] or flags[i + 1]:
                        parent_idx = start + level_len + i // 2
                        self.rates[parent_idx] = self.rates[start + i] + self.rates[start + i + 1]
                        next_flags[i // 2] = 1  # mark parent as changed
                        # Reset flags
                        flags[i] = 0
                        flags[i+1] = 0
                # Move up one level
                flags = next_flags
                start += level_len
                level_len //= 2

            self.total_rate = self.rates[2 * self.nPaddedLength - 2] * self.flush
            self.update = 0

    cpdef np.int32_t idxSearch(self, np.float64_t rate):
        """ Search first idx where cum_rates[idx] > rate"""
        cdef long nIndex = 0
        cdef int iLevelStart = 2 * self.nPaddedLength - 2
        cdef int nLevelLength = 1
        cdef double dLeftRate
        
        while nLevelLength < self.nPaddedLength:
            nLevelLength = nLevelLength * 2
            iLevelStart -= nLevelLength
            nIndex *= 2
            
            dLeftRate = self.rates[iLevelStart + nIndex] * self.flush
            
            if dLeftRate <= rate:
                nIndex += 1
                rate -= dLeftRate
                
        return nIndex
    
    cpdef np.ndarray get_rates_array(self):
        """ Return rates as a NumPy array"""
        return self._rates[:self.nRates] * self.flush

    cpdef np.ndarray get_rates_array_noflush(self):
        """ Return rates as a NumPy array (without flush effect)"""
        return self._rates[:self.nRates]

cdef class SegTree:
    """ Stores values of rates and methods to use them.
    - Uses a Segmented Tree to have O(log(N)) updates and O(log(N)) idx searches --> Best when individual updates are common (e.g., rates affected by changes in one cell only). 
    """

    def __cinit__(self, long size):
        # Initialize tree structure
        self.nRates = size
        
        # Pad array to minimum length that is a power of 2
        self.nTreeLevels = 1
        self.nPaddedLength = 1
        while self.nPaddedLength < size:
            self.nPaddedLength *= 2
            self.nTreeLevels += 1
        
        # Allocate memory for rates array (note: rates_length is length of whole tree array)
        self.tree_length = 2 * self.nPaddedLength - 1
        self._rates = np.zeros(self.tree_length, dtype = np.float64)    # whole tree
        self.rates = self._rates
        self.total_rate = 0.0

        # Flags for batch updates (sporadic - do not use often, it's generally slower than block updates - limit to management consequences)
        self._changed_flags = np.zeros(self.nPaddedLength, dtype=np.int8)
        self.changed_flags = self._changed_flags
        self.update = 0

    cpdef void zeroRates(self):
        self._rates.fill(0.0)
        self.total_rate = 0.0
        self._changed_flags.fill(0.0)
        self.update = 0

    cpdef void submitRate(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx and propagates change throughout the tree """
        cdef int iLevelStart = 0
        cdef int nLevelLength = self.nPaddedLength

        self.rates[idx] = rate

        # Standardize idx to always refer to the left child
        idx = (idx // 2) * 2
        while nLevelLength > 1:
            
            # Parent rate is sum of children real rates
            tree_change = (
                self.rates[iLevelStart + idx] + 
                self.rates[iLevelStart + idx + 1]
            )
            
            parent_idx = iLevelStart + nLevelLength + (idx // 2)
            self.rates[parent_idx] = tree_change
        
            # Move up the tree
            iLevelStart += nLevelLength
            nLevelLength //= 2
            idx //= 2
            idx = (idx // 2) * 2

        self.total_rate = self.rates[2 * self.nPaddedLength - 2]

    cpdef void submitRate_noTotal(self, np.int32_t idx, np.float64_t rate):
        """ Change rate value in position idx but does not update total rate"""
        self.rates[idx] = rate
        self.changed_flags[idx] = 1
        self.update = 1

    cpdef void recompute_totals(self):
        """ Recomputes the tree by propagate only the internal nodes affected by changed leaves.
            self.changed_flags[i] = 1 if leaf i changed, else 0
        """
        cdef np.int32_t level_len = self.nPaddedLength
        cdef np.int32_t start = 0
        cdef np.int32_t i, parent_idx
        cdef np.int8_t[:] next_flags
        cdef np.int8_t[:] flags = self.changed_flags

        if self.update:
            while level_len > 1:
                next_flags = np.zeros(level_len // 2, dtype=np.int8)
                for i in range(0, level_len, 2):    # check every two children
                    # Only update parent if either child changed
                    if flags[i] or flags[i + 1]:
                        parent_idx = start + level_len + i // 2
                        self.rates[parent_idx] = self.rates[start + i] + self.rates[start + i + 1]
                        next_flags[i // 2] = 1  # mark parent as changed
                        # Reset flags
                        flags[i] = 0
                        flags[i+1] = 0
                # Move up one level
                flags = next_flags
                start += level_len
                level_len //= 2

            self.total_rate = self.rates[2 * self.nPaddedLength - 2]
            self.update = 0

    cpdef np.int32_t idxSearch(self, np.float64_t rate):
        """ Search first idx where cum_rates[idx] > rate"""
        cdef long nIndex = 0
        cdef int iLevelStart = 2 * self.nPaddedLength - 2
        cdef int nLevelLength = 1
        cdef double dLeftRate
        
        while nLevelLength < self.nPaddedLength:
            nLevelLength = nLevelLength * 2
            iLevelStart -= nLevelLength
            nIndex *= 2
            
            dLeftRate = self.rates[iLevelStart + nIndex]
            
            if dLeftRate <= rate:
                nIndex += 1
                rate -= dLeftRate
                
        return nIndex
    
    cpdef np.ndarray get_rates_array(self):
        """ Return rates as a NumPy array"""
        return self._rates[:self.nRates]



