# dispersal_updates.pyx

""" Function to test that dispersal is correct. 
    TO BE UPDATED IF DISPERSAL APPROACH CHANGES """

import numpy as np
cimport numpy as np
cimport cython

from .structs cimport CellsByType, State, SpatialStructureCitrus, Management
from .extending_distributions cimport SimulationRNG

cpdef np.ndarray sdd_test(np.int32_t cell, SpatialStructureCitrus space):
    """ Updates pathogen short-distance dispersal rates after changes in vector density and/or infectiousness in [cell] """

    cdef Py_ssize_t i, n_iter = space.patKern.shape[0]
    cdef np.int32_t idx, r, c, r_idx, c_idx
    cdef np.ndarray output = np.full((n_iter, 3), np.nan)
    cdef np.float64_t[:,::1] output_view = output

    r, c = space.coord_landscape[cell,:]

    # Iterate over kernel mask (full-mask)
    for i in range(n_iter):
        kern_value = space.patKern[i]
        r_idx = r + space.sd_mask_coord_pat[i,0]
        c_idx = c + space.sd_mask_coord_pat[i,1]
        if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
            idx = space.landscape2d[r_idx, c_idx]
        else:
            idx = -2
        
        output_view[i, 0] = r_idx
        output_view[i, 1] = c_idx
        output_view[i, 2] = idx

    return output


cpdef np.ndarray mdd_test(np.int32_t oIdx, np.float64_t[::1] mask, np.int16_t[:,::1] coord_mask, SpatialStructureCitrus space, SimulationRNG rng_sim, np.int32_t n_iter):
    """ Finds the destination of a mid-distance dispersal event. """
    cdef np.int32_t o_r, o_c
    cdef np.int32_t d_r, d_c
    cdef np.ndarray output = np.full((n_iter, 3), np.nan)
    cdef np.float64_t[:,::1] output_view = output
    cdef np.float64_t target_value 
    cdef Py_ssize_t mid, upper, lower

    for i in range(n_iter):
        upper = mask.shape[0] - 1
        lower = 0
        target_value = rng_sim.next_uniform() * mask[mask.shape[0] - 1]

        while lower < upper:
            mid = (lower + upper) // 2
            if mask[mid] >= target_value:
                upper = mid
            else:
                lower = mid + 1

        o_r, o_c = space.coord_landscape[oIdx,:]
                
        d_r = o_r + coord_mask[lower,0]
        d_c = o_c + coord_mask[lower,1]

        # check that destination is within the landscape
        if (0 <= d_r < space.max_row) and (0 <= d_c < space.max_col):
            idx = space.landscape2d[d_r, d_c]
        else:
            idx = -2

        output_view[i, 0] = d_r
        output_view[i, 1] = d_c
        output_view[i, 2] = idx

    return output
    
