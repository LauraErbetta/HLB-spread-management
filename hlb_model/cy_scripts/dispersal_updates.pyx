# dispersal_updates.pyx

import numpy as np
cimport numpy as np
cimport cython

from .structs cimport CellsByType, State, SpatialStructureCitrus, Management
from .extending_distributions cimport SimulationRNG

cpdef void update_pSdR(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, 
                    np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_pInf_og, np.float64_t new_pInf_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth, 
                    np.float64_t old_pInf_oth, np.float64_t new_pInf_oth):
    """ Updates pathogen short-distance dispersal rates after changes in vector density and/or infectiousness in [cell] """
    cdef np.float64_t new_inf, susc_og, susc_oth, kern_value, denom
    cdef Py_ssize_t i
    cdef np.int32_t idx, r, c, r_idx, c_idx

    new_inf = og.rates.rInf * (og.ctrl.inf_res[cell] * (new_vec_og * new_pInf_og - old_vec_og * old_pInf_og) +
                            oth.ctrl.inf_res[cell] * (new_vec_oth * new_pInf_oth - old_vec_oth * old_pInf_oth))
    if new_inf:
        r, c = space.coord_landscape[cell,:]
        # Iterate over kernel mask (full-mask)
        for i in range(space.patKern.shape[0]):
            kern_value = space.patKern[i]
            r_idx = r + space.sd_mask_coord_pat[i,0]
            c_idx = c + space.sd_mask_coord_pat[i,1]
            if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
                idx = space.landscape2d[r_idx, c_idx]
                if idx < 0:
                    continue

                denom = og.host.tot[idx] - og.host.rem[idx] + oth.host.tot[idx] - oth.host.rem[idx]

                if denom > 0:
                    susc_og = og.ctrl.susc_res[idx] * og.host.sus[idx] / denom
                    susc_oth = oth.ctrl.susc_res[idx] * oth.host.sus[idx] / denom

                    if susc_og:
                        new_rate = og.rates.pSdR.rates[idx] + susc_og * kern_value * new_inf
                        og.rates.pSdR.submitRate(idx, new_rate if new_rate > 1e-15 else 0.0)

                    if susc_oth:
                        new_rate = oth.rates.pSdR.rates[idx] + susc_oth * kern_value * new_inf
                        oth.rates.pSdR.submitRate(idx, new_rate if new_rate > 1e-15 else 0.0)
            
cpdef void update_vSdR(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth):
    """ Updates vector short-distance dispersal rates after changes in vector density in [cell] """
    cdef np.float64_t susc_og, susc_oth, kern_value
    cdef np.float64_t delta_vec = new_vec_og - old_vec_og + new_vec_oth - old_vec_oth
    cdef Py_ssize_t i
    cdef np.int32_t idx, r, c, r_idx, c_idx

    if delta_vec:
        r, c = space.coord_landscape[cell,:]
        # Iterate over kernel mask (full-mask)
        for i in range(space.vecKern.shape[0]):
            kern_value = space.vecKern[i]
            r_idx = r + space.sd_mask_coord_vec[i,0]
            c_idx = c + space.sd_mask_coord_vec[i,1]
            if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
                idx = space.landscape2d[r_idx, c_idx]
                if idx < 0:
                    continue

                if (og.host.tot[idx] - og.host.rem[idx] > 0) and (og.vector.status_vec[idx] == State.absent):
                    susc_og = space.clim[idx] * og.ctrl.bctrl[idx] * og.ctrl.spray[idx]
                    new_rate = og.rates.vSdR.rates[idx] + og.rates.rSdd * susc_og * kern_value * delta_vec
                    og.rates.vSdR.submitRate(idx, new_rate if new_rate > 1e-15 else 0.0)

                if (oth.host.tot[idx] - oth.host.rem[idx] > 0) and (oth.vector.status_vec[idx] == State.absent): 
                    susc_oth = space.clim[idx] * oth.ctrl.bctrl[idx] * oth.ctrl.spray[idx]
                    new_rate = oth.rates.vSdR.rates[idx] + og.rates.rSdd * susc_oth * kern_value * delta_vec
                    oth.rates.vSdR.submitRate(idx, new_rate if new_rate > 1e-15 else 0.0)

cpdef tuple update_pSdR_noTotal(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, 
                    np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_pInf_og, np.float64_t new_pInf_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth, 
                    np.float64_t old_pInf_oth, np.float64_t new_pInf_oth):
    """ Updates pathogen short-distance dispersal rates after changes in vector density and/or infectiousness in [cell] """
    cdef np.float64_t new_inf, susc_og, susc_oth, kern_value, denom
    cdef Py_ssize_t i
    cdef np.int32_t idx, r, c, r_idx, c_idx
    cdef np.int8_t flag_og = 0, flag_oth = 0

    new_inf = og.rates.rInf * (og.ctrl.inf_res[cell] * (new_vec_og * new_pInf_og - old_vec_og * old_pInf_og) +
                            oth.ctrl.inf_res[cell] * (new_vec_oth * new_pInf_oth - old_vec_oth * old_pInf_oth))
    if new_inf:
        r, c = space.coord_landscape[cell,:]
        # Iterate over kernel mask (full-mask)
        for i in range(space.patKern.shape[0]):
            kern_value = space.patKern[i]
            r_idx = r + space.sd_mask_coord_pat[i,0]
            c_idx = c + space.sd_mask_coord_pat[i,1]
            if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
                idx = space.landscape2d[r_idx, c_idx]
                if idx < 0:
                    continue

                denom = og.host.tot[idx] - og.host.rem[idx] + oth.host.tot[idx] - oth.host.rem[idx]

                if denom > 0:
                    susc_og = og.ctrl.susc_res[idx] * og.host.sus[idx] / denom
                    susc_oth = oth.ctrl.susc_res[idx] * oth.host.sus[idx] / denom

                    if susc_og:
                        new_rate = og.rates.pSdR.rates[idx] + susc_og * kern_value * new_inf
                        og.rates.pSdR.submitRate_noTotal(idx, new_rate if new_rate > 1e-15 else 0.0)
                        flag_og = 1

                    if susc_oth:
                        new_rate = oth.rates.pSdR.rates[idx] + susc_oth * kern_value * new_inf
                        oth.rates.pSdR.submitRate_noTotal(idx, new_rate if new_rate > 1e-15 else 0.0)
                        flag_oth = 1
            
    return flag_og, flag_oth

cpdef tuple update_vSdR_noTotal(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth):
    """ Updates vector short-distance dispersal rates after changes in vector density in [cell] """
    cdef np.float64_t susc_og, susc_oth, kern_value
    cdef np.float64_t delta_vec = new_vec_og - old_vec_og + new_vec_oth - old_vec_oth
    cdef Py_ssize_t i
    cdef np.int32_t idx, r, c, r_idx, c_idx
    cdef np.int8_t flag_og = 0, flag_oth = 0

    if delta_vec:
        r, c = space.coord_landscape[cell,:]
        # Iterate over kernel mask (full-mask)
        for i in range(space.vecKern.shape[0]):
            kern_value = space.vecKern[i]
            r_idx = r + space.sd_mask_coord_vec[i,0]
            c_idx = c + space.sd_mask_coord_vec[i,1]
            if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
                idx = space.landscape2d[r_idx, c_idx]
                if idx < 0:
                    continue

                if (og.host.tot[idx] - og.host.rem[idx] > 0) and (og.vector.status_vec[idx] == State.absent):
                    susc_og = space.clim[idx] * og.ctrl.bctrl[idx] * og.ctrl.spray[idx]
                    new_rate = og.rates.vSdR.rates[idx] + og.rates.rSdd * susc_og * kern_value * delta_vec
                    og.rates.vSdR.submitRate(idx, new_rate if new_rate > 1e-15 else 0.0)
                    flag_og = 1

                if (oth.host.tot[idx] - oth.host.rem[idx] > 0) and (oth.vector.status_vec[idx] == State.absent): 
                    susc_oth = space.clim[idx] * oth.ctrl.bctrl[idx] * oth.ctrl.spray[idx]
                    new_rate = oth.rates.vSdR.rates[idx] + og.rates.rSdd * susc_oth * kern_value * delta_vec
                    oth.rates.vSdR.submitRate(idx, new_rate if new_rate > 1e-15 else 0.0)
                    flag_oth = 1

    return flag_og, flag_oth

cpdef np.int8_t update_nearby_availability_pMdd(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth):
    cdef Py_ssize_t k = 0
    cdef np.int32_t idx, r, c, r_idx, c_idx
    cdef np.int8_t flag = 0

    r, c = space.coord_landscape[cell, :]
    
    for k in range(space.md_mask_coord_pat.shape[0]):
        r_idx = r + space.md_mask_coord_pat[k, 0]
        c_idx = c + space.md_mask_coord_pat[k, 1]
        if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
            idx = space.landscape2d[r_idx, c_idx]
            if idx < 0:
                continue

            space.susc_pMdd_now[idx] -= 1

            if space.susc_pMdd_now[idx] == 0:
                og.rates.pMdR.submitRate(idx, 0.0)
                oth.rates.pMdR.submitRate(idx, 0.0)
                flag = 1

    return flag

cpdef np.int8_t update_nearby_availability_vMdd(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth):
    cdef Py_ssize_t k = 0
    cdef np.int32_t idx, r, c, r_idx, c_idx
    cdef np.int8_t flag = 0

    r, c = space.coord_landscape[cell, :]
    
    for k in range(space.md_mask_coord_vec.shape[0]):
        r_idx = r + space.md_mask_coord_vec[k, 0]
        c_idx = c + space.md_mask_coord_vec[k, 1]
        if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
            idx = space.landscape2d[r_idx, c_idx]
            if idx < 0:
                continue
        
            space.susc_vMdd_now[idx] -= 1

            if space.susc_vMdd_now[idx] == 0:
                og.rates.vMdR.submitRate(idx, 0.0)
                oth.rates.vMdR.submitRate(idx, 0.0)
                flag = 1
    
    return flag

cpdef tuple update_nearby_availability_removal(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.int8_t val_pat, np.int8_t val_vec):
    """ Updates to mid-distance dispersal when necessary after removal in [cell]. Does both pathogen and vector. Returns flags indicating whether updates were made. """
    cdef Py_ssize_t k = 0
    cdef np.int32_t idx, r, c, r_idx, c_idx
    cdef np.int8_t flag_pMd = 0, flag_vMd = 0

    r, c = space.coord_landscape[cell, :]

    if val_pat > 0:
        for k in range(space.md_mask_coord_pat.shape[0]):
            r_idx = r + space.md_mask_coord_pat[k, 0]
            c_idx = c + space.md_mask_coord_pat[k, 1]
            if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
                idx = space.landscape2d[r_idx, c_idx]
                if idx < 0:
                    continue

                space.susc_pMdd_now[idx] -= val_pat

                if space.susc_pMdd_now[idx] == 0:
                    og.rates.pMdR.submitRate(idx, 0.0)
                    oth.rates.pMdR.submitRate(idx, 0.0)

                    flag_pMd = 1
    
    if val_vec > 0:
        for k in range(space.md_mask_coord_vec.shape[0]):
            r_idx = r + space.md_mask_coord_vec[k, 0]
            c_idx = c + space.md_mask_coord_vec[k, 1]
            if 0 <= r_idx < space.max_row and 0 <= c_idx < space.max_col:
                idx = space.landscape2d[r_idx, c_idx]
                if idx < 0:
                    continue

                space.susc_vMdd_now[idx] -= val_vec

                if space.susc_vMdd_now[idx] == 0:
                    og.rates.vMdR.submitRate(idx, 0.0)
                    oth.rates.vMdR.submitRate(idx, 0.0)

                    flag_vMd = 1

    return flag_pMd, flag_vMd

cpdef void pathogen_arrival(np.int32_t idx, np.int16_t val, CellsByType dest, CellsByType oth, np.float64_t time, Management mng, SpatialStructureCitrus space, SimulationParameters params):

    old_pInf_dest = dest.host.pInf[idx]
    old_pInf_oth = oth.host.pInf[idx]
    dest.update_SE(idx, val, time)
    space.infCells[idx] = 1
    space.infCells1km[space.small_to_big[idx]] = 1

    if dest.host.sus[idx] + oth.host.sus[idx] == 0: 
        update_nearby_availability_pMdd(idx, space, dest, oth)
    
    if mng.vectorFull:
        dest.rates.lddR.submitRate(idx, dest.host.pInf[idx] * dest.ctrl.inf_res[idx] * dest.rates.rLdd * dest.vector.dens_vec[idx])

    if dest.vector.status_vec[idx] == State.colonised:
        update_pSdR(idx, space, dest, oth, dest.vector.dens_vec[idx], dest.vector.dens_vec[idx], old_pInf_dest, dest.host.pInf[idx], 
                    oth.vector.dens_vec[idx], oth.vector.dens_vec[idx], old_pInf_oth, oth.host.pInf[idx])

        if space.susc_pMdd_now[idx] > 0:
            dest.rates.pMdR.submitRate(idx, space.p_md_pat * dest.rates.rInf * dest.ctrl.inf_res[idx] * dest.host.pInf[idx] * dest.vector.dens_vec[idx])

cpdef void vector_arrival(np.int32_t idx, CellsByType dest, CellsByType oth, np.float64_t time, Management mng, SpatialStructureCitrus space):
    cdef np.int32_t i

    # Vector arrival
    dest.update_AR(idx, time)

    if oth.vector.status_vec[idx] != State.absent:
        update_nearby_availability_vMdd(idx, space, dest, oth)
    
    if dest.summary.vectorFull + oth.summary.vectorFull == 2:     # if vector now everywhere
        mng.vectorFull = 1
        dest.rates.lddR.change_flush(mng.current_flush)
        oth.rates.lddR.change_flush(mng.current_flush)
        for i in range(dest.nCells):
            dest.rates.lddR.submitRate_noTotal(i, dest.host.pInf[i] * dest.ctrl.inf_res[i] * dest.rates.rLdd * dest.vector.dens_vec[i])
            oth.rates.lddR.submitRate_noTotal(i, oth.host.pInf[i] * oth.ctrl.inf_res[i] * oth.rates.rLdd * oth.vector.dens_vec[i])
        dest.rates.lddR.recompute_totals() # changes all of them
        oth.rates.lddR.recompute_totals()

cpdef void pat_vec_arrival(np.int32_t idx, CellsByType dest, CellsByType oth, np.float64_t time, Management mng, SpatialStructureCitrus space):
    cdef np.int32_t i

    # Pathogen arrival
    dest.update_SE(idx, 1, time)
    space.infCells[idx] = 1
    space.infCells1km[space.small_to_big[idx]] = 1
    # no need to update pSdR, pMsR because vector just arrived (no vector density --> pInf is unconsequential)

    # Vector arrival
    dest.update_AR(idx, time)

    if dest.host.sus[idx] + oth.host.sus[idx] == 0: 
        update_nearby_availability_pMdd(idx, space, dest, oth)

    if oth.vector.status_vec[idx] != State.absent:
        update_nearby_availability_vMdd(idx, space, dest, oth)

    if dest.summary.vectorFull + oth.summary.vectorFull == 2:     # if vector now everywhere
        mng.vectorFull = 1
        dest.rates.lddR.change_flush(mng.current_flush)
        oth.rates.lddR.change_flush(mng.current_flush)
        for i in range(dest.nCells):
            dest.rates.lddR.submitRate_noTotal(i, dest.host.pInf[i] * dest.ctrl.inf_res[i] * dest.rates.rLdd * dest.vector.dens_vec[i])
            oth.rates.lddR.submitRate_noTotal(i, oth.host.pInf[i] * oth.ctrl.inf_res[i] * oth.rates.rLdd * oth.vector.dens_vec[i])
        dest.rates.lddR.recompute_totals() # changes all of them
        oth.rates.lddR.recompute_totals()

cdef inline np.int32_t find_destination_md(np.int32_t oIdx, np.float64_t[::1] mask, np.int16_t[:,::1] coord_mask, SpatialStructureCitrus space, SimulationRNG rng_sim):
    """ Finds the destination of a mid-distance dispersal event. """
    cdef np.int32_t o_r, o_c
    cdef np.int32_t d_r, d_c
    cdef np.float64_t target_value = rng_sim.next_uniform() * mask[mask.shape[0] - 1]
    cdef Py_ssize_t mid, upper = mask.shape[0] - 1, lower = 0

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
        return space.landscape2d[d_r, d_c]

    return -1

    
    












