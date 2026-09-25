# management.pyx

import numpy as np
cimport numpy as np
cimport cython

cimport libc.math as c_math
from .extending_distributions cimport SimulationRNG
from .structs cimport SimulationSetup, SimulationParameters, SpatialStructureCitrus, Management, SurveyHelperClass, CellsByType, State, Event, RatesIdx, SurveyType, SimGoal
from .dispersal_updates cimport update_pSdR_noTotal, update_vSdR_noTotal, update_nearby_availability_pMdd, update_nearby_availability_vMdd
from .linkedList cimport LinkedList, EventNode

from libc.stdio cimport fprintf, stderr, fflush


cpdef np.uint8_t host_detection(np.uint8_t nVis, np.float64_t propPCR, np.int32_t[::1] surveyed, np.uint16_t[:, ::1] detected, CellsByType og, np.float64_t pVis, np.float64_t[::1] pPCR, SurveyHelperClass shc, SimulationRNG rng_sim):
    """ Computes random detection of I, C and D host units in surveyed cells.
    Args:
        nVis: number of total units (standing) visually inspected
        propPCR: proportion of visually inspected units also tested via PCR if negative visual
        surveyed: id of cells where inspection takes place
        og: class of the inspected citrus type
        pPCR: probability of PCR detection
        pVIS: probability of visual detection
        rng_sim: Random Numbers Generator
    Returns:
        detected: 2D memoryview (rows: I, C, D; cols: cells)
    Notes:
        a. detected spans across all cells, not just surveyed
        b. nVis and nPCR are taken from the same sample. For instance, if nPCR < nVis, the units sampled for PCR are among those visually inspected. """

    cdef np.uint8_t[::1] unit_types = np.empty(og.summary.maxCit, dtype=np.uint8)     # preallocate
    cdef Py_ssize_t i, idx, pos, j, idx_rand
    cdef np.int16_t cell_I, cell_C, cell_N, tot_units
    cdef np.int16_t n_actual_vis, n_actual_pcr, n_pcr_done
    cdef np.uint8_t found = 0
    cdef np.uint8_t[::1] sampled_units
    cdef np.float64_t[::1] rand_values = np.empty(og.summary.maxCit * 2 * surveyed.shape[0], dtype=np.float64)  # max detection attempts

    rng_sim.fill_uniform(rand_values, rand_values.shape[0])

    
    for i in range(surveyed.shape[0]):
        idx = surveyed[i]
        cell_I = og.host.inf[idx] - detected[0,idx] 
        cell_C = og.host.crp[idx] - detected[1,idx]
        cell_N = og.host.exp[idx] + og.host.sus[idx]  # not detectable
        tot_units = cell_I + cell_C + cell_N

        if tot_units == 0:
            continue

        # Random sampling of units
        n_actual_vis = nVis if nVis < tot_units else tot_units
        n_actual_pcr = <np.int16_t>c_math.ceil(propPCR * n_actual_vis)
        shc.n_vis += n_actual_vis
        if (cell_I + cell_C == 0):
            shc.n_pcr += n_actual_pcr
            continue
            
        pos = 0
        for j in range(cell_I):
            unit_types[pos] = 0
            pos += 1
        for j in range(cell_C):
            unit_types[pos] = 1
            pos += 1
        for j in range(cell_N):
            unit_types[pos] = 3
            pos += 1

        # sampled_units = np.random.choice(unit_types[:tot_units], n_actual_vis, replace=False)  # sample without replacement

        sampled_units = rng_sim.random_choice_without_replacement(unit_types[:tot_units], n_actual_vis)

        n_pcr_done = 0
        for j in range(n_actual_vis):
            idx_rand = (og.summary.maxCit * 2) * i + j
            if sampled_units[j] == 0 and rand_values[idx_rand] < pVis:
                detected[0, idx] += 1
                shc.cell_success_host[idx] = 1
                found = 1
            elif n_pcr_done < n_actual_pcr:
                idx_rand = (og.summary.maxCit * 2) * i + og.summary.maxCit + n_pcr_done
                if sampled_units[j] == 0 and rand_values[idx_rand] < pPCR[1]:
                    detected[0, idx] += 1
                    shc.cell_success_host[idx] = 1
                    found = 1
                elif sampled_units[j] == 1 and rand_values[idx_rand] < pPCR[0]:
                    detected[1, idx] += 1
                    shc.cell_success_host[idx] = 1
                    found = 1
                n_pcr_done += 1
                shc.n_pcr += 1
    return found

cpdef remove_detected(np.float64_t time, SurveyHelperClass shc, np.uint8_t[::1] rates_flags_com, np.uint8_t[::1] rates_flags_rsd, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SimulationParameters params, SimulationRNG rng_sim):
    """ Removes detected units and units within radius (via removal mask).
        Does not update rates, as they are updated separetely after spraying.
    Notes:
        a. Removal within radius is in addition to detected units (i.e., detected units are not counted as units removed within radius) """

    cdef np.int32_t idx, idx_og, nCells = com.nCells
    cdef np.int32_t r, c, r_rel, c_rel, r_rem, c_rem
    cdef Py_ssize_t i, z, w
    cdef np.int8_t removed = 0, temp_pat = 0, temp_vec = 0, flag_com = 0, flag_rsd = 0
    cdef np.float64_t compute_amount, old_vec_com, old_vec_rsd, old_pInf_com, old_pInf_rsd, susc_ratio
    cdef np.uint16_t r_com_I, r_com_C, r_com_E, r_com_S, r_rsd_I, r_rsd_C, r_rsd_E, r_rsd_S, tot_left_com, tot_left_rsd, tot_left
    cdef np.uint16_t remove_amount
    cdef np.int8_t update_pMd_com = 0, update_vMd_com = 0, update_pMd_rsd = 0, update_vMd_rsd = 0
    cdef np.uint8_t[::1] unit_types = np.zeros(com.summary.maxCit + rsd.summary.maxCit, dtype=np.uint8)     # preallocate to avoid calling np.zeros too many times
    cdef np.uint16_t[::1] sampled_units
    cdef np.uint16_t[::1] group_array = np.zeros(8, np.uint16)
    cdef np.float64_t[::1] prop_remove = np.zeros(nCells, np.float64)

    if mng.keep_attempt and mng.switchRemoval and time - mng.time_det >= params.removal_duration:
        # Stop removing with radius, switch to removing just detected units
        mng.keep_attempt = 0
        space.rem_mask_to_use = space.default_0_rem_mask
        mng.time_stop_attempt = time

    for i in range(nCells):
        if shc.cell_success_host[i]:
            r, c = space.coord_landscape[i,:]
            for w in range(space.rem_mask_to_use.shape[0]):
                r_rel, c_rel = space.rem_coord[w,:]
                for z in range(8):
                    r_rem = r + r_rel * space.coord_transf[z,0] + c_rel * space.coord_transf[z,1]
                    c_rem = c + r_rel * space.coord_transf[z,2] + c_rel * space.coord_transf[z,3]
                    if 0 <= r_rem < space.max_row and 0 <= c_rem < space.max_col:
                        idx_rem = space.landscape2d[r_rem, c_rem]
                        if idx_rem < 0:
                            continue
                        prop_remove[idx_rem] += space.rem_mask_to_use[w]


    for idx in range(nCells):    # goes through all cells        

        # Compute how much to remove based on removal radius (commercial and residential)
        tot_left_com = com.host.tot[idx] - com.host.rem[idx]
        tot_left_rsd = rsd.host.tot[idx] - rsd.host.rem[idx]
        tot_left = tot_left_com + tot_left_rsd
        compute_amount = c_math.round((prop_remove[idx] if prop_remove[idx] < 1 else 1) * (com.host.tot[idx] + rsd.host.tot[idx]))
        remove_amount = <np.uint16_t>(compute_amount if compute_amount < tot_left else tot_left)

        if (remove_amount > 0 or shc.cell_success_host[idx]) and (rng_sim.next_uniform() < params.pRem):
            shc.add_demarcated[idx] = 1
            space.infCells1km[space.small_to_big[idx]] = 1 # removal still affects cells
            # Reset flags
            update_pMd_com = 0
            update_vMd_com = 0
            update_pMd_rsd = 0
            update_vMd_rsd = 0
            
            # Reset removal temp variables
            r_com_I = 0
            r_com_C = 0
            r_com_E = 0
            r_com_S = 0
            r_rsd_I = 0
            r_rsd_C = 0
            r_rsd_E = 0
            r_rsd_S = 0
            old_pInf_com = com.host.pInf[idx]
            old_pInf_rsd = rsd.host.pInf[idx]
            old_vec_com = com.vector.dens_vec[idx]
            old_vec_rsd = rsd.vector.dens_vec[idx]

            if shc.cell_success_host[idx]:
                # Remove detected units
                r_com_I += shc.comDetected[0, idx]
                r_com_C += shc.comDetected[1, idx]
                r_rsd_I += shc.rsdDetected[0, idx]
                r_rsd_C += shc.rsdDetected[1, idx]

            remove_amount = remove_amount - r_com_I - r_com_C - r_rsd_I - r_rsd_C if remove_amount > (r_com_I + r_com_C + r_rsd_I + r_rsd_C) else 0
            tot_left -= (r_com_I + r_com_C + r_rsd_I + r_rsd_C)

            if remove_amount == tot_left : # remove everything
                r_com_I = com.host.inf[idx]
                r_com_C = com.host.crp[idx]
                r_com_E = com.host.exp[idx]
                r_com_S = com.host.sus[idx]

                r_rsd_I = rsd.host.inf[idx]
                r_rsd_C = rsd.host.crp[idx]
                r_rsd_E = rsd.host.exp[idx]
                r_rsd_S = rsd.host.sus[idx]

            elif remove_amount > 0 and tot_left > 0:
                # Randomly select commercial host units to remove
                group_array[0] = com.host.inf[idx] - r_com_I
                group_array[1] = com.host.crp[idx] - r_com_C
                group_array[2] = com.host.exp[idx]
                group_array[3] = com.host.sus[idx]
                group_array[4] = rsd.host.inf[idx] - r_rsd_I
                group_array[5] = rsd.host.crp[idx] - r_rsd_C
                group_array[6] = rsd.host.exp[idx]
                group_array[7] = rsd.host.sus[idx]
                
                sampled_units = rng_sim.sample_items_from_weighted_groups(group_array, params.w_rem, remove_amount)
                
                r_com_I += sampled_units[0]
                r_com_C += sampled_units[1]
                r_com_E += sampled_units[2]
                r_com_S += sampled_units[3]
                r_rsd_I += sampled_units[4]
                r_rsd_C += sampled_units[5]
                r_rsd_E += sampled_units[6]
                r_rsd_S += sampled_units[7]

            # Update after removal
            shc.n_sus_rem += r_com_S + r_rsd_S
            
            if r_com_S + r_com_E + r_com_C + r_com_I > 0:
                update_pMd_com, update_vMd_com, rates_flags_com = com.update_removal(idx, r_com_S, r_com_E, r_com_C, r_com_I, time, rates_flags_com) # accounts for new spray value
                if mng.vectorFull == 1:
                    com.rates.lddR.submitRate_noTotal(idx, com.ctrl.inf_res[idx] * com.host.pInf[idx] * com.rates.rLdd * com.vector.dens_vec[idx])
                    rates_flags_com[<int>RatesIdx.ldd] = 1
                else:
                    com.rates.lddR.submitRate_noTotal(idx, com.rates.rLdd * com.vector.dens_vec[idx])
                    rates_flags_com[<int>RatesIdx.ldd] = 1

            if r_rsd_S + r_rsd_E + r_rsd_C + r_rsd_I > 0:
                update_pMd_rsd, update_vMd_rsd, rates_flags_rsd = rsd.update_removal(idx, r_rsd_S, r_rsd_E, r_rsd_C, r_rsd_I, time, rates_flags_rsd) # accounts for new spray value
                if mng.vectorFull == 1:
                    rsd.rates.lddR.submitRate_noTotal(idx, rsd.ctrl.inf_res[idx] * rsd.host.pInf[idx] * rsd.rates.rLdd * rsd.vector.dens_vec[idx])
                    rates_flags_rsd[<int>RatesIdx.ldd] = 1
                else:
                    rsd.rates.lddR.submitRate_noTotal(idx, rsd.rates.rLdd * rsd.vector.dens_vec[idx])
                    rates_flags_rsd[<int>RatesIdx.ldd] = 1

            if (update_pMd_com + update_pMd_rsd > 0) and (com.host.sus[idx] + rsd.host.sus[idx] == 0):
                temp_pat = update_nearby_availability_pMdd(idx, space, com, rsd)
                rates_flags_com[<int>RatesIdx.pMd] |= temp_pat
                rates_flags_rsd[<int>RatesIdx.pMd] |= temp_pat
                update_pMd_com = 0
                update_pMd_rsd = 0

            if (update_vMd_com + update_vMd_rsd > 0) and (rsd.vector.status_vec[idx] != State.absent and com.vector.status_vec[idx] != State.absent):
                temp_vec = update_nearby_availability_vMdd(idx, space, com, rsd)
                rates_flags_com[<int>RatesIdx.vMd] |= temp_vec
                rates_flags_rsd[<int>RatesIdx.vMd] |= temp_vec
                update_vMd_com = 0
                update_vMd_rsd = 0



            # if update_pMd_com + update_vMd_com + update_pMd_rsd + update_vMd_rsd > 0:
            #     temp_pat, temp_vec = update_nearby_availability_removal(idx, space, com, rsd, update_pMd_com + update_pMd_rsd, update_vMd_com + update_vMd_rsd)

            #     rates_flags_com[<int>RatesIdx.pMd] |= temp_pat
            #     rates_flags_rsd[<int>RatesIdx.pMd] |= temp_pat
            #     rates_flags_com[<int>RatesIdx.vMd] |= temp_vec
            #     rates_flags_rsd[<int>RatesIdx.vMd] |= temp_vec

            #     update_pMd_com = 0
            #     update_vMd_com = 0
            #     update_pMd_rsd = 0
            #     update_vMd_rsd = 0

            # Update susceptibility in pSd
            if (com.host.tot[idx] + rsd.host.tot[idx] - com.host.rem[idx] - rsd.host.rem[idx] > 0): # if there are units standing in the cell
                # denominator change (always when removal happens)
                susc_ratio = <np.float64_t>(com.host.tot[idx] + rsd.host.tot[idx] - com.host.rem[idx] - rsd.host.rem[idx] + (r_com_S + r_com_E + r_com_C + r_com_I + r_rsd_S + r_rsd_E + r_rsd_C + r_rsd_I)) / <np.float64_t>(com.host.tot[idx] + rsd.host.tot[idx] - com.host.rem[idx] - rsd.host.rem[idx])
                if (com.host.sus[idx] + r_com_S > 0) and (com.rates.pSdR.rates[idx] > 0):
                    # numerator change
                    com.rates.pSdR.submitRate_noTotal(idx, com.rates.pSdR.rates[idx] * susc_ratio * (<np.float64_t>(com.host.sus[idx]) / <np.float64_t>(com.host.sus[idx] + r_com_S)))
                    rates_flags_com[<int>RatesIdx.pSd] = 1
                if (rsd.host.sus[idx] + r_rsd_S > 0) and (rsd.rates.pSdR.rates[idx] > 0):
                    # numerator change
                    rsd.rates.pSdR.submitRate_noTotal(idx, rsd.rates.pSdR.rates[idx] * susc_ratio * (<np.float64_t>(rsd.host.sus[idx]) / <np.float64_t>(rsd.host.sus[idx] + r_rsd_S)))
                    rates_flags_rsd[<int>RatesIdx.pSd] = 1
            else: # if there are no units standing in the cell
                if com.rates.pSdR.rates[idx] > 0:
                    com.rates.pSdR.submitRate_noTotal(idx, 0.0)
                    rates_flags_com[<int>RatesIdx.pSd] = 1
                if rsd.rates.pSdR.rates[idx] > 0:
                    rsd.rates.pSdR.submitRate_noTotal(idx, 0.0)
                    rates_flags_rsd[<int>RatesIdx.pSd] = 1

            if mng.vectorFull == 0 and (com.summary.vectorFull + rsd.summary.vectorFull == 2):              # if after removal vector is everywhere for first time
                mng.vectorFull = 1
                # Add multipliers to ldd
                com.rates.lddR.change_flush(mng.current_flush)
                rsd.rates.lddR.change_flush(mng.current_flush)
                for ii in range(com.nCells):
                    com.rates.lddR.submitRate_noTotal(ii, com.host.pInf[ii] * com.ctrl.inf_res[ii] * com.rates.rLdd * com.vector.dens_vec[ii])
                    rsd.rates.lddR.submitRate_noTotal(ii, rsd.host.pInf[ii] * rsd.ctrl.inf_res[ii] * rsd.rates.rLdd * rsd.vector.dens_vec[ii])
                com.rates.vMdR.zeroRates()
                rsd.rates.vMdR.zeroRates()
                com.rates.vSdR.zeroRates()
                rsd.rates.vSdR.zeroRates()
                com.rates.estR.zeroRates()
                rsd.rates.estR.zeroRates()
                rates_flags_com[<int>RatesIdx.vMd] = 0   # already changed the sum, should not change again
                rates_flags_rsd[<int>RatesIdx.vMd] = 0
                rates_flags_com[<int>RatesIdx.vSd] = 0
                rates_flags_rsd[<int>RatesIdx.vSd] = 0
                rates_flags_com[<int>RatesIdx.est] = 0
                rates_flags_rsd[<int>RatesIdx.est] = 0

                rates_flags_com[<int>RatesIdx.ldd] = 1
                rates_flags_rsd[<int>RatesIdx.ldd] = 1

            # Mid- and short-distance dispersals need to be updated only if cell was/is colonised
            if old_vec_com + old_vec_rsd > 0:
                flag_com, flag_rsd = update_pSdR_noTotal(idx, space, com, rsd, old_vec_com, com.vector.dens_vec[idx], 
                            old_pInf_com, com.host.pInf[idx], old_vec_rsd, rsd.vector.dens_vec[idx], old_pInf_rsd, rsd.host.pInf[idx])
                rates_flags_com[<int>RatesIdx.pSd] |= flag_com
                rates_flags_rsd[<int>RatesIdx.pSd] |= flag_rsd

                if mng.vectorFull == 0:
                    flag_com, flag_rsd = update_vSdR_noTotal(idx, space, com, rsd, old_vec_com, com.vector.dens_vec[idx], old_vec_rsd, rsd.vector.dens_vec[idx])
                    rates_flags_com[<int>RatesIdx.vSd] |= flag_com
                    rates_flags_rsd[<int>RatesIdx.vSd] |= flag_rsd

                if old_vec_com > 0:
                    if old_vec_com * old_pInf_com != com.host.pInf[idx] * com.vector.dens_vec[idx] and (space.susc_pMdd_now[idx] > 0): # if anything changed
                        com.rates.pMdR.submitRate_noTotal(idx, space.p_md_pat * com.rates.rInf * com.ctrl.inf_res[idx] * com.host.pInf[idx] * com.vector.dens_vec[idx])
                        rates_flags_com[<int>RatesIdx.pMd] = 1
                    if old_vec_com != com.vector.dens_vec[idx] and (space.susc_vMdd_now[idx] > 0):
                        com.rates.vMdR.submitRate_noTotal(idx, space.p_md_vec * com.rates.rMdd * com.vector.dens_vec[idx])
                        rates_flags_com[<int>RatesIdx.vMd] = 1
                if old_vec_rsd > 0:
                    if old_vec_rsd * old_pInf_rsd != rsd.host.pInf[idx] * rsd.vector.dens_vec[idx] and (space.susc_pMdd_now[idx] > 0): # if anything changed
                        rsd.rates.pMdR.submitRate_noTotal(idx, space.p_md_pat * rsd.rates.rInf * rsd.ctrl.inf_res[idx] * rsd.host.pInf[idx] * rsd.vector.dens_vec[idx])
                        rates_flags_rsd[<int>RatesIdx.pMd] = 1
                    if old_vec_rsd != rsd.vector.dens_vec[idx] and (space.susc_vMdd_now[idx] > 0):
                        rsd.rates.vMdR.submitRate_noTotal(idx, space.p_md_vec * rsd.rates.rMdd * rsd.vector.dens_vec[idx])
                        rates_flags_rsd[<int>RatesIdx.vMd] = 1

cpdef void update_spray(np.uint8_t[::1] where_spray, np.uint8_t[::1] rates_flags_com, np.uint8_t[::1] rates_flags_rsd, CellsByType com, CellsByType rsd, 
                        Management mng, SpatialStructureCitrus space, SimulationParameters params):
    """ Updates cumulative spray value in [where_spray] cells and updates vector density and rates (but not total rates) accordingly. """

    cdef np.float64_t old_vec_dens, old_spray
    cdef Py_ssize_t idx
    cdef np.int8_t flag_com = 0, flag_rsd = 0
    cdef np.int32_t idx_com, idx_rsd

    for idx in range(where_spray.shape[0]):
        if where_spray[idx] == 0:
            continue

        old_spray = com.ctrl.spray[idx]
        com.ctrl.spray[idx] = com.ctrl.sprayDem[idx] * com.ctrl.sprayReg[idx] * com.ctrl.sprayBG[idx]

        # Add this to check in test that when spray doesn't change (and vector density doesn't change) certain rates should not be flagged (e.g. ldd)
        # if com.ctrl.spray[idx] == old_spray:
        #     continue

        if com.vector.status_vec[idx] == State.colonised:
            old_vec_dens = com.vector.dens_vec[idx]
            com.vector.dens_vec[idx] = com.vector.cc_vec[idx] * com.ctrl.spray[idx] * (1.0 - <np.float64_t>(com.host.rem[idx]) / <np.float64_t>(com.host.tot[idx]))

            if mng.vectorFull == 0:
                flag_com, flag_rsd = update_vSdR_noTotal(<np.int32_t>idx, space, com, rsd, old_vec_dens, com.vector.dens_vec[idx], rsd.vector.dens_vec[idx], rsd.vector.dens_vec[idx])
                rates_flags_com[<int>RatesIdx.vSd] |= flag_com
                rates_flags_rsd[<int>RatesIdx.vSd] |= flag_rsd

                com.rates.lddR.submitRate_noTotal(<np.int32_t>idx, com.rates.rLdd * com.vector.dens_vec[idx])
                rates_flags_com[<int>RatesIdx.ldd] = 1
            elif com.ctrl.inf_res[idx] * com.host.pInf[idx] > 0:
                com.rates.lddR.submitRate_noTotal(<np.int32_t>idx, com.ctrl.inf_res[idx] * com.host.pInf[idx] * com.rates.rLdd * com.vector.dens_vec[idx])
                rates_flags_com[<int>RatesIdx.ldd] = 1
            
            flag_com, flag_rsd = update_pSdR_noTotal(<np.int32_t>idx, space, com, rsd, old_vec_dens, com.vector.dens_vec[idx], com.host.pInf[idx], com.host.pInf[idx],
                        rsd.vector.dens_vec[idx], rsd.vector.dens_vec[idx], rsd.host.pInf[idx], rsd.host.pInf[idx])
            rates_flags_com[<int>RatesIdx.pSd] |= flag_com
            rates_flags_rsd[<int>RatesIdx.pSd] |= flag_rsd

            if (space.susc_pMdd_now[idx] > 0) and (com.host.pInf[idx] * com.ctrl.inf_res[idx] > 0):
                com.rates.pMdR.submitRate_noTotal(<np.int32_t>idx, space.p_md_pat * com.rates.rInf * com.host.pInf[idx] * com.ctrl.inf_res[idx] * com.vector.dens_vec[idx])
                rates_flags_com[<int>RatesIdx.pMd] = 1
            if space.susc_vMdd_now[idx] > 0:
                com.rates.vMdR.submitRate_noTotal(<np.int32_t>idx, space.p_md_vec * com.rates.rMdd * com.vector.dens_vec[idx])
                rates_flags_com[<int>RatesIdx.vMd] = 1

        elif com.vector.status_vec[idx] == State.absent:
            com.rates.vSdR.submitRate_noTotal(<np.int32_t>idx, com.rates.vSdR.rates[<np.int32_t>idx] * com.ctrl.spray[idx] / old_spray)
            rates_flags_com[<int>RatesIdx.vSd] = 1

cpdef tuple update_demarcated_area(SurveyHelperClass shc, np.float64_t time, Management mng, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, LinkedList ll, SimulationParameters params, SimulationSetup simSetup):
    """ Updates demarcated area based on new detections and changes spray values (dem and reg) accordingly.
    Note:
        a. Update of rates and vector densities following changes in spray is computed separately."""
    cdef np.int8_t spray = 0, new_dem = 0
    cdef np.int32_t idx, idx_dem, r, c, r_rel, c_rel, r_dem, c_dem
    cdef Py_ssize_t i, w, z
    cdef np.uint8_t[::1] buffer_spray = np.zeros(shc.cell_success.shape[0], dtype=np.uint8)

    for i in range(shc.cell_success.shape[0]):
        if mng.used_for_dem[i]: # avoid re-adding from cells already used (always the same)
            continue
        if shc.cell_success[i] + shc.cell_success_host[i] + shc.add_demarcated[i]:
            mng.used_for_dem[i] = 1 # flag as used
            r, c = space.coord_landscape[i,:]
            for w in range(space.dem_coord.shape[0]):
                r_rel, c_rel = space.dem_coord[w,:]
                for z in range(8):
                    r_dem = r + r_rel * space.coord_transf[z,0] + c_rel * space.coord_transf[z,1]
                    c_dem = c + r_rel * space.coord_transf[z,2] + c_rel * space.coord_transf[z,3]
                    if 0 <= r_dem < space.max_row and 0 <= c_dem < space.max_col:
                        idx_dem = space.landscape2d[r_dem, c_dem]
                        if idx_dem < 0:
                            continue
                        if mng.buffer[idx_dem] or mng.infected[idx_dem]:
                            continue
                        new_dem = 1
                        buffer_spray[idx_dem] = 1 * mng.compliance[idx_dem] # non-compliant keep regional spray regime
                        mng.demarcated[idx_dem] = 1
                        mng.buffer[idx_dem] = 1
                        mng.regional[idx_dem] = 0
                        mng.totReg -= 1
            
            mng.infected[i] = 1
            mng.buffer[i] = 0

    if new_dem:
        if mng.totReg == 0:
            # Remove regional surveyes in the queue (can only be one at a time in the list for each type)
            ll.pop_next_of_type(Event.SR00, 1)

        # Change spray value if necessary
        if mng.detected == 0:   # first time setting up demarcated area
            mng.detected = 1

            ll.insert(time + params.deltaDem[1], Event.SI0, mng._inf) # infected
            ll.insert(time + params.deltaDem[0], Event.SB00, mng._buf) # buffer

            if not simSetup.sprayContinousDem:  # add end of spray effect + next spray
                # Add this only if it's the first time in demarcated area, otherwise they are automatically set during spraying events.
                ll.insert(time + params.durationSpray, Event.SD0, mng._dem.copy())     # copy 
                ll.insert(time + params.deltaSprayDem, Event.SD1, mng._dem)
                spray = 1
                for i in range(buffer_spray.shape[0]):
                    if buffer_spray[i] == 0:
                        continue
                    com.ctrl.sprayDem[i] = 1.0 - com.ctrl.spray_prop[i] * params.sprayEffectDem
                    if simSetup.sprayContinousReg:  # remove regional spray if continuous, otherwise wait for it to end its effect normally
                        com.ctrl.sprayReg[i] = 1.0

        if simSetup.sprayContinousDem:
            spray = 1
            for i in range(buffer_spray.shape[0]):
                if buffer_spray[i] == 0:
                        continue
                com.ctrl.sprayDem[i] = 1.0 - com.ctrl.spray_prop[i] * params.sprayEffectDem
                if simSetup.sprayContinousReg:  # remove regional spray if continuous, otherwise wait for it to end its effect normally
                    com.ctrl.sprayReg[i] = 1.0
        elif simSetup.sprayContinousReg:
            spray = 1
            for i in range(buffer_spray.shape[0]):
                if buffer_spray[i] == 0:
                        continue
                com.ctrl.sprayReg[i] = 1.0

    return buffer_spray, spray

cpdef np.uint8_t randomized_search_survey(np.ndarray cells, np.uint8_t nVis, np.float64_t propPCR, np.float64_t pComs, np.float64_t pRsds, CellsByType com, CellsByType rsd, Management mng, SurveyHelperClass shc, SimulationParameters params, SimulationRNG rng_sim):
    """ Computes randomized host surveys across [cells] (e.g., demarcated area, regional area, search area). 
    Args:
        cells: array (not memoryview) of cells to be sampled for the survey
        com, rsd: classes for cells of citrus type com and rsd
        nVis, nPCR: number of visual inspections and PCR tests to be done in each surveyed cell
        pComs, pRsds: proportion of compliant commercial and residential cells to be surveyed
        mng: management class
        shc: survey helper class
        params: simulation parameters
        rng_sim: for random sampling
    Returns:
        found_host: 1 if detection is successful, 0 otherwise

    Notes:
        a. Surveys are affected by compliance"""

    cdef np.int32_t nCells = com.nCells
    cdef np.int32_t n_eligible, n_mandatory, n_optional, n_surv, nSurvCom = 0, nSurvRsd = 0
    cdef np.ndarray weights, weights_mask, mandatory_mask, optional, mandatory, eligible, selected_idx
    cdef np.int32_t[::1] idxSurvCom, idxSurvRsd
    cdef np.int8_t found_host = 0
    cdef np.float64_t tot_weight

    if cells.shape[0] != nCells:
        raise ValueError("Binary array of cells to survey must have the same length as the number of cells in com and rsd.")

    idxCells = np.arange(com.nCells).astype(np.int32)

    mandatory_mask = shc._cell_success | shc._cell_success_host

    if com.summary.totStanding > 0 and pComs > 0:
        if pComs == 1:
            idxSurvCom = (np.nonzero((cells + mandatory_mask) * com.summary._standing * mng._comp)[0]).astype(np.int32)
        else:
            eligible = cells * com.summary._standing # includes mandatory
            optional = eligible * (1 - mandatory_mask) # does not include mandatory
            mandatory = com.summary._standing * mandatory_mask
            n_mandatory = mandatory.sum()
            n_optional = optional.sum()
            n_eligible = eligible.sum()
            n_surv = <np.int32_t>c_math.ceil(pComs * n_eligible) # total sampled (including mandatory)
            if n_mandatory >= n_surv:
                idxSurvCom = (np.nonzero(mandatory * mng._comp)[0]).astype(np.int32) 
            elif n_optional:
                weights = optional * com.summary._n_stand * (1.0-params.pNoSearch*(shc._cells_surveyed)) # already searched cells 90% less likely to be picked
                weights_mask = (weights > 0)
                if (weights_mask).sum() >= n_surv - n_mandatory:
                    tot_weight = weights.sum()
                    selected_idx = np.random.choice(idxCells, size=n_surv - n_mandatory, replace=False, p = weights / <np.float64_t>(tot_weight))
                    idxSurvCom = np.concatenate([selected_idx[mng._comp[selected_idx] == 1], (np.nonzero(mandatory * mng._comp)[0]).astype(np.int32)])
                else:
                    idxSurvCom = (np.nonzero((weights_mask + mandatory) * mng._comp)[0]).astype(np.int32) 
            else:
                idxSurvCom = (np.nonzero(mandatory * mng._comp)[0]).astype(np.int32) 
 
        nSurvCom = <np.int32_t>idxSurvCom.shape[0]

    if rsd.summary.totStanding > 0 and pRsds > 0:
        if pRsds == 1:
            idxSurvRsd = (np.nonzero((cells + mandatory_mask) * rsd.summary._standing * mng._comp)[0]).astype(np.int32)
        else:
            eligible = cells * rsd.summary._standing # includes mandatory
            optional = eligible * (1 - mandatory_mask) # does not include mandatory
            mandatory = rsd.summary._standing * mandatory_mask
            n_mandatory = mandatory.sum()
            n_optional = optional.sum()
            n_eligible = eligible.sum()
            n_surv = <np.int32_t>c_math.ceil(pRsds * n_eligible) # total sampled (including mandatory)
            if n_mandatory >= n_surv:
                idxSurvRsd = (np.nonzero(mandatory * mng._comp)[0]).astype(np.int32) 
            elif n_optional:
                weights = optional * rsd.summary._n_stand * (1.0-params.pNoSearch*(shc._cells_surveyed)) # already searched cells 90% less likely to be picked
                weights_mask = (weights > 0)
                if (weights_mask).sum() >= n_surv - n_mandatory:
                    tot_weight = weights.sum()
                    selected_idx = np.random.choice(idxCells, size=n_surv - n_mandatory, replace=False, p = weights / <np.float64_t>(tot_weight))
                    idxSurvRsd = np.concatenate([selected_idx[mng._comp[selected_idx] == 1], (np.nonzero(mandatory * mng._comp)[0]).astype(np.int32)])
                else:
                    idxSurvRsd = (np.nonzero((weights_mask + mandatory) * mng._comp)[0]).astype(np.int32) 
            else:
                idxSurvRsd = (np.nonzero(mandatory * mng._comp)[0]).astype(np.int32) 
 
        nSurvRsd = <np.int32_t>idxSurvRsd.shape[0]

    if nSurvCom + nSurvRsd == 0:
        return found_host

    if nSurvCom > 0:
        found_host |= host_detection(nVis, propPCR, idxSurvCom, shc.comDetected, com, params.pVis, params.pPCR, shc, rng_sim)

    if nSurvRsd > 0:
        found_host |= host_detection(nVis, propPCR, idxSurvRsd, shc.rsdDetected, rsd, params.pVis, params.pPCR, shc, rng_sim)

    return found_host

cpdef void update_spray_with_rates(np.uint8_t[::1] where_spray, CellsByType com, CellsByType rsd, Management mng, SpatialStructureCitrus space, SimulationParameters params):
    """ Updates cumulative spray value in [where_spray] cells and updates vector density and rates (and total rates) accordingly. """

    cdef np.float64_t old_vec_dens, old_spray
    cdef Py_ssize_t idx
    cdef np.int8_t flag_com = 0, flag_rsd = 0
    cdef np.int32_t idx_com, idx_rsd

    for idx in range(where_spray.shape[0]):
        if where_spray[idx] == 0:
            continue

        old_spray = com.ctrl.spray[idx]
        com.ctrl.spray[idx] = com.ctrl.sprayDem[idx] * com.ctrl.sprayReg[idx] * com.ctrl.sprayBG[idx]

        if com.vector.status_vec[idx] == State.colonised:
            old_vec_dens = com.vector.dens_vec[idx]
            com.vector.dens_vec[idx] = com.vector.cc_vec[idx] * com.ctrl.spray[idx] * (1.0 - <np.float64_t>(com.host.rem[idx]) / <np.float64_t>(com.host.tot[idx]))

            if mng.vectorFull == 0:
                update_vSdR_noTotal(<np.int32_t>idx, space, com, rsd, old_vec_dens, com.vector.dens_vec[idx], rsd.vector.dens_vec[idx], rsd.vector.dens_vec[idx])

                com.rates.lddR.submitRate_noTotal(<np.int32_t>idx, com.rates.rLdd * com.vector.dens_vec[idx])
            else:
                com.rates.lddR.submitRate_noTotal(<np.int32_t>idx, com.ctrl.inf_res[idx] * com.host.pInf[idx] * com.rates.rLdd * com.vector.dens_vec[idx])

            update_pSdR_noTotal(<np.int32_t>idx, space, com, rsd, old_vec_dens, com.vector.dens_vec[idx], com.host.pInf[idx], com.host.pInf[idx],
                        rsd.vector.dens_vec[idx], rsd.vector.dens_vec[idx], rsd.host.pInf[idx], rsd.host.pInf[idx])

            if space.susc_pMdd_now[idx] > 0:
                com.rates.pMdR.submitRate_noTotal(<np.int32_t>idx, space.p_md_pat * com.rates.rInf * com.host.pInf[idx] * com.ctrl.inf_res[idx] * com.vector.dens_vec[idx])
            if space.susc_vMdd_now[idx] > 0:
                com.rates.vMdR.submitRate_noTotal(<np.int32_t>idx, space.p_md_vec * com.rates.rMdd * com.vector.dens_vec[idx])

        elif com.vector.status_vec[idx] == State.absent:
            com.rates.vSdR.submitRate_noTotal(<np.int32_t>idx, com.rates.vSdR.rates[idx] * com.ctrl.spray[idx] / old_spray)

    com.rates.pSdR.recompute_totals()
    rsd.rates.pSdR.recompute_totals()
    com.rates.lddR.recompute_totals() # only com because it's particle emission and no rsd spray
    com.rates.pMdR.recompute_totals() # only com because it's particle emission and no rsd spray
    if mng.vectorFull == 0:
        com.rates.vSdR.recompute_totals()
        rsd.rates.vSdR.recompute_totals()
        com.rates.vMdR.recompute_totals() # only com because it's particle emission and no rsd spray

cpdef tuple randomized_survey(np.float64_t time, np.uint8_t survey_type, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, SimulationParameters params, SimulationRNG rng_sim):
    """ Computes randomized surveys across cells (e.g., demarcated area, regional area). 
    Args:
        event_type: type of survey (demarcated for demarcated or search area, or regional)
        cells: array (not memoryview) of cells to be sampled for the survey
        com, rsd: classes for cells of citrus type com and rsd
        params: simulation parameters
        bg: BitGenerator for random sampling
    Returns:
        found: 1 if detection is successful, 0 otherwise
        cell_success: cells, regardless of type, where detection was successful (binary, spans across all cells, not just surveyed)
    Notes:
        a. Surveys are not affected by compliance"""
    cdef np.uint32_t tot_infected = 0, tot_tot = 0, cells_infected = 0, cell_infections = 0
    cdef np.int32_t nCells = com.nCells
    cdef np.int32_t i, j
    cdef np.int32_t nSurvCom = 0, nSurvRsd = 0
    cdef np.int32_t[::1] idxSurvCom, idxSurvRsd
    cdef np.int8_t found_vec = 0, found_host = 0
    cdef np.float64_t p_det_vec
    cdef np.float64_t[::1] rand_val

    cdef np.uint8_t nVec = params.nVec[mng.detected, survey_type]

    if params.survey_by_prop[survey_type]:
        idxSurvCom, idxSurvRsd = compute_n_cells_to_survey_prop(survey_type, com, rsd, mng, shc, params)
    else:
        idxSurvCom, idxSurvRsd = compute_n_cells_to_survey(survey_type, com, rsd, mng, shc, params)

    nSurvCom = <np.int32_t>idxSurvCom.shape[0]
    nSurvRsd = <np.int32_t>idxSurvRsd.shape[0]

    if nSurvCom + nSurvRsd == 0:
        return found_vec, found_host

    rand_val = np.empty((nSurvCom + nSurvRsd) * nVec, dtype=np.float64)
    rng_sim.fill_uniform(rand_val, rand_val.shape[0])

    if nSurvCom > 0:
        # Vector
        if nVec:

            for i in range(nSurvCom):
                idx = idxSurvCom[i]
                shc.n_vectors += nVec

                if com.host.pInf[idx] > 0:
                    p_det_vec = params.pVec * com.host.pInf[idx]
                else: # no infected vector
                    continue
                for j in range(nVec):
                    if p_det_vec > rand_val[i * nVec + j]:
                        shc.cell_success[idx] = 1
                        found_vec = 1
                        break

        # Host
        found_host |= host_detection(params.nHostVis[mng.detected, survey_type], params.pHostPCR[mng.detected, survey_type], idxSurvCom, shc.comDetected, com, params.pVis, params.pPCR, shc, rng_sim)
    
    if nSurvRsd > 0:
        # Vector
        if nVec:
            for i in range(nSurvRsd):
                idx = idxSurvRsd[i]
                shc.n_vectors += nVec

                if rsd.host.pInf[idx] > 0:
                    p_det_vec = params.pVec * rsd.host.pInf[idx]
                else: # no infected vector
                    continue
                
                for j in range(nVec):
                    if p_det_vec > rand_val[nSurvCom * nVec + i * nVec + j]:
                        shc.cell_success[idx] = 1
                        found_vec = 1
                        break

        # Host
        found_host |= host_detection(params.nHostVis[mng.detected, survey_type], params.pHostPCR[mng.detected, survey_type], idxSurvRsd, shc.rsdDetected, rsd, params.pVis, params.pPCR, shc, rng_sim)
    
    if (mng.detected == 0) and (found_vec or found_host): # update if first detection
        # mng.detected changed later in demarcated area
        cells_infected, tot_infected, tot_tot = compute_incidences(com, rsd)

        mng.time_det = time
        mng.cell_inc_det = cells_infected / <np.float64_t>(nCells)
        mng.cit_inc_det = tot_infected / <np.float64_t>(tot_tot)
        mng.cell1km_inc_det = space._infCells1km.sum() / <np.float64_t>(space.unique_1km)

        mng.n_pcr_det = shc.n_pcr
        mng.n_vec_det = shc.n_vectors
        mng.n_vis_det = shc.n_vis

        if found_vec + found_host > 1:
            mng.which_det = 3
        elif found_vec:
            mng.which_det = 1
        else:
            mng.which_det = 2

    # Compute search area around detected cells in buffer / regional
    if (survey_type != <np.uint8_t>SurveyType.infArea) and (found_vec or found_host):

        success_idx = (np.nonzero(shc._cell_success + shc._cell_success_host)[0]).astype(np.int32)
        for idx in success_idx:
            build_search_area(idx, space, shc)

        found_host |= randomized_search_survey(shc._cells_search, params.nHostVis[1, <np.uint8_t>SurveyType.searchArea], params.pHostPCR[1, <np.uint8_t>SurveyType.searchArea], params.pComSurv[1, <np.uint8_t>SurveyType.searchArea], params.pRsdSurv[1, <np.uint8_t>SurveyType.searchArea], com, rsd, mng, shc, params, rng_sim)

    return found_vec, found_host

cpdef np.int8_t do_survey(EventNode event, np.float64_t time, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, LinkedList ll, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim):

    cdef np.int32_t nCells = com.nCells
    cdef np.int8_t found_vec = 0, found_host = 0, spray = 0
    cdef np.int32_t tot_search = 0
    cdef np.uint8_t[::1] add_demarcated, buffer_spray
    cdef np.uint8_t[::1] rates_flags_com = np.zeros(8, dtype=np.uint8)
    cdef np.uint8_t[::1] rates_flags_rsd = np.zeros(8, dtype=np.uint8)
    cdef np.int32_t r, c, r_rel, c_rel, r_search, c_search, idx_search
    cdef Py_ssize_t i, w, z

    shc.reset() # reset survey helper class for new survey

    if event.event_type == Event.SI0: # infected area survey

        shc._cells_surveyed[:] = mng._inf

        found_vec, found_host = randomized_survey(time, <np.uint8_t>SurveyType.infArea, com, rsd, space, mng, shc, params, rng_sim)
    
    elif event.event_type == Event.SB00: # buffer survey
        shc._cells_surveyed[:] = mng._buf

        found_vec, found_host = randomized_survey(time, <np.uint8_t>SurveyType.buffer, com, rsd, space, mng, shc, params, rng_sim)

    elif event.event_type == Event.SR00: # regional survey
        shc._cells_surveyed[:] = mng._reg

        found_vec, found_host = randomized_survey(time, <np.uint8_t>SurveyType.regional, com, rsd, space, mng, shc, params, rng_sim)
        
    else:
        raise ValueError(f"Invalid survey: {event.event_type}")

    if found_vec:
        event.event_type += 2   # successful vector survey

        if found_host:
            event.event_type += 1   # successful vector and host survey

    elif found_host:
        event.event_type += 1   # successful host survey only

    else: # no detection
        if mng.detected and time - mng.time_last_det >= params.time_for_eradication:
            if (ll.peek_time() != time) or (ll.peek_event_type() not in [Event.SB00, Event.SR00, Event.SI0]): # check no other surveys planned at same time
                mng.believed_eradicated = 1
                cells_infected, tot_infected, tot_tot = compute_incidences(com, rsd)

                mng.time_erad = time
                mng.cell_inc_erad = cells_infected / <np.float64_t>(com.nCells)
                mng.cit_inc_erad = tot_infected / <np.float64_t>(tot_tot)
                mng.cell1km_inc_erad = space._infCells1km.sum() / <np.float64_t>(space.unique_1km)
        
    if found_host and simSetup.removal:
        remove_detected(time, shc, rates_flags_com, rates_flags_rsd, com, rsd, space, mng, params, rng_sim)

    if found_vec + found_host:
        mng.time_last_det = time
        # Change demarcated area around detected and/or removed units
        buffer_spray, spray = update_demarcated_area(shc, time, mng, com, rsd,space, ll, params, simSetup)
        if spray:
            update_spray(buffer_spray, rates_flags_com, rates_flags_rsd, com, rsd, mng, space, params)

        # Update rates
        if rates_flags_com[<int>RatesIdx.pSd]:
            com.rates.pSdR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.vSd]:
            com.rates.vSdR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.pMd]:
            com.rates.pMdR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.vMd]:
            com.rates.vMdR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.ldd]:
            com.rates.lddR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.lat]:
            com.rates.latR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.sym]:
            com.rates.symR.recompute_totals()
        if rates_flags_com[<int>RatesIdx.est]:
            com.rates.estR.recompute_totals()

        if rates_flags_rsd[<int>RatesIdx.pSd]:
            rsd.rates.pSdR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.vSd]:
            rsd.rates.vSdR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.pMd]:
            rsd.rates.pMdR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.vMd]:
            rsd.rates.vMdR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.ldd]:
            rsd.rates.lddR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.lat]:
            rsd.rates.latR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.sym]:
            rsd.rates.symR.recompute_totals()
        if rates_flags_rsd[<int>RatesIdx.est]:
            rsd.rates.estR.recompute_totals()

        return event.event_type

    return event.event_type

cdef inline void build_search_area(np.int32_t idx, SpatialStructureCitrus space, SurveyHelperClass shc):
    cdef np.int32_t r, c, r_rel, c_rel, r_search, c_search, idx_search
    cdef Py_ssize_t w, z

    r, c = space.coord_landscape[idx,:]
    for w in range(space.search_coord.shape[0]):
        r_rel, c_rel = space.search_coord[w,:]
        for z in range(8):
            r_search = r + r_rel * space.coord_transf[z,0] + c_rel * space.coord_transf[z,1]
            c_search = c + r_rel * space.coord_transf[z,2] + c_rel * space.coord_transf[z,3]
            if 0 <= r_search < space.max_row and 0 <= c_search < space.max_col:
                idx_search = space.landscape2d[r_search, c_search]
                if idx_search < 0:
                    continue
                shc.cells_search[idx_search] = 1

cpdef tuple compute_n_cells_to_survey(np.uint8_t survey_type, CellsByType com, CellsByType rsd, Management mng, SurveyHelperClass shc, SimulationParameters params):

    """ Design survey: find number of cells to survey to satisfy desired design prevalence and confidence level.
    Note: does not include PCR, so it could overestimate the number of cells if PCR is used.
    """

    cdef np.int32_t n_eligible
    cdef np.ndarray weights, eligible, selected_idx
    cdef np.int32_t[::1] idxSurvCom, idxSurvRsd
    cdef np.ndarray idxCells = np.arange(com.nCells).astype(np.int32)
    cdef np.float64_t ms_units_I, ms_pooled, ms_I, ms_C, avg_units, dp_crp, dp_inf, total_ms, relative_ms
    cdef np.uint32_t n_cells_fin

    cdef np.uint8_t nVec = params.nVec[mng.detected, survey_type]
    cdef np.uint8_t nVis = params.nHostVis[mng.detected, survey_type]
    cdef np.float64_t dp_com = params.dp[0, survey_type], dp_rsd = params.dp[1, survey_type]

    ## Commercial
    if com.summary.totInfested > 0 and dp_com > 0:

        eligible = shc._cells_surveyed * com.summary._infested
        n_eligible = eligible.sum()
        standing_units = com.summary._n_stand[eligible > 0]

        if standing_units.size:
            # Since cells for survey are sampled weighted by amount of citrus in it, we compute the weighted average number of units in cells
            weights = standing_units / standing_units.sum()
            avg_units = c_math.ceil((standing_units * weights).sum())
            # avg_units must be > 0 because in order for cells to be infested (eligible) there must be some citrus

            # Method sensitivities (assuming one unit per infection status)
            ms_units_I = 1.0 - (1.0-nVis*params.pVis/(avg_units - 0.5*(params.pVis - 1.0)))
            ms_pooled = 1.0 - (1.0-params.pVec/avg_units)**nVec # same for C or I

            ms_I = 1.0 - (1.0-ms_units_I)*(1.0-ms_pooled)
            ms_C = ms_pooled # no PCR
            # exposed units have ms = 0

            dp_crp = params.prop_status[0, survey_type] * dp_com
            dp_inf = params.prop_status[1, survey_type] * dp_com

            total_ms = 1.0 - (1.0 - ms_I*dp_inf)*(1.0 - ms_C*dp_crp)
            relative_ms = total_ms / dp_com

            # Using finite population formula (approx.)
            n_cells_fin = <np.uint32_t>c_math.ceil(
                ((1.0-c_math.pow(1.0-params.cl[survey_type], 1.0/(n_eligible*dp_com)))*(n_eligible-0.5*(n_eligible*dp_com*relative_ms-1.0)))/relative_ms
                ) #n_eligible > 0 checked with standing_units.size

            if n_cells_fin > n_eligible:
                idxSurvCom = (np.nonzero(shc._cells_surveyed * com.summary._infested * mng._comp)[0]).astype(np.int32)
            elif n_cells_fin > 0:
                selected_idx = np.random.choice(idxCells[eligible > 0], size=n_cells_fin, replace=False, p = weights)
                idxSurvCom = selected_idx[mng._comp[selected_idx] == 1]
            else:
                idxSurvCom = np.array([], dtype=np.int32)
        else:
            idxSurvCom = np.array([], dtype=np.int32)
    else:
        idxSurvCom = np.array([], dtype=np.int32)

    ## Residential
    if rsd.summary.totInfested > 0 and dp_rsd > 0:

        eligible = shc._cells_surveyed * rsd.summary._infested
        n_eligible = eligible.sum()
        standing_units = rsd.summary._n_stand[eligible > 0]
        if standing_units.size:
            weights = standing_units / standing_units.sum()
            avg_units = c_math.ceil((standing_units * weights).sum())

            # Method sensitivities (assuming one unit per infection status)
            ms_units_I = 1.0 - (1.0-nVis*params.pVis/(avg_units - 0.5*(params.pVis - 1.0)))
            ms_pooled = 1.0 - (1.0-params.pVec/avg_units)**nVec # same for C or I

            ms_I = 1.0 - (1.0-ms_units_I)*(1.0-ms_pooled)
            ms_C = ms_pooled
            # exposed units have ms = 0

            dp_crp = params.prop_status[0, survey_type] * dp_com
            dp_inf = params.prop_status[1, survey_type] * dp_com

            total_ms = 1.0 - (1.0 - ms_I*dp_inf)*(1.0 - ms_C*dp_crp)
            relative_ms = total_ms / dp_com

            # Using finite population formula (approx.)
            n_cells_fin = <np.uint32_t>c_math.ceil(
                ((1.0-c_math.pow(1.0-params.cl[survey_type], 1.0/(n_eligible*dp_com)))*(n_eligible-0.5*(n_eligible*dp_com*relative_ms-1.0)))/relative_ms
                )
            n_cells_fin = min(n_cells_fin, n_eligible)
            n_cells_fin = <np.uint32_t>c_math.ceil(params.reduce_rsd * n_cells_fin)

            if n_cells_fin > 0:
                selected_idx = np.random.choice(idxCells[eligible > 0], size=n_cells_fin, replace=False, p = weights)
                idxSurvRsd = selected_idx[mng._comp[selected_idx] == 1]
            else:
                idxSurvRsd = np.array([], dtype=np.int32)
        else:
            idxSurvRsd = np.array([], dtype=np.int32)
    else:
        idxSurvRsd = np.array([], dtype=np.int32)

    return idxSurvCom, idxSurvRsd

cpdef tuple compute_n_cells_to_survey_prop(np.uint8_t survey_type, CellsByType com, CellsByType rsd, Management mng, SurveyHelperClass shc, SimulationParameters params):

    cdef np.int32_t nCells = com.nCells
    cdef np.int32_t n_eligible, n_surv
    cdef np.ndarray weights, eligible, selected_idx, idxCells
    cdef np.int32_t[::1] idxSurvCom, idxSurvRsd

    cdef np.float64_t pComs = params.pComSurv[mng.detected, survey_type], pRsds = params.pRsdSurv[mng.detected, survey_type]

    idxCells = np.arange(nCells).astype(np.int32)

    # Only compliant cells are surveyed
    if com.summary.totInfested > 0 and pComs > 0:
        if pComs == 1:
            idxSurvCom = (np.nonzero(shc._cells_surveyed * com.summary._infested * mng._comp)[0]).astype(np.int32)
        else:
            eligible = shc._cells_surveyed * com.summary._infested
            n_eligible = eligible.sum()
            n_surv = <np.int32_t>c_math.ceil(pComs * n_eligible)
            if n_surv < n_eligible:
                weights = com.summary._n_stand * eligible
                selected_idx = np.random.choice(idxCells, size=n_surv, replace=False, p = weights / <np.float64_t>(weights.sum()))
                idxSurvCom = selected_idx[mng._comp[selected_idx] == 1]
            else:
                idxSurvCom = (np.nonzero(eligible * mng._comp)[0]).astype(np.int32)
    else:
        idxSurvCom = np.array([], dtype=np.int32)

    if rsd.summary.totInfested > 0 and pRsds > 0:
        if pRsds == 1:
            idxSurvRsd = (np.nonzero(shc._cells_surveyed * rsd.summary._infested * mng._comp)[0]).astype(np.int32)
        else:
            eligible = shc._cells_surveyed * rsd.summary._infested
            n_eligible = eligible.sum()
            n_surv = <np.int32_t>c_math.ceil(pRsds * n_eligible)
            if n_surv < n_eligible:
                weights = rsd.summary._n_stand * eligible
                selected_idx = np.random.choice(idxCells, size=n_surv, replace=False, p = weights / <np.float64_t>(weights.sum()))
                idxSurvRsd = selected_idx[mng._comp[selected_idx] == 1]
            else:
                idxSurvRsd = (np.nonzero(eligible * mng._comp)[0]).astype(np.int32)
    else:
        idxSurvRsd = np.array([], dtype=np.int32)

    return idxSurvCom, idxSurvRsd

cpdef make_detection_happen(np.float64_t time, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, LinkedList ll, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim):

    cdef np.int32_t idx_det
    cdef np.ndarray com_inf, com_crp, com_inf_idx, com_crp_idx, weights
    cdef np.int8_t spray = 0
    cdef np.uint8_t[::1] buffer_spray
    cdef np.uint8_t[::1] rates_flags_com = np.zeros(8, dtype=np.uint8)
    cdef np.uint8_t[::1] rates_flags_rsd = np.zeros(8, dtype=np.uint8)
    cdef np.int32_t[::1] idxCells = np.arange(com.nCells, dtype=np.int32)
    

    shc.reset()
    # Randomly select a cryptic / symptomatic unit to survey (priority is symptomatic)
    com_inf = np.array(com.host.inf) > 0
    com_crp = np.array(com.host.crp) > 0

    # Randomly select cell for detection
    if com_inf.any():
        weights = com.summary._n_stand * com_inf
        weights = weights / <np.float64_t>(weights.sum())

        idx_det = np.random.choice(idxCells, size=1, replace=False, p = weights)[0]
        shc.comDetected[0, idx_det] = 1
    elif com_crp.any():
        weights = com.summary._n_stand * com_crp
        weights = weights / <np.float64_t>(weights.sum())

        idx_det = np.random.choice(idxCells, size=1, replace=False, p = weights)[0]
        shc.comDetected[1, idx_det] = 1

    # Impossible for no commercial crp / inf to be available as the epidemic starts in commercial and to spread even just one cell it needs to become infectious (i.e., crp or inf)
    
    shc.cell_success_host[idx_det] = 1

    # Update detection and do search
    cells_infected, tot_infected, tot_tot = compute_incidences(com, rsd)

    if space._infCells.sum() != cells_infected:
        raise ValueError("!!!!! Wrong number of cells infected???")

    mng.time_det = time
    mng.cell_inc_det = cells_infected / <np.float64_t>(com.nCells)
    mng.cit_inc_det = tot_infected / <np.float64_t>(tot_tot)
    mng.cell1km_inc_det = space._infCells1km.sum() / <np.float64_t>(space.unique_1km)

    mng.which_det = 2 # host
    mng.time_last_det = time

    # Compute search area around detected cells in buffer / regional
    build_search_area(idx_det, space, shc)
    randomized_search_survey(shc._cells_search, params.nHostVis[1, <np.uint8_t>SurveyType.searchArea], params.pHostPCR[1, <np.uint8_t>SurveyType.searchArea], params.pComSurv[1, <np.uint8_t>SurveyType.searchArea], params.pRsdSurv[1, <np.uint8_t>SurveyType.searchArea], com, rsd, mng, shc, params, rng_sim)

    # Remove and update demarcated area
    remove_detected(time, shc, rates_flags_com, rates_flags_rsd, com, rsd, space, mng, params, rng_sim)

    buffer_spray, spray = update_demarcated_area(shc, time, mng, com, rsd,space, ll, params, simSetup)
    if spray:
        update_spray(buffer_spray, rates_flags_com, rates_flags_rsd, com, rsd, mng, space, params)

    # Update rates
    if rates_flags_com[<int>RatesIdx.pSd]:
        com.rates.pSdR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.vSd]:
        com.rates.vSdR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.pMd]:
        com.rates.pMdR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.vMd]:
        com.rates.vMdR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.ldd]:
        com.rates.lddR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.lat]:
        com.rates.latR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.sym]:
        com.rates.symR.recompute_totals()
    if rates_flags_com[<int>RatesIdx.est]:
        com.rates.estR.recompute_totals()

    if rates_flags_rsd[<int>RatesIdx.pSd]:
        rsd.rates.pSdR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.vSd]:
        rsd.rates.vSdR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.pMd]:
        rsd.rates.pMdR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.vMd]:
        rsd.rates.vMdR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.ldd]:
        rsd.rates.lddR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.lat]:
        rsd.rates.latR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.sym]:
        rsd.rates.symR.recompute_totals()
    if rates_flags_rsd[<int>RatesIdx.est]:
        rsd.rates.estR.recompute_totals()

cpdef tuple compute_incidences(CellsByType com, CellsByType rsd):
    cdef np.int32_t i
    cdef np.uint32_t tot_infected = 0, tot_tot = 0, cells_infected = 0, cell_infections = 0

    for i in range(com.nCells):
        cell_infections = com.host.exp[i] + com.host.crp[i] + com.host.inf[i] + rsd.host.exp[i] + rsd.host.crp[i] + rsd.host.inf[i] + com.host.rem[i] + rsd.host.rem[i]
        tot_infected += cell_infections
        tot_tot += com.host.tot[i] + rsd.host.tot[i]
        cells_infected += <int>(cell_infections > 0)

    return cells_infected, tot_infected, tot_tot




