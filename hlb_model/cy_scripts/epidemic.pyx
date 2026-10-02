# epidemic.pyx

import time
import numpy as np
cimport numpy as np
cimport cython

from numpy.random cimport BitGenerator

cimport libc.math as c_math
#from libc.math cimport floor, ceil, round, log, M_PI, cos, sin
from .extending_distributions cimport SimulationRNG
from .structs cimport SimulationSetup, SetupParameters, LandscapePostprocessing, SimulationParameters, SpatialStructureCitrus, Management, CellsByType, SurveyHelperClass,SaveStructure, SaveMetrics
from .structs cimport State, Event, SimGoal
from .dispersal_updates cimport update_pSdR, update_vSdR, pathogen_arrival, vector_arrival, pat_vec_arrival, find_destination_md
from .management cimport update_spray_with_rates, do_survey, make_detection_happen
from .linkedList cimport LinkedList

# Type definitions
np.import_array()

cpdef tuple spiral_initialization(np.int32_t idx, CellsByType com, SpatialStructureCitrus space, LandscapePostprocessing landscape, SetupParameters setupPrms, SimulationSetup simSetup):
    """ Defines which cells are affected by initial infection and in what amount (i.e., how many units in each cell are exposed). Starting from one cell pick at random based on citrus density (idx),the algorithm fills it with exposed units and moves around it in a spiral until all required exposured are assigned. 
        Spiral visualisation:
        g → h → i → ...
        ↑
        f   a → b
        ↑       ↓
        e ← d ← c
        Note: asymmetrics spiral, max_step down / right, (max_step - 1) up / left. Covers at max an N x N area (N = 2*max_step), N even.
    """
    # Results & Setup
    cdef np.uint8_t max_step = 5 # max half / width of the spiral, asymmetric (max_step down /right, but only max_step - 1 up / left). With 5, it results in a 10x10 spiral, i.e., 1 km (0.1 res)
    cdef np.uint8_t top = max_step-1, bottom = max_step, left = max_step-1, right = max_step
    cdef np.int32_t[::1] locations = -np.ones((max_step*2)**2, dtype = np.int32)
    cdef np.uint8_t[::1] amounts = np.zeros((max_step*2)**2, dtype = np.uint8)

    # Helper variables
    cdef np.int8_t[:, ::1] directions = np.array([[0,1], [1,0], [0,-1], [-1,0]], dtype=np.int8) # right, down, left, up
    cdef np.uint8_t chain_size = 1 # starting number of steps until direction change [1, 2*max_step]
    cdef np.int8_t current_direction = 0 # 0 to 4
    cdef np.int16_t current_pos = 0
    cdef np.int16_t add, left_to_add, total = 0
    cdef np.int8_t n_iter = 2 # number of iterations for each chain
    cdef np.int32_t r, c, dr, dc, nr, nc, r0, c0, nidx

    # Ring 0
    if idx < 0 or com.host.tot[idx] == 0:
        raise ValueError("Initial cell is without citrus.")
    r0, c0 = space.coord_landscape[idx,:]
    add = min(setupPrms.nExposed0, com.host.tot[idx])
    total += add
    left_to_add = setupPrms.nExposed0 - total

    locations[current_pos] = idx
    amounts[current_pos] = add
    current_pos += 1
    if left_to_add == 0:
        return 1, locations[:current_pos], amounts[:current_pos]

    r = r0
    c = c0
    while chain_size <= (max_step*2):
        n_iter = 2 if chain_size < (max_step*2) else 1
        for _ in range(n_iter):
            dr, dc = directions[current_direction % 4,:]
            for _ in range(chain_size):
                nr = r + dr
                nc = c + dc

                if nc > c0 + right or nc < c0 - left:
                    continue # outside max area (hor)
                if nr > r0 + bottom or nr < r0 - top:
                    continue # outside max area (ver)

                r = nr
                c = nc

                if (0 <= r < space.max_row) and (0 <= c < space.max_col):
                    nidx = space.landscape2d[r, c]
                    if nidx >= 0 and com.host.tot[nidx] > 0: # cells with commercial citrus
                        add = min(left_to_add, com.host.tot[nidx])
                        total += add
                        left_to_add = setupPrms.nExposed0 - total
                        locations[current_pos] = nidx
                        amounts[current_pos] = add
                        current_pos += 1
                        if left_to_add == 0:
                            return 1, locations[:current_pos], amounts[:current_pos]
            current_direction += 1
        chain_size += 1
    
    return 0, locations[:current_pos], amounts[:current_pos]

cpdef void initialize_run(CellsByType com, CellsByType rsd, SpatialStructureCitrus space,
                            Management mng, LinkedList ll, LandscapePostprocessing landscape, SimulationParameters params, 
                            SetupParameters setupPrms, SimulationSetup simSetup, SimulationRNG rng_sim):
    """ Initialize classes at the beginning of each individual run.
    Initial conditions are defined in simSetup.
    Initial time (i.e., time the simulation starts) is picked randomly.
    Time of first regional surveyes is picked randomly.
    It is assumed that there is no demarcated area at the beginning and that the first HLB+ hosts are exposed (no detectable, cryptic or symptomatic)."""

    # General
    cdef np.int32_t nCells = com.nCells
    cdef Py_ssize_t i, k
    cdef np.int8_t temp_flag = 0, flag_com = 0, flag_rsd = 0
    cdef np.int32_t j
    cdef np.int16_t newE
    cdef np.int32_t idx, idx1, idx2
    cdef np.float64_t time0, flush_time
    cdef np.uint8_t[::1] rates_flags_com = np.zeros(8, dtype = np.uint8)
    cdef np.uint8_t[::1] rates_flags_rsd = np.zeros(8, dtype = np.uint8)
    cdef object rng = np.random.Generator(rng_sim.bg)

    # Spiral initialization
    cdef np.int32_t[::1] locations
    cdef np.uint8_t[::1] amounts
    cdef np.uint8_t flag_init = 0, flag_idx = 0

    # Pathogen and vector
    cdef np.int32_t[::1] loc_infec, loc_infest

    # Control & Management
    cdef np.int32_t[::1] loc_bctrl, loc_res, loc_nc
    cdef Py_ssize_t idxDay
    cdef np.float64_t[::1] sprayTimes = np.arange(0, 365 + params.deltaSprayReg, step=params.deltaSprayReg, dtype=np.float64)
    cdef np.int16_t nSprays

    cdef np.int32_t[::1] isActive = np.empty(sprayTimes.shape[0], dtype = np.int32)
    cdef np.float64_t expected_spray_duration
    
    # Biocontrol
    if simSetup.bctrl0 == b"absent":
        loc_bctrl = np.array([], dtype=np.int32)
    elif simSetup.bctrl0 == b"random":
        loc_bctrl = rng.choice(landscape.bctrl_cell, size=setupPrms.bctrl0 if setupPrms.bctrl0 < landscape.bctrl_cell.shape[0] else landscape.bctrl_cell.shape[0], replace = False)
    elif simSetup.bctrl0 == b"full":
        loc_bctrl = landscape.bctrl_cell
    else:
        raise ValueError(f"Invalid biocontrol initial conditions: {simSetup.bctrl0.decode()}")

    for i in range(loc_bctrl.shape[0]):
        idx = loc_bctrl[i]
        com.ctrl.bctrl[idx] = (1.0-landscape.propBctrlCom[idx] * params.bctrlEffect)
        rsd.ctrl.bctrl[idx] = (1.0-landscape.propBctrlRsd[idx] * params.bctrlEffect)
        com.vector.cc_vec[idx] *= com.ctrl.bctrl[idx]
        rsd.vector.cc_vec[idx] *= rsd.ctrl.bctrl[idx]

    # Resistance
    if simSetup.res0 == b"absent":
        loc_res = np.array([], dtype=np.int32)
    elif simSetup.res0 == b"random":
        loc_res = rng.choice(landscape.res_cell, size=setupPrms.resistance0 if setupPrms.resistance0 < landscape.res_cell.shape[0] else landscape.res_cell.shape[0], replace = False) 
    elif simSetup.res0 == b"full":
        loc_res = landscape.res_cell
    else:
        raise ValueError(f"Invalid resistant cells initial conditions: {simSetup.res0.decode()}")

    for i in range(loc_res.shape[0]):
        idx = loc_res[i]
        com.ctrl.inf_res[idx] = landscape.propResCom[idx] * params.infectiousness[0] + (1.0-landscape.propResCom[idx]) * params.infectiousness[1]
        com.ctrl.susc_res[idx] = landscape.propResCom[idx] * params.susceptibility[0] + (1.0-landscape.propResCom[idx]) * params.susceptibility[1]
        rsd.ctrl.inf_res[idx] = landscape.propResRsd[idx] * params.infectiousness[0] + (1.0-landscape.propResRsd[idx]) * params.infectiousness[1]
        rsd.ctrl.susc_res[idx] = landscape.propResRsd[idx] * params.susceptibility[0] + (1.0-landscape.propResRsd[idx]) * params.susceptibility[1]

    # Compliance
    loc_nc = rng.choice(np.arange(0, com.nCells, dtype=np.int32), size=<int>((1.0-setupPrms.compliance) * nCells), replace = False)
    for i in range(loc_nc.shape[0]):
        idx = loc_nc[i]
        mng.compliance[idx] = 0

    # Pick random initial time to select flushing season
    time0 = rng_sim.next_uniform() * 365.0
    if simSetup.flush:
        flush_time = 0.0
        for i in range(params.flushValues.shape[0]):
            if time0 < flush_time + params.deltaFlush[i]:
                mng.current_flush = params.flushValues[i]
                ll.insert(flush_time + params.deltaFlush[i] - time0, Event.FC)
                break
            flush_time += params.deltaFlush[i]

        # No need to recompute total, rates are 0 at the this point
        com.rates.pSdR.change_flush(mng.current_flush)
        rsd.rates.pSdR.change_flush(mng.current_flush)
        com.rates.pMdR.change_flush(mng.current_flush)
        rsd.rates.pMdR.change_flush(mng.current_flush)

    # Background spray
    if simSetup.backgroundSpray:
        com.ctrl._sprayBG[:] = (1.0 - com.ctrl._spray_prop * params.bgSprayEffect) # regardless of compliance

    # Pick random time of next regional surveys (vector and host)
    if simSetup.management:

        if not simSetup.detectPercInfectedCells:
            ll.insert(rng_sim.next_uniform() * params.deltaReg[0], Event.SR00, mng._reg)

        # Spraying (region, NOT BACKGROUND -- generally never active as modReg = 0)
        if simSetup.sprayContinousReg:
            expected_spray_duration = params.durationSpray * 365.0 / params.deltaSprayReg
            params.sprayEffectReg = expected_spray_duration * params.sprayEffect * params.modReg / 365.0
            com.ctrl._sprayReg[:] = (1.0 - (mng._comp * mng._reg) * com.ctrl._spray_prop * params.sprayEffectReg)    # only in compliant cells
        else:
            params.sprayEffectReg = params.sprayEffect * params.modReg
            # Find active sprays
            nSprays = 0
            for j in range(<np.int32_t>(sprayTimes.shape[0])):
                if sprayTimes[j] <= time0 < sprayTimes[j] + params.durationSpray:
                    isActive[nSprays] = j
                    nSprays += 1
            if nSprays > 0:
                # Apply spray effect
                com.ctrl._sprayReg[:] = (1.0 - (mng._comp * mng._reg) * com.ctrl._spray_prop * params.sprayEffectReg) ** nSprays
                # Add spray ending events
                for j in range(nSprays):
                    ll.insert(sprayTimes[isActive[j]] + params.durationSpray, Event.SR0, (mng._comp * mng._reg).copy())  # copy to save where spray took place
                # Add next spray
                ll.insert(sprayTimes[isActive[nSprays-1] + 1], Event.SR1, mng._reg)

            else:
                # Find the next spray time
                idxDay = 0
                while idxDay < sprayTimes.shape[0] and sprayTimes[idxDay] < time0:
                    idxDay += 1
                # Add next spray
                ll.insert(sprayTimes[idxDay], Event.SR1, mng._reg)

        if simSetup.sprayContinousDem:
            expected_spray_duration = params.durationSpray * 365.0 / params.deltaSprayDem
            params.sprayEffectDem = expected_spray_duration * params.sprayEffect / 365.0
        else:
            params.sprayEffectDem = params.sprayEffect

    com.ctrl._spray[:] = com.ctrl._sprayBG * com.ctrl._sprayDem * com.ctrl._sprayReg

    # Infection (HLB+ com cells)
    if simSetup.infected0 >= 0: # if a cell is specified for initial infected
        loc_infec = np.array([simSetup.infected0], dtype=np.int32)
        if com.host.tot[simSetup.infected0] <= 0:
            raise ValueError(f"The selected cell for initial infection {simSetup.infected0} does not have any commercial citrus")
    else:
        loc_infec = rng.choice(np.arange(com.host.tot.shape[0], dtype=np.int32), size=setupPrms.infected0 if setupPrms.infected0 < com.host.tot.shape[0] else com.host.tot.shape[0], p=landscape.weights_org_conv, replace = False)

    for i in range(loc_infec.shape[0]):
        idx = loc_infec[i]

        # Spiral initialization
        flag_init = 0
        while flag_init == 0:
            flag_init, locations, amounts = spiral_initialization(idx, com, space, landscape, setupPrms, simSetup)
            if flag_init == 0:
                if simSetup.infected0 >= 0:
                    raise ValueError("Cell selected for initial infection does not contain enough units to be exposed")
                # try another idx
                flag_idx = 1
                while True:
                    idx = rng.choice(np.arange(com.host.tot.shape[0], dtype=np.int32), size=1, p=landscape.weights_org_conv, replace = False)[0] # get the new idx
                    for k in range(loc_infec.shape[0]):
                        if loc_infec[k] == idx:
                            flag_idx = 0
                            break
                    if flag_idx:
                        break

        for k in range(locations.shape[0]):
            pathogen_arrival(locations[k], amounts[k], com, rsd, 0.0, mng, space, params)
            if simSetup.infested0 == b"host":          # present, not colonised
                vector_arrival(locations[k], com, rsd, 0.0, mng, space)

        # newE = setupPrms.nExposed0 if setupPrms.nExposed0 < com.host.tot[idx] else com.host.tot[idx]
        # pathogen_arrival(idx, newE, com, rsd, 0.0, mng, space, params)

        # if simSetup.infested0 == b"host":          # present, not colonised
        #     vector_arrival(idx, com, rsd, 0.0, mng, space)

    # Infestation (com cells with vector)
    if simSetup.infested0 == b"full":           # colonised everything (com and rsd)
        loc_infest = np.array([], dtype=np.int32)
        mng.vectorFull = 1
        com.summary.totInfested = com.summary.cit_cells
        rsd.summary.totInfested = rsd.summary.cit_cells
        com.summary.vectorFull = 1
        rsd.summary.vectorFull = 1
        com.rates.lddR.change_flush(mng.current_flush)
        rsd.rates.lddR.change_flush(mng.current_flush)
        
        for j in range(nCells):

            space.susc_vMdd_now[j] = 0 # all colonized

            # Update colonisation manually to avoid recomputing total rates
            if com.host.tot[j] > 0:
                com.vector.status_vec[j] = State.colonised
                com.vector.time_vec[1, j] = 0.0
                com.vector.dens_vec[j] = com.vector.cc_vec[j] * com.ctrl.spray[j] * (1.0 - <np.float64_t>(com.host.rem[j])/<np.float64_t>(com.host.tot[j]))
                com.summary.infested[j] = 1
                com.summary.totInfested += 1

                # est rate already 0
                if com.ctrl.inf_res[j]*com.host.pInf[j] > 0:

                    com.rates.lddR.submitRate(j, com.ctrl.inf_res[j] * com.host.pInf[j] * com.rates.rLdd * com.vector.dens_vec[j])

                    if space.susc_pMdd_now[j] > 0:
                        com.rates.pMdR.submitRate(j, space.p_md_pat * com.ctrl.inf_res[j] * com.rates.rInf * com.host.pInf[j] * com.vector.dens_vec[j])

                    update_pSdR(j, space, com, rsd, 0.0, com.vector.dens_vec[j], 0.0, com.host.pInf[j], 0.0, rsd.vector.dens_vec[j], 0.0, rsd.host.pInf[j])

            if rsd.host.tot[j] > 0:
                rsd.vector.status_vec[j] = State.colonised
                rsd.vector.time_vec[1, j] = 0.0
                rsd.vector.dens_vec[j] = rsd.vector.cc_vec[j] * rsd.ctrl.spray[j] * (1.0 - <np.float64_t>(rsd.host.rem[j])/<np.float64_t>(rsd.host.tot[j]))
                rsd.summary.infested[j] = 1
                rsd.summary.totInfested += 1

                # est rate already 0
                if rsd.ctrl.inf_res[j]*rsd.host.pInf[j] > 0:

                    rsd.rates.lddR.submitRate(j, rsd.ctrl.inf_res[j] * rsd.host.pInf[j] * rsd.rates.rLdd * rsd.vector.dens_vec[j])
                    
                    if space.susc_pMdd_now[j] > 0:
                        rsd.rates.pMdR.submitRate(j, space.p_md_pat * rsd.ctrl.inf_res[j] * rsd.rates.rInf * rsd.host.pInf[j] * rsd.vector.dens_vec[j])

                    update_pSdR(j, space, com, rsd, com.vector.dens_vec[j], com.vector.dens_vec[j], com.host.pInf[j], com.host.pInf[j], 0.0, rsd.vector.dens_vec[j], 0.0, rsd.host.pInf[j])

            # No need to update vSdd or vMdd because vector is already everywhere
        
    elif simSetup.infested0 == b"random":          # present, not colonised
        loc_infest = rng.choice(landscape.com_cell, size=setupPrms.infested0 if setupPrms.infested0 < landscape.com_cell.shape[0] else landscape.com_cell.shape[0], p=landscape.weights_org_conv)
        for i in range(loc_infest.shape[0]):
            idx = loc_infest[i]
            vector_arrival(idx, com, rsd, 0.0, mng, space)

    elif simSetup.infested0 == b"host":             # done when setting up infection
        if com.summary.vectorFull + rsd.summary.vectorFull == 2:
            mng.vectorFull = 1
    else:
        raise ValueError(f"Invalid infested cells initial conditions: {simSetup.infested0.decode()}")

cpdef tuple find_next_det_event(np.float64_t t_new, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, 
                Management mng, SurveyHelperClass shc, LinkedList ll, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim):
    """ Finds which deterministic event happens next and does it. """
    cdef np.float64_t next_ll
    cdef np.uint8_t i, n_flush = <np.uint8_t>params.flushValues.shape[0]
    cdef np.int32_t s
    cdef Py_ssize_t j
    cdef np.ndarray _cells_spray = np.zeros(com.nCells, dtype=np.uint8)
    cdef np.uint8_t[::1] cells_spray = _cells_spray
    cdef np.int32_t idx

    ev = ll.find_next_event(t_new)

    if ev.event_type == Event.FC: # flushing
        for i in range(n_flush):
            if params.flushValues[i] == mng.current_flush:
                ll.insert(t_new + params.deltaFlush[(i+1) % n_flush], Event.FC)
                mng.current_flush = params.flushValues[(i+1) % n_flush]
                break

        com.rates.pSdR.change_flush(mng.current_flush)
        rsd.rates.pSdR.change_flush(mng.current_flush)
        com.rates.pMdR.change_flush(mng.current_flush)
        rsd.rates.pMdR.change_flush(mng.current_flush)

        if mng.vectorFull == 1:
            com.rates.lddR.change_flush(mng.current_flush)
            rsd.rates.lddR.change_flush(mng.current_flush)

    elif ev.event_type == Event.SR00:              # regional survey
        event = do_survey(ev, t_new, com, rsd, space, mng, shc, ll, params, simSetup, rng_sim)
        ll.insert(t_new + params.deltaReg[mng.detected], Event.SR00)

    elif ev.event_type == Event.SB00:               # demarcated survey
        event = do_survey(ev, t_new, com, rsd, space, mng, shc, ll, params, simSetup, rng_sim)
        ll.insert(t_new + params.deltaDem[0], Event.SB00)

    elif ev.event_type == Event.SI0:               # demarcated survey
        event = do_survey(ev, t_new, com, rsd, space, mng, shc, ll, params, simSetup, rng_sim)
        ll.insert(t_new + params.deltaDem[1], Event.SI0)

    elif ev.event_type == Event.SR0:              # deactivate spray (regional)
        # ev.cells is a copy of where the spray was applied
        cells_spray = ev.cells
        for j in range(cells_spray.shape[0]):
            if cells_spray[j] == 0:
                continue
            com.ctrl.sprayReg[j] /= (1.0 - com.ctrl.spray_prop[j] * params.sprayEffectReg)
        update_spray_with_rates(cells_spray, com, rsd, mng, space, params)

    elif ev.event_type == Event.SR1:              # activate spray (regional)
        s = 0
        for j in range(mng.regional.shape[0]):
            if mng.regional[j] * mng.compliance[j] == 1:
                com.ctrl.sprayReg[j] *= (1.0 - com.ctrl.spray_prop[j] * params.sprayEffectReg)
                cells_spray[j] = 1
                s += 1
        if s > 0:
            update_spray_with_rates(cells_spray, com, rsd, mng, space, params)
            ll.insert(t_new + params.durationSpray, Event.SR0, _cells_spray.copy())     # copy 
        ll.insert(t_new + params.deltaSprayReg, Event.SR1, mng._reg)

    elif ev.event_type == Event.SD0:              # deactivate spray (demarcated)
        # ev.cells is a copy of where the spray was applied
        cells_spray = ev.cells
        for j in range(cells_spray.shape[0]):
            if cells_spray[j] == 0:
                continue
            com.ctrl.sprayDem[j] /= (1.0 - com.ctrl.spray_prop[j] * params.sprayEffectDem)
        update_spray_with_rates(cells_spray, com, rsd, mng, space, params)

    elif ev.event_type == Event.SD1:              # activate spray (demarcated)
        s = 0
        for j in range(mng.demarcated.shape[0]):
            if mng.demarcated[j] * mng.compliance[j] == 1:
                com.ctrl.sprayDem[j] *= (1.0 - com.ctrl.spray_prop[j] * params.sprayEffectDem)
                cells_spray[j] = 1
                s += 1
        if s > 0:
            update_spray_with_rates(cells_spray, com,  rsd, mng, space, params)
            ll.insert(t_new + params.durationSpray, Event.SD0, _cells_spray.copy())     # copy 
        # Schedule spray ending and next spray
        ll.insert(t_new + params.deltaSprayDem, Event.SD1, mng._dem)

    else:
        raise ValueError(f"Event:{ev.event_type} not supported.")

    next_ll = ll.peek_time()

    return next_ll, ev.event_type

cpdef tuple find_next_sto_event(np.float64_t target_R, np.float64_t time, CellsByType og, CellsByType oth, Management mng, 
                            SpatialStructureCitrus space, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim):
    """ Finds which stochastic event happens next and does it. """
    # General
    cdef np.int8_t event = 0
    cdef np.int32_t idx
    cdef np.float64_t old_pInf

    # Long- and mid-distance dispersal
    cdef np.float64_t p_og, p_inf, p_arr, old_pInf_og, old_pInf_oth
    cdef np.int32_t oX, oY, dX, dY, dIdx = -1
    cdef np.float64_t dSusc, oInf
    cdef np.float64_t theta, dDist
    cdef CellsByType dest, oth_dest
    cdef np.int8_t ldd_out = 1 # flag to track where ldd ends

    # Rates
    cdef np.float64_t est_R = og.rates.estR.total_rate, vSd_R = og.rates.vSdR.total_rate, ldd_R = og.rates.lddR.total_rate, vMd_R = og.rates.vMdR.total_rate
    cdef np.float64_t lat_R = og.rates.latR.total_rate, sym_R = og.rates.symR.total_rate, pSd_R = og.rates.pSdR.total_rate, pMd_R = og.rates.pMdR.total_rate
    cdef np.float64_t rate_vec = est_R + vSd_R + ldd_R + vMd_R
    cdef np.float64_t rate_pat = lat_R + sym_R + pSd_R + pMd_R

    if rate_pat > target_R:                                         # pathogen-related event

        if lat_R > target_R:                                        # E --> C
            event = Event.EC
            idx = og.rates.latR.idxSearch(target_R)
            old_pInf = og.host.pInf[idx]
            og.update_EC(idx, 1, time)

            if mng.vectorFull:
                og.rates.lddR.submitRate(idx, og.host.pInf[idx] * og.ctrl.inf_res[idx] * og.rates.rLdd * og.vector.dens_vec[idx])

            if og.vector.status_vec[idx] == State.colonised:
                update_pSdR(idx, space, og, oth, og.vector.dens_vec[idx], og.vector.dens_vec[idx], old_pInf, og.host.pInf[idx], 
                            oth.vector.dens_vec[idx], oth.vector.dens_vec[idx], oth.host.pInf[idx], oth.host.pInf[idx])

                if space.susc_pMdd_now[idx] > 0:
                    og.rates.pMdR.submitRate(idx, space.p_md_pat * og.rates.rInf * og.ctrl.inf_res[idx] * og.host.pInf[idx] * og.vector.dens_vec[idx])

        elif sym_R > target_R - lat_R:                              # C --> I
            event = Event.CI
            idx = og.rates.symR.idxSearch(target_R - lat_R)
            og.update_CI(idx, 1, time)
            # no changes in pSd, pMd or ldd because C and I contribute the same to pInf

        elif pSd_R > target_R - lat_R - sym_R:                      # S --> E (pSd)
            event = Event.SE
            idx = og.rates.pSdR.idxSearch(target_R - lat_R - sym_R)
            if og.summary.infested[idx]: # only pathogen if already infested
                pathogen_arrival(idx, 1, og, oth, time, mng, space, params)
            else: # rare, but possible if vector is also spreading
                pat_vec_arrival(idx, og, oth, time, mng, space)

        else:                                                       # pMd (potential S --> E)
            idx = og.rates.pMdR.idxSearch(target_R - lat_R - sym_R - pSd_R)

            dIdx = find_destination_md(idx, space.md_pat_mask, space.md_mask_coord_pat, space, rng_sim)

            if dIdx < 0:  # destination cell without citrus
                event = Event.MP0
                return event, idx

            if og.host.sus[dIdx] + oth.host.sus[dIdx] == 0:    # no sus to infect
                event = Event.MP0
                return event, idx

            p_og = <np.float64_t>(og.host.tot[dIdx] - og.host.rem[dIdx]) / <np.float64_t>(og.host.tot[dIdx] - og.host.rem[dIdx] + oth.host.tot[dIdx] - oth.host.rem[dIdx])

            if p_og > rng_sim.next_uniform():      # lands on citrus of type og
                dest = og
                oth_dest = oth
            else:                       # lands on citrus of type oth
                dest = oth
                oth_dest = og

            p_inf = dest.ctrl.susc_res[dIdx] * dest.host.sus[dIdx] / <np.float64_t>(dest.host.tot[dIdx] - dest.host.rem[dIdx])

            if p_inf > rng_sim.next_uniform():
                event = Event.MP1
                pathogen_arrival(dIdx, 1, dest, oth_dest, time, mng, space, params)

            else:
                event = Event.MP0

    else:                                                           # vector-related event
        target_R -= rate_pat
        if ldd_R > target_R:                                        # long-distance dispersal (potential S --> E and/or A --> P)

            idx = og.rates.lddR.idxSearch(target_R)

            oX, oY = space.coord_landscape[idx, :]

            if simSetup.ldd_forced:
                while ldd_out:
                    # Force ldd to fall within landscape
                    # t-student kernel
                    theta = rng_sim.next_uniform() * 2.0 * c_math.M_PI
                    dDist = rng_sim.next_t(3) * params.ld_scale / space.res       # /res to transform into cell distance

                    dX = <np.int32_t>(oX + c_math.round(dDist * c_math.cos(theta)))
                    dY = <np.int32_t>(oY + c_math.round(dDist * c_math.sin(theta)))

                    if 0 <= dX < space.max_row and 0 <= dY < space.max_col:
                        dIdx = space.landscape2d[dX, dY] # -1 where there's no citrus
                        ldd_out = 0
            else:
                # t-student kernel
                theta = rng_sim.next_uniform() * 2.0 * c_math.M_PI
                dDist = rng_sim.next_t(3) * params.ld_scale / space.res       # /res to transform into cell distance

                dX = <np.int32_t>(oX + c_math.round(dDist * c_math.cos(theta)))
                dY = <np.int32_t>(oY + c_math.round(dDist * c_math.sin(theta)))

                if 0 <= dX < space.max_row and 0 <= dY < space.max_col:
                    dIdx = space.landscape2d[dX, dY] # -1 where there's no citrus
                else:
                    dIdx = -1              

            if dIdx < 0:  # destination cell outside of grid or without citrus
                event = Event.L00
                return event, idx

            if og.summary.standing[dIdx] + oth.summary.standing[dIdx] == 0:    # no standing citrus
                event = Event.L00
                return event, idx

            p_og = <np.float64_t>(og.host.tot[dIdx] - og.host.rem[dIdx]) / <np.float64_t>(og.host.tot[dIdx] - og.host.rem[dIdx] + oth.host.tot[dIdx] - oth.host.rem[dIdx])

            if p_og > rng_sim.next_uniform():      # lands on citrus of type og
                dest = og
                oth_dest = oth
            else:                       # lands on citrus of type oth
                dest = oth
                oth_dest = og

            p_arr = space.clim[dIdx] * dest.ctrl.bctrl[dIdx] * dest.ctrl.spray[dIdx] 
            if p_arr > rng_sim.next_uniform(): # vector arrives
                dSusc = dest.ctrl.susc_res[dIdx] * dest.host.sus[dIdx] / <np.float64_t>(dest.host.tot[dIdx] - dest.host.rem[dIdx]) # susceptibility of destination cell
                
                if mng.vectorFull == 1:
                    # infectiousness of origin cell already considered
                    p_inf = dSusc 
                else:
                    oInf = og.ctrl.inf_res[idx] * og.host.pInf[idx]      # infectiousness of origin cell
                    p_inf = oInf * dSusc * mng.current_flush

                if p_inf > rng_sim.next_uniform(): # pathogen arrives

                    if dest.vector.status_vec[dIdx] == State.absent:   # vector + pathogen
                        event = Event.L11
                        pat_vec_arrival(dIdx, dest, oth_dest, time, mng, space)

                    else: # only pathogen
                        event = Event.L01
                        pathogen_arrival(dIdx, 1, dest, oth_dest, time, mng, space, params)

                elif dest.vector.status_vec[dIdx] == State.absent:   # only vector
                    event = Event.L10
                    vector_arrival(dIdx, dest, oth_dest, time, mng, space)
                
                else: # no vector, no pathogen
                    event = Event.L00
            else:
                event = Event.L00

        elif vMd_R > target_R - ldd_R:                              # vMd (potential  A --> P)
            idx = og.rates.vMdR.idxSearch(target_R - ldd_R)

            dIdx = find_destination_md(idx, space.md_vec_mask, space.md_mask_coord_vec, space, rng_sim)

            if dIdx < 0:  # destination cell without citrus
                event = Event.MV0
                return event, idx

            if (og.summary.standing[dIdx] + oth.summary.standing[dIdx] == 0) or (og.vector.status_vec[dIdx] != State.absent and oth.vector.status_vec[dIdx] != State.absent):    # no standing trees or vector already present (or no vector population possible because cell is empty)
                event = Event.MV0
                return event, idx

            p_og = <np.float64_t>(og.host.tot[dIdx] - og.host.rem[dIdx]) / <np.float64_t>(og.host.tot[dIdx] - og.host.rem[dIdx] + oth.host.tot[dIdx] - oth.host.rem[dIdx])

            if p_og > rng_sim.next_uniform():      # lands on citrus of type og
                dest = og
                oth_dest = oth
            else:                       # lands on citrus of type oth
                dest = oth
                oth_dest = og

            p_arr = space.clim[dIdx] * dest.ctrl.bctrl[dIdx] * dest.ctrl.spray[dIdx] * (dest.vector.status_vec[dIdx] == State.absent)

            if p_arr > rng_sim.next_uniform():
                event = Event.MV1
                vector_arrival(dIdx, dest, oth_dest, time, mng, space)
                
            else:
                event = Event.MV0

        elif est_R > target_R - ldd_R - vMd_R:                      # P --> C (vector)
            event = Event.ES
            idx = og.rates.estR.idxSearch(target_R - ldd_R - vMd_R)
            og.update_ES(idx, time)

            if mng.vectorFull == 0:
                og.rates.lddR.submitRate(idx, og.rates.rLdd * og.vector.dens_vec[idx])

                update_vSdR(idx, space, og, oth, 0, og.vector.dens_vec[idx], oth.vector.dens_vec[idx], oth.vector.dens_vec[idx])

                if space.susc_vMdd_now[idx] > 0:
                    og.rates.vMdR.submitRate(idx, space.p_md_vec * og.rates.rMdd * og.vector.dens_vec[idx])

                if og.host.pInf[idx] * og.ctrl.inf_res[idx] > 0:
                    update_pSdR(idx, space, og, oth, 0, og.vector.dens_vec[idx], og.host.pInf[idx], og.host.pInf[idx], 
                                oth.vector.dens_vec[idx], oth.vector.dens_vec[idx], oth.host.pInf[idx], oth.host.pInf[idx])

                    if space.susc_pMdd_now[idx] > 0:
                        og.rates.pMdR.submitRate(idx, space.p_md_pat * og.rates.rInf * og.ctrl.inf_res[idx] * og.host.pInf[idx] * og.vector.dens_vec[idx])

            elif og.host.pInf[idx] * og.ctrl.inf_res[idx] > 0:
                og.rates.lddR.submitRate(idx, og.host.pInf[idx] * og.ctrl.inf_res[idx] * og.rates.rLdd * og.vector.dens_vec[idx])

                update_pSdR(idx, space, og, oth, 0, og.vector.dens_vec[idx], og.host.pInf[idx], og.host.pInf[idx], 
                                oth.vector.dens_vec[idx], oth.vector.dens_vec[idx], oth.host.pInf[idx], oth.host.pInf[idx])

                if space.susc_pMdd_now[idx] > 0:
                    og.rates.pMdR.submitRate(idx, space.p_md_pat * og.rates.rInf * og.ctrl.inf_res[idx] * og.host.pInf[idx] * og.vector.dens_vec[idx])

        else:                                                       # A --> P (vSd)
            event = Event.AR
            idx = og.rates.vSdR.idxSearch(target_R - ldd_R - est_R - vMd_R)
            vector_arrival(idx, og, oth, time, mng, space)

    return event, idx

cpdef tuple simulation(int run, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, LinkedList ll, LandscapePostprocessing landscape, SimulationParameters params, SetupParameters setupPrms, SimulationSetup simSetup, SimulationRNG rng_sim, SaveStructure save_output):
    """ Simulates simSetup.nRuns with the same parameters """

    # cdef int nRuns = simSetup.nRuns
    cdef np.int8_t event
    cdef np.int8_t reason = 0                                   # save reason simulation ends
    cdef np.float64_t next_ll, t_new, dt
    cdef np.float64_t com_rate, rsd_rate, tot_rate, target_rate
    cdef np.uint8_t n_flush = <np.uint8_t>params.flushValues.shape[0]
    cdef np.int32_t totS
    cdef np.float64_t start_sim
    cdef np.float64_t stop_at = simSetup.percStop * (com.summary.citrus_total+rsd.summary.citrus_total)
    cdef np.float64_t stop_at_cells = (1-simSetup.percStop) * (com.host.sus.shape[0])
    cdef np.float64_t detect_at_cells = simSetup.percDetect * (com.host.sus.shape[0])
    cdef np.float64_t stop_at_vec = (1.0 - simSetup.percStop)
    cdef np.uint8_t survey_notdone = True if simSetup.management else False # don't have mandatory detection if management is not even on

    # Save outputs
    cdef Py_ssize_t sp_new, sp_last, sp_new_yield, sp_last_yield
    cdef np.float64_t last_save = 0.0, next_save = simSetup.tMax + 1
    cdef np.float64_t last_save_yield = 0.0, next_save_yield = simSetup.tMax + 1

    # Start
    cdef np.uint64_t step = 0
    cdef np.float64_t t = 0.0

    # Reset classes from previous simulation
    com.reset(landscape.cc_com)
    rsd.reset(landscape.cc_rsd)
    space.reset()
    mng.reset()
    ll.reset()
    shc.reset_full()

    mng.switchRemoval = simSetup.switchRemoval # save correct one

    totS = com.summary.S_tot + rsd.summary.S_tot

    # Random initial conditions
    initialize_run(com, rsd, space, mng, ll, landscape, params, setupPrms, simSetup, rng_sim)

    # Save initial conditions
    if simSetup.sim_goal == SimGoal.maps:
        save_output.update_maps(-1, 0, com.host, rsd.host, mng)
        next_save = last_save + 365.0 / <np.float64_t>(simSetup.spyrMaps)
        sp_last = 0
    elif simSetup.sim_goal == SimGoal.traj:
        save_output.update_trajectories(run, -1, 0, com, rsd)
        next_save = last_save + 365.0 / <np.float64_t>(simSetup.spyrTraj)
        sp_last = 0
    elif simSetup.sim_goal == SimGoal.calibration:
        save_output.update_cali_trajectories(run, -1, 0, com, rsd)
        next_save = last_save + 365.0 / <np.float64_t>(simSetup.spyrTraj)
        sp_last = 0
    elif simSetup.sim_goal == SimGoal.metrics_traj:
        save_output.update_trajectories(run, -1, 0, com, rsd)
        next_save = last_save + 365.0 / <np.float64_t>(simSetup.spyrTraj)
        next_save_yield = last_save_yield + 365.0 / <np.float64_t>(simSetup.spyrMetrics)
        sp_last_yield = 0   # year 0 = yield is 0
        sp_last = 0
    elif simSetup.sim_goal == SimGoal.metrics:
        next_save_yield = last_save_yield + 365.0 / <np.float64_t>(simSetup.spyrMetrics)
        sp_last_yield = 0   # year 0 = yield is 0


    next_ll = ll.peek_time()    # time of next deterministic event

    start_sim = time.time()

    # Simulation
    while t < simSetup.tMax and step < simSetup.nSteps:
        event = 0   # reset event

        com_rate = com.rates.get_total_rate()
        rsd_rate = rsd.rates.get_total_rate()

        tot_rate = com_rate + rsd_rate

        if tot_rate > 0:
            dt = -c_math.log(1.0 - rng_sim.next_uniform()) / tot_rate       # avoid log(0)
            t_new = t + dt
        elif mng.current_flush == 0 and mng.vectorFull:
            t_new = next_ll                             # do deterministic
        else:
            reason = -1
            print(f"Total rate: {tot_rate}. pSdd_com: {com.rates.pSdR.total_rate}. pSdd_rsd: {rsd.rates.pSdR.total_rate}", flush = True)
            break
        
        if t_new >= next_ll and next_ll <= simSetup.tMax:       # deterministic event
            t_new = next_ll
            next_ll, event = find_next_det_event(t_new, com, rsd, space, mng, shc, ll, params, simSetup, rng_sim)

        elif t_new <= simSetup.tMax:                            # stochatic event
            target_rate = tot_rate * rng_sim.next_uniform()
            if com_rate > target_rate:                  # event concerns commercial citrus
                event, cell = find_next_sto_event(target_rate, t_new, com, rsd,  mng, space, params, simSetup, rng_sim)
            else:                                       # event concerns residential citrus
                event, cell = find_next_sto_event(target_rate - com_rate, t_new, rsd, com, mng, space, params, simSetup, rng_sim)
        else:
            reason = -2      # over time
            break
        
        if simSetup.detectPercInfectedCells and survey_notdone and (space._infCells.sum()) >= detect_at_cells:
            # Have a successful regional survey detection, then switch to default surveillance and management
            make_detection_happen(t, com, rsd, space, mng, shc, ll, params, simSetup, rng_sim)
            ll.insert(t_new + params.deltaReg[mng.detected], Event.SR00)
            survey_notdone = False

        # >> Update savepoint <<
        # Trajectories
        if t_new >= next_save:
            if simSetup.sim_goal == SimGoal.maps:  # maps
                sp_new = <int>c_math.floor((t_new*simSetup.spyrMaps / 365.0))
                if sp_new <= simSetup.spyrMaps*simSetup.nYears:
                    save_output.update_maps(sp_last, sp_new, com.host, rsd.host, mng)
                last_save = next_save
                next_save = (sp_new + 1) * 365.0 / <np.float64_t>(simSetup.spyrMaps)
                sp_last = sp_new

            elif simSetup.sim_goal == SimGoal.calibration:  # calibration
                sp_new = <int>c_math.floor((t_new*simSetup.spyrTraj / 365.0))
                if sp_new <= simSetup.spyrTraj*simSetup.nYears:
                    save_output.update_cali_trajectories(run, sp_last, sp_new, com, rsd)
                last_save = next_save
                next_save = (sp_new + 1) * 365.0 / <np.float64_t>(simSetup.spyrTraj)
                
                sp_last = sp_new

            else: # trajectories
                sp_new = <int>c_math.floor((t_new*simSetup.spyrTraj / 365.0))
                if sp_new <= simSetup.spyrTraj*simSetup.nYears:
                    save_output.update_trajectories(run, sp_last, sp_new, com, rsd)
                last_save = next_save
                next_save = (sp_new + 1) * 365.0 / <np.float64_t>(simSetup.spyrTraj)
                
                sp_last = sp_new

        if t_new >= next_save_yield:
            sp_new_yield = <int>c_math.floor((t_new*simSetup.spyrMetrics / 365.0))
            save_output.update_metrics(run, sp_last_yield , sp_new_yield, com, rsd, mng)
            last_save_yield = next_save_yield
            next_save_yield = (sp_new_yield + 1) * 365.0 / <np.float64_t>(simSetup.spyrMetrics)
            sp_last_yield = sp_new_yield

        t = t_new
        step += 1

        if simSetup.stopDetection and (mng.detected > 0):
            reason = 2
            break
        if simSetup.stopPercInfected and (com.summary.S_tot + rsd.summary.S_tot) <= stop_at:
            reason = 4
            break
        if com.summary.totInfected + rsd.summary.totInfected == 0:  # disease free
            reason = 5
            break
        if simSetup.stopPercInfested and save_output.traj_V_cells[run, sp_last] >= stop_at_vec:
            reason = 6
            break
        if simSetup.stopPercInfectedCells and (space._infCells.sum()) >= stop_at_cells:
            reason = 7
            print(f"Stopping at {(1-simSetup.percStop)*100:.2f} %  cells infected. Works properly only when not removing.")
            break
        
    print(f"Simulation {run} ended. Reason: {reason}. Time: {t}. Step: {step - 1}. Elapsed time {time.time() - start_sim}")

    if (simSetup.sim_goal == SimGoal.traj or simSetup.sim_goal == SimGoal.metrics_traj) and sp_last < save_output.traj_P_units.shape[1]:
        sp_next = <int>c_math.floor((next_save*simSetup.spyrTraj / 365.0))
        if sp_next >= save_output.traj_P_units.shape[1] - 1:
            sp_next = save_output.traj_P_units.shape[1] - 1

        save_output.update_trajectories_end(run, sp_last, sp_next, com, rsd)
    if (simSetup.sim_goal == SimGoal.calibration) and sp_last < save_output.traj_P_units.shape[1]:
        sp_next = <int>c_math.floor((next_save*simSetup.spyrTraj / 365.0))
        if sp_next >= save_output.traj_P_units.shape[1] - 1:
            sp_next = save_output.traj_P_units.shape[1] - 1

        save_output.update_cali_trajectories_end(run, sp_last, sp_next, com, rsd)
            
    if (simSetup.sim_goal == SimGoal.maps) and sp_last < save_output.maps_S.shape[0]:
        sp_next = <int>c_math.floor((next_save*simSetup.spyrMaps / 365.0))
        if sp_next >= save_output.maps_S.shape[0] - 1:
            sp_next = save_output.maps_S.shape[0] - 1
        save_output.update_maps_end(sp_last, sp_next, com.host, rsd.host, mng)

    if (simSetup.sim_goal == SimGoal.metrics or simSetup.sim_goal == SimGoal.metrics_traj) and sp_last_yield < save_output.yield_proxy_s_c0.shape[1]:
        sp_next = <int>c_math.floor((next_save_yield*simSetup.spyrMetrics / 365.0))
        
        if sp_next >= save_output.yield_proxy_s_c0.shape[1]:
            sp_next = save_output.yield_proxy_s_c0.shape[1]
        save_output.update_metrics_end(run, sp_last_yield , sp_next, com, rsd, mng)

    if simSetup.sim_goal == SimGoal.cell_status:
        save_output.save_cell_status(run, com, rsd)
    
    return reason, t

cpdef tuple run_simulations(int n, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, 
                Management mng, LinkedList ll, LandscapePostprocessing landscape, SimulationParameters params, 
                SetupParameters setupPrms, SimulationSetup simSetup, BitGenerator bg):
    """ Runs [n] simulations of the epidemic """
    cdef SimulationRNG rng_sim = SimulationRNG(bg)
    cdef np.float64_t[::1] detection_results = np.zeros(13, dtype=np.float64)
    cdef np.int16_t nSpMaps = 0, nSpTraj = 0, nSpYield = 0, n_Runs = 0
    cdef SaveStructure save_output
    cdef SaveMetrics metrics

    if simSetup.sim_goal == SimGoal.traj:
        nSpMaps = 0
        nSpTraj = simSetup.nYears * simSetup.spyrTraj + 1
        nSpYield = 0
        n_Runs = simSetup.nRuns

    elif simSetup.sim_goal == SimGoal.calibration:
        nSpMaps = 0
        nSpTraj = simSetup.nYears * simSetup.spyrTraj + 1
        nSpYield = 0
        n_Runs = simSetup.nRuns
        
    elif simSetup.sim_goal == SimGoal.maps:
        nSpMaps = simSetup.nYears * simSetup.spyrMaps + 1
        nSpTraj = 0
        nSpYield = 0
        n_Runs = 0

    elif simSetup.sim_goal == SimGoal.metrics:
        nSpMaps = 0
        nSpTraj = 0
        nSpYield = simSetup.nYears * simSetup.spyrMetrics
        n_Runs = simSetup.nRuns
        
    elif simSetup.sim_goal == SimGoal.metrics_traj:
        nSpMaps = 0
        nSpTraj = simSetup.nYears * simSetup.spyrTraj + 1
        nSpYield = simSetup.nYears * simSetup.spyrMetrics
        n_Runs = simSetup.nRuns

    elif simSetup.sim_goal == SimGoal.cell_status:
        nSpMaps = 0
        nSpTraj = 0
        nSpYield = 0
        n_Runs = simSetup.nRuns
        
    save_output = SaveStructure(com.nCells, nSpMaps, nSpTraj, nSpYield, n_Runs)
    metrics = SaveMetrics(n_Runs)

    shc = SurveyHelperClass(com.nCells)

    for k in range(n):

        reason, t = simulation(k, com, rsd, space, mng, shc, ll, landscape, params, setupPrms, simSetup, rng_sim, save_output)

        if simSetup.sim_goal == SimGoal.metrics or simSetup.sim_goal == SimGoal.metrics_traj:
            detection_results[0] = mng.which_det
            detection_results[1] = mng.time_det
            detection_results[2] = mng.cell_inc_det
            detection_results[3] = mng.cit_inc_det
            detection_results[4] = mng.n_vec_det
            detection_results[5] = mng.n_vis_det
            detection_results[6] = mng.n_pcr_det
            detection_results[7] = mng.cell1km_inc_det
            
            # Practical eradication
            detection_results[8] = mng.time_erad
            detection_results[9] = mng.cit_inc_erad
            detection_results[10] = mng.cell_inc_erad
            detection_results[11] = mng.cell1km_inc_erad

            # Switched removal strategy
            detection_results[12] = mng.time_stop_attempt

            metrics.update_run(k, com, rsd, shc, detection_results, reason, t)

    return save_output, metrics










