# structs.pyx

import numpy as np
cimport numpy as np
cimport cython
from .ratesStructs cimport SqrtBlocks, SqrtBlocks_with_base, SegTree, SegTree_with_base
cimport libc.math as c_math
from libc.string cimport memset

cdef class SimulationSetup:
    """Container for simulation setup parameters."""
    
    def __cinit__(self, bytes name, bytes region, bytes vector, np.int8_t sim_goal, np.int16_t nRuns, np.int16_t nYears):
        self.sim_name = name
        self.region = region
        self.vector = vector
        self.sim_goal = sim_goal
        self.nRuns = nRuns
        self.nYears = nYears
        self.tMax = nYears * 365.0

        # Save points per year
        self.spyrMaps = 4   # yearly
        self.spyrTraj = 365  # daily
        self.spyrMetrics = 1  # yearly

        self.nSteps = 10000000000
        self.switchRemoval = False
        self.stopDetection = False
        self.stopPercInfected = False
        self.stopPercInfectedCells = False
        self.stopPercInfested = False
        self.detectPercInfectedCells = False
        self.percDetect = 1.1 # proportion infected when detection force to happen
        self.percStop = 0.0 # proportion susceptibles remaining to stop simulation
        self.management = True
        self.removal = True
        self.flush = True
        self.backgroundSpray = True
        self.sprayContinousDem = True
        self.sprayContinousReg = True
        self.infected0 = -1             # specify a cell if needed
        self.infested0 = b"full"         # options: random, full, current, host
        self.bctrl0 = b"absent"          # options: random, full, absent
        self.res0 = b"absent"            # options: random, full, absent

        # Ldd type
        self.ldd_forced = False

cdef class SetupParameters:
    """Container for setup parameters (i.e., only used at beginning of simulation)."""

    def __cinit__(self):
        self.resolution = 0.1
        self.res_big = 1.0
        self.maxCitCom = 25
        self.maxCitRsd = 500

        # Initialization
        self.infected0 = 1          # overriden if SimulationSetup.inf0 == "full"
        self.infested0 = 1          # overriden if SimulationSetup.infest0 == "current" or "host" or "full"
        self.bctrl0 = 5             # overriden if SimulationSetup.bctrl0 == "absent"
        self.resistance0 = 5        # overriden if SimulationSetup.res0 == "absent"
        self.nExposed0 = 1

        # Dispersal and management masks
        self.sd_scale = 1.96
        self.sd_dist_vec = 6.0
        self.sd_dist_pat = 0.6
        self.md_max_dist = 6
        self.propWithin = 0.35
        self.compliance = 0.9
        self.r_dem = 3.0
        self.r_search = 1.0
        self.r_rem = 0.0

cdef class SimulationParameters:
    """Container for simulation parameters."""

    def __cinit__(self):

        # Transition rates
        self.rDisp = 300.0    # NEW VALUE
        self.rInf = 7.4     # NEW VALUE (with flush)
        self.rEC = 1.0 / 365.0
        self.rCI = 5.0 / 365.0
        self.rEst = 1.0 / 365.0
        self.ld_scale = 130.0 # [km]
        self.ldProp = 10 ** (-3) # proportion of long distance dispersal
        self.pE = 0.0
        self.pCI = 1.0

        ### Management: surveys
        self.deltaReg = np.array([365.0, 365.0], dtype = np.float64)
        self.deltaDem = np.array([365.0 / 2, 365.0 / 4], dtype = np.float64) # buffer, infected

        self.survey_by_prop = np.array([0, 0, 1, 1], dtype = np.uint8) # reg, buf, inf, search
        self.time_for_eradication = 3*365
        self.removal_duration = 30*365

        ## Survey by proportion of cells 
        self.pComSurv= np.array([[0.02, 0.0, 0.0, 0.0], [0.02, 0.25, 1.0, 1.0]], dtype = np.float64) # before, after
        self.pRsdSurv= np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.125, 0.5, 0.5]], dtype = np.float64) 

        ## Survey by design prevalence
        self.dp = np.array([[0.01, 0.001, 0.001, 0.001], [0.0, 0.001, 0.001, 0.001]], dtype = np.float64) # com, rsd (0 if no survey) - rsd needs to be the same as com when > 0 because prop_status is defined based on commercial prevalence
        self.cl = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float64)
        self.prop_status = np.array([[0.07, 0.08, 0.08, 0.08], [0.38, 0.37, 0.37, 0.37]], dtype = np.float64) # crp, inf (technically defined based on design prevalence), doesn't sum to 1 necessarily because part of HLB+ are exposed. These are default values (assuming 1%, 0.1%, 0.1%, 0.1%), should be corrected based on simulations when running simulations
        self.reduce_rsd = 0.5 # only % of cells actually surveyed of rsd

        # Common to both survey types
        self.nHostVis = np.array([[5, 0, 0, 0], [5, 5, 25, 25]], dtype = np.uint16)
        self.pHostPCR = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.1]], dtype = np.float64)
        self.nVec = np.array([[5, 0, 0, 0], [5, 5, 0, 0]], dtype = np.uint8)
        self.pVec = 1.0
        self.pVis = 0.5 
        self.pPCR = np.array([0.5, 0.5], dtype = np.float64) # cryptic, infected
        self.w_rem = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64) # weights for inf, crp, exp, sus for both com (first 4) and rsd (last 4), in the order.
        self.pNoSearch = 0.9 # reduction in weight for selecting cells already surveyed in search area
        self.pRem = 1.0

        # Management: control
        self.infectiousness = np.array([0.0, 1.0], dtype = np.float64)
        self.susceptibility = np.array([0.0, 1.0], dtype = np.float64)
        self.bctrlEffect = 0.2
        self.durationSpray = 15
        self.bgSprayEffect = 0.9                                    # nominal value of background spray effect (conventional commercial only)
        self.sprayEffect = 0.25                                      # nominal value of spray effect
        self.modReg = 0.0                                           # change in nominal value of spray effect in regional area compared to demarcated
        self.sprayEffectDem = self.sprayEffect                      # proportion of vectors removed after insecticides application (demarcated area) 
        self.sprayEffectReg = self.sprayEffect * self.modReg        # proportion of vectors removed after insecticides application (regional area) 
        self.deltaSprayDem = 15.0             
        self.deltaSprayReg = 365.0 / 4.0

        # Flush season
        self.flushValues = np.array([1.0, 0.125, 0], dtype = np.float64)
        self.deltaFlush = np.array([365/4, 365/2, 365/4], dtype = np.float64)      # time interval to next flush

cdef class LandscapePostprocessing:
    """ Container for processed landscape arrays to use when setting up initial conditions of simulations """

    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] propBctrlCom, np.ndarray[np.float64_t, ndim=1] propBctrlRsd, np.ndarray[np.float64_t, ndim=1] propResCom, np.ndarray[np.float64_t, ndim=1] propResRsd, np.ndarray[np.float64_t, ndim=1] cc_com, np.ndarray[np.float64_t, ndim=1] cc_rsd, np.ndarray[np.int32_t, ndim=1] com_cell, np.ndarray[np.int32_t, ndim=1] bctrl_cell, np.ndarray[np.int32_t, ndim=1] res_cell, np.ndarray[np.float64_t, ndim=1] weights_org_conv):

        self.cc_com = cc_com
        self.cc_rsd = cc_rsd

        # Proportions
        self.propBctrlCom = propBctrlCom
        self.propBctrlRsd = propBctrlRsd
        self.propResCom = propResCom
        self.propResRsd = propResRsd

        # Cells with properties
        self.com_cell = com_cell
        self.bctrl_cell = bctrl_cell
        self.res_cell = res_cell
        self.weights_org_conv = weights_org_conv

cdef class SpatialStructureCitrus:
    """Container for spatial structure information of citrus cells."""
    
    def __cinit__(self, np.float64_t res, np.float64_t res_big, np.int32_t[:, ::1] landscape2d, np.int32_t[:, ::1] coord_landscape, np.float64_t[::1] clim, np.int16_t[:, ::1] sd_mask_coord_vec, np.int16_t[:, ::1] sd_mask_coord_pat, np.float64_t[::1] vecKern, np.float64_t[::1] patKern, np.int16_t[:, ::1] md_mask_coord_vec, np.int16_t[:, ::1] md_mask_coord_pat, np.float64_t[::1] md_vec_mask, np.float64_t[::1] md_pat_mask, np.float64_t[::1] rem_prop_mask, np.uint8_t[:, ::1] rem_coord, np.uint8_t[:, ::1] dem_coord, np.uint8_t[:, ::1] search_coord):

        # >> General <<
        self.res = res
        self.landscape2d = landscape2d  # matrix with cell idx
        self.coord_landscape = coord_landscape
        self.max_row = landscape2d.shape[0]
        self.max_col = landscape2d.shape[1]
        self.clim = clim

        self.coord_transf = np.array([[+1, 0, 0, +1], # slice 0
                             [+1, 0, 0, -1], # 1
                             [0, +1, -1, 0], # 2
                             [0, -1, -1, 0], # 3
                             [-1, 0, 0, -1], # 4
                             [-1, 0, 0, +1], # 5
                             [0, -1, +1, 0], # 6
                             [0, +1, +1, 0] # 7
                             ], dtype=np.int8)

        # >> Short-distance dispersal << full mask
        self.sd_mask_coord_pat = sd_mask_coord_pat
        self.sd_mask_coord_vec = sd_mask_coord_vec
        self.vecKern = vecKern
        self.patKern = patKern

        # >> Mid-distance dispersal << full mask
        self.md_mask_coord_pat = md_mask_coord_pat
        self.md_mask_coord_vec = md_mask_coord_vec
        self.md_vec_mask = md_vec_mask
        self.md_pat_mask = md_pat_mask
        self.p_md_vec = md_vec_mask[md_vec_mask.shape[0] - 1] if md_vec_mask.shape[0] > 0 else 0
        self.p_md_pat = md_pat_mask[md_pat_mask.shape[0] - 1] if md_pat_mask.shape[0] > 0 else 0

        # Initial values of susceptibles cells to mid-distance dispersal
        self._susc_vMdd_0 = np.zeros(clim.shape[0], dtype = np.uint32)
        self.susc_vMdd_0 = self._susc_vMdd_0
        self._susc_pMdd_0 = np.zeros(clim.shape[0], dtype = np.uint32)
        self.susc_pMdd_0 = self._susc_pMdd_0

        self.susc_vMdd_now = self._susc_vMdd_0.copy()
        self.susc_pMdd_now = self._susc_pMdd_0.copy()

        # >> Control & Management << 1/8 mask
        self.rem_coord = rem_coord
        self.rem_p_mask = rem_prop_mask
        self.dem_coord = dem_coord
        self.search_coord = search_coord

        self.default_0_rem_mask = np.zeros((0), dtype=np.float64)
        self.rem_mask_to_use = self.rem_p_mask

        # >> Keep track of infected cells
        self.res_big = res_big
        self._infCells = np.zeros(clim.shape[0], dtype=np.uint8)
        self.infCells = self._infCells
        self._infCells1km = np.zeros(int((self.landscape2d.shape[0] * (self.res / self.res_big))*(self.landscape2d.shape[1] * (self.res / self.res_big))), dtype=np.uint8)
        self.infCells1km = self._infCells1km

        # Map to bigger landscape
        self._small_to_big = -np.ones(self.coord_landscape.shape[0], dtype=np.int32)
        self.small_to_big = self._small_to_big
        self.compute_small_to_big()
        self.unique_1km = np.unique(self._small_to_big).shape[0]

    cdef compute_small_to_big(self):
        cdef Py_ssize_t i
        cdef np.int32_t r, c, r_big_i, c_big_i, c_big = np.int32(self.landscape2d.shape[1] * self.res / self.res_big)

        for i in range(self.coord_landscape.shape[0]):
            r = self.coord_landscape[i, 0]
            c = self.coord_landscape[i, 1]
            
            r_big_i = <np.int32_t>(r // (self.res_big / self.res))
            c_big_i = <np.int32_t>(c // (self.res_big / self.res))
            
            val = c_big*r_big_i + c_big_i
            self.small_to_big[i] = val

    cpdef void reset(self):
        self.susc_vMdd_now = self._susc_vMdd_0.copy()
        self.susc_pMdd_now = self._susc_pMdd_0.copy()

        self._infCells.fill(0)
        self._infCells1km.fill(0)

        self.rem_mask_to_use = self.rem_p_mask # reset removal mask

    cpdef void compute_arrivals_mdd_pat(self, np.uint16_t[::1] com_units, np.uint16_t[::1] rsd_units):
        """ Computes the number of potential arrival points for pathogen mdd nearby any given cell. 
        Note: if the sliced mask is being used, the number does not correspond to the number of cells itself, but rather the amount of times the slice can point to a given cell (e.g., the central cell, origin of movement, is counted 8 times, because the slice represents 1/8 and each rotation of it will point to the central cell once)
        Note: rsd and com counted as one"""
        cdef Py_ssize_t i, j
        cdef np.int32_t idx, r, c, r_idx, c_idx

        for i in range(self.susc_pMdd_0.shape[0]):
            r, c = self.coord_landscape[i, :]
            for j in range(self.md_mask_coord_pat.shape[0]):
                r_idx = r + self.md_mask_coord_pat[j, 0]
                c_idx = c + self.md_mask_coord_pat[j, 1]
                if 0 <= r_idx < self.max_row and 0 <= c_idx < self.max_col:
                    idx = self.landscape2d[r_idx, c_idx]
                    if idx < 0:
                        continue
                    else:
                        self.susc_pMdd_0[i] += (com_units[idx] + rsd_units[idx] > 0)

        self.susc_pMdd_now = self._susc_pMdd_0.copy()

    cpdef void compute_arrivals_mdd_vec(self, np.uint16_t[::1] com_units, np.uint16_t[::1] rsd_units):
        """ Computes the number of potential arrival points for vector mdd nearby any given cell. 
        Note: if the sliced mask is being used, the number does not correspond to the number of cells itself, but rather the amount of times the slice can point to a given cell (e.g., the central cell, origin of movement, is counted 8 times, because the slice represents 1/8 and each rotation of it will point to the central cell once)
        Note: rsd and com counted as one"""
        cdef Py_ssize_t i, j
        cdef np.int32_t idx, r, c, r_idx, c_idx

        for i in range(self.susc_vMdd_0.shape[0]):
            r, c = self.coord_landscape[i, :]
            for j in range(self.md_mask_coord_vec.shape[0]):
                r_idx = r + self.md_mask_coord_vec[j, 0]
                c_idx = c + self.md_mask_coord_vec[j, 1]
                if 0 <= r_idx < self.max_row and 0 <= c_idx < self.max_col:
                    idx = self.landscape2d[r_idx, c_idx]
                    if idx < 0:
                        continue
                    else:
                        self.susc_vMdd_0[i] += (com_units[idx] + rsd_units[idx] > 0)

        self.susc_vMdd_now = self._susc_vMdd_0.copy()

cdef class Management:
    """ Container for management measures """

    def __cinit__(self, np.int32_t nCells):
        self.nCells = nCells
        self._reg = np.ones(nCells, dtype=np.uint8)
        self._dem = np.zeros(nCells, dtype=np.uint8)
        self._buf = np.zeros(nCells, dtype=np.uint8)
        self._inf = np.zeros(nCells, dtype=np.uint8)
        self._comp = np.ones(nCells, dtype=np.uint8)
        self._used = np.zeros(nCells, dtype=np.uint8)
        self.used_for_dem = self._used
        self.demarcated = self._dem
        self.buffer = self._buf
        self.infected = self._inf
        self.regional = self._reg
        self.compliance = self._comp
        self.current_flush = 1.0
        self.vectorFull = 0
        self.detected = 0
        self.totReg = nCells

        # Surveys results
        self.which_det = -1
        self.time_det = 0.0
        self.cit_inc_det = 0.0
        self.cell_inc_det = 0.0
        self.cell1km_inc_det = 0.0
        self.time_stop_attempt = 0.0
        self.keep_attempt = 1
        self.switchRemoval = 0

        # Practical eradication
        self.time_last_det = 0.0
        self.believed_eradicated = 0
        self.time_erad = -1.0
        self.cit_inc_erad = 0.0
        self.cell_inc_erad = 0.0
        self.cell1km_inc_erad = 0.0

    cpdef reset(self):
        self._reg.fill(1)
        self._dem.fill(0)
        self._buf.fill(0)
        self._inf.fill(0)
        self._comp.fill(1)
        self._used.fill(0)
        self.current_flush = 1.0
        self.totReg = self.nCells
        self.vectorFull = 0
        self.detected = 0

        # Surveys results
        self.which_det = -1
        self.time_det = np.nan
        self.cit_inc_det = 0.0
        self.cell_inc_det = 0.0
        self.cell1km_inc_det = 0.0
        self.time_stop_attempt = 0.0
        self.keep_attempt = 1

        # Practical eradication
        self.time_last_det = 0.0
        self.believed_eradicated = 0
        self.time_erad = -1.0
        self.cit_inc_erad = 0.0
        self.cell_inc_erad = 0.0
        self.cell1km_inc_erad = 0.0

cdef class ControlByType:
    def __cinit__(self, np.int32_t nCells, np.ndarray sprayProp):
        self.nCells = nCells
        self._spray = np.ones(nCells, dtype=np.float64)
        self._sprayBG = np.ones(nCells, dtype=np.float64)
        self._sprayDem = np.ones(nCells, dtype=np.float64)
        self._sprayReg = np.ones(nCells, dtype=np.float64)
        self._spray_prop = sprayProp
        self.spray = self._spray
        self.sprayBG = self._sprayBG
        self.sprayDem = self._sprayDem
        self.sprayReg = self._sprayReg
        self.spray_prop = self._spray_prop
        self._bctrl = np.ones(nCells, dtype=np.float64)
        self._inf_res = np.ones(nCells, dtype=np.float64)                        # infectiousness (based on resistance, if any)
        self._susc_res = np.ones(nCells, dtype=np.float64)                       # susceptibility (based on resistance, if any)
        self.bctrl = self._bctrl
        self.inf_res = self._inf_res
        self.susc_res = self._susc_res

cdef class HostClass:

    def __cinit__(self, np.int32_t nCells, np.ndarray tot, np.float64_t pE, np.float64_t pCI):

        cdef np.ndarray _status_host = np.full(nCells, State.empty, dtype=np.uint8)
        _status_host[tot > 0] = State.susceptible

        # Parameters
        self.pE = pE
        self.pCI = pCI

        # Status of cell (host units)
        self.tot = tot
        self.sus = tot.copy()
        self.exp = np.zeros(nCells, dtype=np.uint16)
        self.crp = np.zeros(nCells, dtype=np.uint16)
        self.inf = np.zeros(nCells, dtype=np.uint16)
        self.rem = np.zeros(nCells, dtype=np.uint16)
        self.status_host = _status_host
        self.time_host = np.full((4, nCells), np.nan, dtype=np.float64)         # store time host became exposed, detectable, cryptic, symptomatic, removed

        self.pInf = np.zeros(nCells, dtype=np.float64)

    cdef inline void update_pInf(self, np.int32_t idx):
        cdef np.float64_t numer, denom = self.tot[idx] - self.rem[idx]
        if denom > 0:
            numer = self.pCI * (self.inf[idx] + self.crp[idx]) + self.pE * (self.exp[idx])
            self.pInf[idx] = numer / denom
        else:
            self.pInf[idx] = 0.0

    cdef inline void update_SE(self, np.int32_t idx, np.int16_t val, np.float64_t time, RateClass rates, CellSummary summary):
        cdef np.float64_t susc_mult

        # Update status of host units
        self.sus[idx] -= val
        self.exp[idx] += val
        
        if self.status_host[idx] == State.susceptible:
            self.status_host[idx] = State.exposed
            self.time_host[0, idx] = time
            summary.infected[idx] += 1
            summary.totInfected += 1

        summary.S_tot -= val
        if summary.S_tot == 0:
            summary.all_infected = 1

        susc_mult = self.sus[idx] / <np.float64_t>(self.sus[idx] + val)

        # Update rates (only simple ones)
        rates.latR.submitRate(idx, self.exp[idx] * rates.rEC)
        rates.pSdR.submitRate(idx, rates.pSdR.rates[idx] * susc_mult)

        # Update pInf
        self.update_pInf(idx)

    cdef inline void update_EC(self, np.int32_t idx, np.int16_t val, np.float64_t time, RateClass rates):
        # Update status of host units
        self.exp[idx] -= val
        self.crp[idx] += val
        if self.status_host[idx] == State.exposed:
            self.status_host[idx] = State.cryptic
            self.time_host[1, idx] = time

        # Update rates (only simple ones)
        rates.latR.submitRate(idx, self.exp[idx] * rates.rEC)
        rates.symR.submitRate(idx, self.crp[idx] * rates.rCI)

        # Update pInf
        self.update_pInf(idx)

    cdef inline void update_CI(self, np.int32_t idx, np.int16_t val, np.float64_t time, RateClass rates):
        # Update status of host units
        self.crp[idx] -= val
        self.inf[idx] += val
        if self.status_host[idx] == State.cryptic:
            self.status_host[idx] = State.infected
            self.time_host[2, idx] = time

        # Update rates (only simple ones)
        rates.symR.submitRate(idx, self.crp[idx] * rates.rCI)

    cpdef tuple update_removal(self, np.int32_t idx, np.int16_t S_rem, np.int16_t E_rem, np.int16_t C_rem, np.int16_t I_rem, np.float64_t time, ControlByType ctrl, VectorClass vector, RateClass rates, CellSummary summary, np.uint8_t[::1] rates_flags):
        cdef np.int8_t update_pMd = 0, update_vMd = 0
        # Update status of host units and rates (only simple ones)
        if S_rem > 0:
            self.sus[idx] -= S_rem
            summary.S_tot -= S_rem
            summary.S_R_tot += S_rem
            if self.sus[idx] == 0:
                update_pMd = 1
            if summary.S_tot == 0:
                summary.all_infected = 1
        if E_rem > 0:
            self.exp[idx] -= E_rem
            rates.latR.submitRate_noTotal(idx, self.exp[idx] * rates.rEC)
            rates_flags[<int>RatesIdx.lat] = 1
        if C_rem > 0:
            self.crp[idx] -= C_rem
            rates.symR.submitRate_noTotal(idx, self.crp[idx] * rates.rCI)
            rates_flags[<int>RatesIdx.sym] = 1

        
        self.inf[idx] -= I_rem
        self.rem[idx] += S_rem + E_rem + C_rem + I_rem 
        summary.n_stand[idx] = self.tot[idx] - self.rem[idx]

        self.update_pInf(idx)
        
        # Update vector density, if any
        if vector.dens_vec[idx] > 0:
            vector.dens_vec[idx] = vector.cc_vec[idx] * ctrl.spray[idx] * (1 - <np.float64_t>(self.rem[idx])/ <np.float64_t>(self.tot[idx]))

        if self.inf[idx] + self.crp[idx] + self.exp[idx] == 0:
            if summary.infected[idx] == 1:
                summary.infected[idx] = 0
                summary.totInfected -= 1
            
            # Note: not setting state back to susceptible if cells is only susceptible because it would mess up with the timings (can get exposed again, resetting the exposition time)
            if self.sus[idx] == 0:
                self.status_host[idx] = State.removed # only when all removed
                self.time_host[3, idx] = time

                summary.standing[idx] = 0
                summary.totStanding -= 1

                if vector.status_vec[idx] == State.absent:
                    rates.vSdR.submitRate_noTotal(idx, 0.0) 
                    rates_flags[<int>RatesIdx.vSd] = 1
                    update_vMd = 1
                    
                else:
                    summary.totInfested -= 1
                    summary.infested[idx] = 0
                    if vector.status_vec[idx] == State.present:
                        rates.estR.submitRate_noTotal(idx, 0.0)
                        rates_flags[<int>RatesIdx.est] = 1

                vector.status_vec[idx] = State.removed
                vector.time_vec[2, idx] = time
                        
                if summary.totInfested == summary.totStanding:
                    summary.vectorFull = 1

        return update_pMd, update_vMd, rates_flags

cdef class RateClass:

    def __cinit__(self, np.int32_t nCells, np.float64_t rEC, np.float64_t rCI, np.float64_t rInf, np.float64_t rEst, np.float64_t rLdd, np.float64_t rMdd):

        # Parameters
        self.rEC = rEC
        self.rCI = rCI
        self.rInf = rInf
        self.rEst = rEst
        self.rLdd = rLdd
        self.rMdd = rMdd
        self.rSdd = rMdd    # same as Mdd because it's the same dispersal, split in two

        # # Find number of blocks and limits
        # cdef np.int32_t nBlocks = <np.int32_t>(c_math.sqrt(nCells))
        # cdef np.int32_t min_block_size = (nCells + nBlocks - 1) // nBlocks # ceiling division
        # cdef np.int32_t start, end
        # cdef np.int32_t[::1] blocks_limits = np.zeros(nBlocks + 1, dtype = np.int32)
        
        # start = 0
        # for b in range(nBlocks + 1):
        #     end = start + min_block_size
        #     if end > nCells:
        #         end = nCells
        #     blocks_limits[b] = start
        #     start = end

        # Rates affecting cells
        self.latR = SegTree(nCells)
        self.symR = SegTree(nCells)
        self.estR = SegTree(nCells)
        self.vSdR = SegTree(nCells)
        # self.vSdR = SqrtBlocks(nCells, nBlocks, blocks_limits)
        self.vMdR = SegTree(nCells)
        # self.pSdR = SqrtBlocks_with_base(nCells, nBlocks, blocks_limits)
        self.pSdR = SegTree_with_base(nCells)
        self.pMdR = SegTree_with_base(nCells)
        self.lddR = SegTree_with_base(nCells)

    cdef inline np.float64_t get_total_rate(self):
        return self.latR.total_rate + self.symR.total_rate + self.estR.total_rate + self.vSdR.total_rate + self.pSdR.total_rate + self.vMdR.total_rate + self.pMdR.total_rate + self.lddR.total_rate

cdef class VectorClass:

    def __cinit__(self, np.int32_t nCells, np.ndarray cc):

        cdef np.ndarray _status_vec = np.full(nCells, State.empty, dtype=np.uint8)
        _status_vec[cc > 0] = State.absent
        
        # Status of cell (vector)
        self.status_vec = _status_vec
        self.time_vec = np.full((3, nCells), np.nan, dtype=np.float64)          # store time vector became present, colonised, removed
        self._dens_vec = np.zeros(nCells, dtype=np.float64)
        self._cc_vec = cc.copy()
        self.dens_vec = self._dens_vec
        self.cc_vec = self._cc_vec

    cdef inline void update_AR(self, np.int32_t idx, np.float64_t time, RateClass rates, CellSummary summary):
        # Update status of host units
        self.status_vec[idx] = State.present
        self.time_vec[0, idx] = time
        summary.totInfested += 1
        summary.infested[idx] = 1

        if summary.totInfested == summary.totStanding:
            summary.vectorFull = 1
            # self.vSdR.zeroRates()

        # Update rates (only simple ones)
        rates.estR.submitRate(idx, rates.rEst)
        rates.vSdR.submitRate(idx, 0.0)

    cdef inline void update_ES(self, np.int32_t idx, np.float64_t time, ControlByType ctrl, HostClass host, RateClass rates, CellSummary summary):
        # Update status of host units
        self.status_vec[idx] = State.colonised
        self.time_vec[1, idx] = time
        self.dens_vec[idx] = self.cc_vec[idx] * ctrl.spray[idx] * (1.0 - <np.float64_t>(host.rem[idx])/<np.float64_t>(host.tot[idx]))
        # Update rates (only simple ones)
        rates.estR.submitRate(idx, 0.0)
        # vMd and pMd updated outside because not necessary when no susceptible cells around

cdef class CellSummary:

    def __cinit__(self, np.int32_t nCells, np.ndarray tot):        
        
        self.maxCit = tot.max()                                                 # max amount of cit in cells
        self.citrus_total = tot.sum()
        self.S_tot = self.citrus_total
        self.S_R_tot = 0
        self.cit_cells = (tot > 0).sum()                                        # total amount of cells with citrus of cit_type
        self._infected = np.zeros(nCells, dtype=np.uint8)                       # keep track of infected cells (at least one infection) - binary
        self._infested = np.zeros(nCells, dtype=np.uint8)                       # keep track of infested cells - binary
        self._standing = (tot > 0).astype(np.uint8)                             # keep track of cells with standing citrus - binary
        self._n_stand = tot.copy()                                              # keep track of number of standing units in each cell
        self.infected = self._infected
        self.infested = self._infested
        self.standing = self._standing
        self.n_stand = self._n_stand
        self.totStanding = self.cit_cells
        self.totInfected = 0                                                    # keep track of total infected cells
        self.totInfested = 0                                                    # keep track of total infested cells
        self.vectorFull = 0                                                     # are all cells infested?
        self.all_infected = 0                                                   # are all cells infected?

cdef class CellsByType:

    def __cinit__(self, np.int32_t nCells, np.ndarray tot, np.ndarray cc, np.ndarray propSpray, np.float64_t pE, np.float64_t pCI, np.float64_t rEC, np.float64_t rCI, np.float64_t rInf, np.float64_t rEst, np.float64_t rLdd, np.float64_t rMdd):

        self.nCells = nCells
        self.ctrl = ControlByType(nCells, propSpray)
        self.host = HostClass(nCells, tot, pE, pCI)
        self.vector = VectorClass(nCells, cc)
        self.rates = RateClass(nCells, rEC, rCI, rInf, rEst, rLdd, rMdd)
        self.summary = CellSummary(nCells, tot)

    cpdef void reset(self, np.ndarray cc):
        cdef np.int32_t i
        cdef Py_ssize_t j

        self.vector._cc_vec = cc.copy()

        self.ctrl._spray.fill(1.0)
        self.ctrl._sprayBG.fill(1.0)
        self.ctrl._sprayDem.fill(1.0)
        self.ctrl._sprayReg.fill(1.0)
        self.ctrl._bctrl.fill(1.0)
        self.ctrl._inf_res.fill(1.0)
        self.ctrl._susc_res.fill(1.0)

        # Individual cells
        for i in range(self.nCells):
            self.host.sus[i] = self.host.tot[i]
            self.summary.n_stand[i] = self.host.tot[i]
            self.host.exp[i] = 0
            self.host.crp[i] = 0
            self.host.inf[i] = 0
            self.host.rem[i] = 0
            self.host.status_host[i] = State.susceptible
            self.vector.status_vec[i] = State.absent
            self.vector.dens_vec[i] = 0.0
            self.host.pInf[i] = 0.0
            for j in range(self.host.time_host.shape[0]):
                self.host.time_host[j,i] = np.nan
            for j in range(self.vector.time_vec.shape[0]):
                self.vector.time_vec[j,i] = np.nan

            if self.host.tot[i] == 0:
                self.summary.standing[i] = 0
                self.host.status_host[i] = State.empty
                self.vector.status_vec[i] = State.empty
            else:
                self.summary.standing[i] = 1
        # Rates
        self.rates.latR.zeroRates()
        self.rates.symR.zeroRates()
        self.rates.estR.zeroRates()
        self.rates.vSdR.zeroRates()
        self.rates.pSdR.zeroRates()
        self.rates.vMdR.zeroRates()
        self.rates.pMdR.zeroRates()
        self.rates.lddR.zeroRates()
        # General
        self.summary._infected[:] = 0
        self.summary._infested[:] = 0
        self.summary.all_infected = 0
        self.summary.totInfested = 0
        self.summary.totInfected = 0
        self.summary.totStanding = self.summary.cit_cells
        self.summary.vectorFull = 0
        self.summary.S_tot = self.summary.citrus_total
        self.summary.S_R_tot = 0

    cpdef void update_pInf(self, np.int32_t idx):
        self.host.update_pInf(idx)

    cpdef void update_SE(self, np.int32_t idx, np.int16_t val, np.float64_t time):
        self.host.update_SE(idx, val, time, self.rates, self.summary)

    cpdef void update_EC(self, np.int32_t idx, np.int16_t val, np.float64_t time):
        self.host.update_EC(idx, val, time, self.rates)

    cpdef void update_CI(self, np.int32_t idx, np.int16_t val, np.float64_t time):
        self.host.update_CI(idx, val, time, self.rates)

    cpdef tuple update_removal(self, np.int32_t idx, np.int16_t S_rem, np.int16_t E_rem, np.int16_t C_rem, np.int16_t I_rem, np.float64_t time, np.uint8_t[::1] rates_flags):
        return self.host.update_removal(idx, S_rem, E_rem, C_rem, I_rem, time, self.ctrl, self.vector, self.rates, self.summary, rates_flags)

    cpdef void update_AR(self, np.int32_t idx, np.float64_t time):
        self.vector.update_AR(idx, time, self.rates, self.summary)

    cpdef void update_ES(self, np.int32_t idx, np.float64_t time):
        self.vector.update_ES(idx, time, self.ctrl, self.host, self.rates, self.summary)

cdef class SurveyHelperClass:

    def __cinit__(self, np.int32_t nCells):
        self.nCells = nCells

        self._cells_search = np.zeros(nCells, dtype=np.uint8)
        self.cells_search = self._cells_search
        self._cells_surveyed = np.zeros(nCells, dtype=np.uint8)

        # Vector (initially)
        self._cell_success = np.zeros(nCells, dtype=np.uint8)
        self.cell_success = self._cell_success

        # Host surveys
        self._comDetected = np.zeros((2, nCells), dtype=np.uint16)
        self.comDetected = self._comDetected
        self._rsdDetected = np.zeros((2, nCells), dtype=np.uint16)
        self.rsdDetected = self._rsdDetected
        
        self._cell_success_host = np.zeros(nCells, dtype=np.uint8)
        self.cell_success_host = self._cell_success_host

        # Removal
        self._add_demarcated = np.zeros(nCells, dtype=np.uint8)
        self.add_demarcated = self._add_demarcated

        # Save stats
        self.n_vectors = 0
        self.n_vis = 0
        self.n_pcr = 0
        self.n_sus_rem = 0

    cpdef reset(self):
        self._cells_search.fill(0)
        self._cells_surveyed.fill(0)
        self._cell_success.fill(0)
        self._cell_success_host.fill(0)
        self._comDetected.fill(0)
        self._rsdDetected.fill(0)
        self._add_demarcated.fill(0)

    cpdef reset_full(self):
        self.reset()
        self.n_vectors = 0
        self.n_vis = 0
        self.n_pcr = 0
        self.n_sus_rem = 0

cdef class SaveStructure:

    def __cinit__(self, np.int32_t nCells, np.int16_t nSpMaps, np.int16_t nSpTraj, np.int16_t nSpMetrics, np.int16_t nRuns):
        
        """ Saving info for maps. We are interested in total amounts, regardless of citrus type. 
        - Total susceptibles
        - Total infected (E + C + I)
        - Total removed 
        NoDataValue = 9999 """
        self._maps_S = np.full((nSpMaps, nCells), 9999, dtype = np.uint16)
        self._maps_I = np.full((nSpMaps, nCells), 9999, dtype = np.uint16)
        self._maps_R = np.full((nSpMaps, nCells), 9999, dtype = np.uint16)
        self._zones = np.zeros((nSpMaps, nCells), dtype = np.uint8)
        self.maps_S = self._maps_S
        self.maps_I = self._maps_I
        self.maps_R = self._maps_R
        self.zones = self._zones

        """ Saving info for ensemble trajectories over [nRunsTraj] runs. We are interested in total amounts, regardless of citrus type. 
        - Total pathogen-affected units (E + C + I + R)
        - Total pathogen-affected cells (E + C + I + R)
        - Total vector-infested cells
        NoDataValue = 999999999 """
        self._traj_P_units = np.full((nRuns, nSpTraj), -1, dtype = np.float32)
        self._traj_P_cells = np.full((nRuns, nSpTraj), -1, dtype = np.float32)
        self._traj_V_cells = np.full((nRuns, nSpTraj), -1, dtype = np.float32)
        self.traj_P_units = self._traj_P_units
        self.traj_P_cells = self._traj_P_cells
        self.traj_V_cells = self._traj_V_cells

        self._traj_IC = np.zeros((nRuns, nSpTraj), dtype = np.uint32)
        self.traj_IC = self._traj_IC

        """ Saving yield proxy for ensemble trajectories over [nRunsTraj] runs. We are interested only in commercial citrus.
        NoDataValue = 999999999 """

        # In compliant cells
        self._yield_proxy_s_c1 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self._yield_proxy_e_c1 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self._yield_proxy_c_c1 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self._yield_proxy_i_c1 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)

        self.yield_proxy_s_c1 = self._yield_proxy_s_c1
        self.yield_proxy_e_c1 = self._yield_proxy_e_c1
        self.yield_proxy_c_c1 = self._yield_proxy_c_c1
        self.yield_proxy_i_c1 = self._yield_proxy_i_c1

        self._yield_proxy_s_c0 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self._yield_proxy_e_c0 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self._yield_proxy_c_c0 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self._yield_proxy_i_c0 = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)

        self.yield_proxy_s_c0 = self._yield_proxy_s_c0
        self.yield_proxy_e_c0 = self._yield_proxy_e_c0
        self.yield_proxy_c_c0 = self._yield_proxy_c_c0
        self.yield_proxy_i_c0 = self._yield_proxy_i_c0

        self._tot_c0 = np.zeros(nRuns, dtype = np.uint32)
        self._tot_c1 = np.zeros(nRuns, dtype = np.uint32)
        self.tot_c0 = self._tot_c0
        self.tot_c1 = self._tot_c1

        self._infections_all = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self.infections_all = self._infections_all
        self._infections_rsd = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self.infections_rsd = self._infections_rsd
        self._R_units = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self.R_units = self._R_units
        self._RS_units = np.zeros((nRuns, nSpMetrics), dtype = np.uint32)
        self.RS_units = self._RS_units

        """ Cell status """
        self.n_e_cells = np.zeros(nRuns, dtype = np.uint32)
        self.n_c_cells = np.zeros(nRuns, dtype = np.uint32)
        self.n_i_cells = np.zeros(nRuns, dtype = np.uint32)
        

    cdef update_maps(self, Py_ssize_t sp_last, Py_ssize_t sp_current, HostClass com_cit, HostClass rsd_cit, Management mng):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int16_t sus, inf, rem

        for sp in range(sp_last + 1, sp_current):
            self.maps_S[sp, :] = self.maps_S[sp_last, :]
            self.maps_I[sp, :] = self.maps_I[sp_last, :]
            self.maps_R[sp, :] = self.maps_R[sp_last, :]
            self.zones[sp, :] = self.zones[sp_last, :]

        for i in range(com_cit.sus.shape[0]):
            sus = com_cit.sus[i] + rsd_cit.sus[i]
            inf = com_cit.exp[i] + rsd_cit.exp[i] + com_cit.crp[i] + rsd_cit.crp[i] + com_cit.inf[i] + rsd_cit.inf[i]
            rem = com_cit.rem[i] + rsd_cit.rem[i]

            self.maps_S[sp_current, i] = sus
            self.maps_I[sp_current, i] = inf
            self.maps_R[sp_current, i] = rem
        
        self._zones[sp_current, :] += mng._buf * 1
        self._zones[sp_current, :] += mng._inf * 2

    cdef update_trajectories(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int16_t inf_add
        cdef np.uint32_t inf_units = 0, inf_cells = 0, vec_cells = 0, rem_units = 0, ic_units = 0

        for i in range(com.host.sus.shape[0]):
            inf_add = com.host.exp[i] + rsd.host.exp[i] + com.host.crp[i] + rsd.host.crp[i] + com.host.inf[i] + rsd.host.inf[i] + com.host.rem[i] + rsd.host.rem[i]
            ic_units += com.host.crp[i] + rsd.host.crp[i] + com.host.inf[i] + rsd.host.inf[i]
            rem_units += com.host.rem[i] + rsd.host.rem[i]
            inf_units += inf_add
            inf_cells += (inf_add > 0)
            # vec_cells += (com.vector.status_vec[i] == State.colonised) or (rsd.vector.status_vec[i] == State.colonised)
            vec_cells += (com.vector.status_vec[i] == State.present or com.vector.status_vec[i] == State.colonised) or (rsd.vector.status_vec[i] == State.present or rsd.vector.status_vec[i] == State.colonised)

        for sp in range(sp_last + 1, sp_current):
            self.traj_P_units[run, sp] = self.traj_P_units[run, sp_last]
            self.traj_P_cells[run, sp] = self.traj_P_cells[run, sp_last]
            self.traj_V_cells[run, sp] = self.traj_V_cells[run, sp_last]
            self.traj_IC[run, sp] = self.traj_IC[run, sp_last]

        self.traj_P_units[run, sp_current] = np.float32(<np.float64_t>(inf_units) / <np.float64_t>(com.summary.citrus_total + rsd.summary.citrus_total))
        self.traj_P_cells[run, sp_current] = np.float32(<np.float64_t>(inf_cells) / <np.float64_t>(com.host.sus.shape[0]))
        self.traj_V_cells[run, sp_current] = np.float32(<np.float64_t>(vec_cells) / <np.float64_t>(com.host.sus.shape[0]))
        self.traj_IC[run, sp_current] = ic_units

    cdef update_cali_trajectories(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int16_t inf_add
        cdef np.uint32_t inf_units = 0, inf_cells = 0, vec_cells = 0

        for i in range(com.host.sus.shape[0]):
            inf_add = com.host.exp[i] + rsd.host.exp[i] + com.host.crp[i] + rsd.host.crp[i] + com.host.inf[i] + rsd.host.inf[i] + com.host.rem[i] + rsd.host.rem[i]
            inf_units += inf_add
            inf_cells += (inf_add > 0)
            vec_cells += (com.vector.status_vec[i] == State.colonised) or (rsd.vector.status_vec[i] == State.colonised)

        for sp in range(sp_last + 1, sp_current):
            self.traj_P_units[run, sp] = self.traj_P_units[run, sp_last]
            self.traj_P_cells[run, sp] = self.traj_P_cells[run, sp_last]
            self.traj_V_cells[run, sp] = self.traj_V_cells[run, sp_last]

        self.traj_P_units[run, sp_current] = np.float32(<np.float64_t>(inf_units) / <np.float64_t>(com.summary.citrus_total + rsd.summary.citrus_total))
        self.traj_P_cells[run, sp_current] = np.float32(<np.float64_t>(inf_cells) / <np.float64_t>(com.host.sus.shape[0]))
        self.traj_V_cells[run, sp_current] = np.float32(<np.float64_t>(vec_cells) / <np.float64_t>(com.host.sus.shape[0]))


    cdef update_metrics(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd, Management mng):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int32_t s_c1 = 0, e_c1 = 0, c_c1 = 0, i_c1 = 0, s_c0 = 0, e_c0 = 0, c_c0 = 0, i_c0 = 0
        cdef np.int32_t inf_rsd = 0, inf_com = 0, rem_tot = 0, rem_sus = 0

        for i in range(com.host.sus.shape[0]):
            if mng.compliance[i]:
                s_c1 += com.host.sus[i]
                e_c1 += com.host.exp[i]
                c_c1 += com.host.crp[i]
                i_c1 += com.host.inf[i]
            else:
                s_c0 += com.host.sus[i]
                e_c0 += com.host.exp[i]
                c_c0 += com.host.crp[i]
                i_c0 += com.host.inf[i]
            inf_rsd += rsd.host.exp[i] + rsd.host.crp[i] + rsd.host.inf[i]
            inf_com += com.host.exp[i] + com.host.crp[i] + com.host.inf[i]
            rem_tot += com.host.rem[i] + rsd.host.rem[i]
        rem_sus = com.summary.S_R_tot + rsd.summary.S_R_tot

        for sp in range(sp_last, sp_current - 1):
            self.yield_proxy_s_c0[run, sp] = self.yield_proxy_s_c0[run, sp_last]
            self.yield_proxy_e_c0[run, sp] = self.yield_proxy_e_c0[run, sp_last]
            self.yield_proxy_c_c0[run, sp] = self.yield_proxy_c_c0[run, sp_last]
            self.yield_proxy_i_c0[run, sp] = self.yield_proxy_i_c0[run, sp_last]

            self.yield_proxy_s_c1[run, sp] = self.yield_proxy_s_c1[run, sp_last]
            self.yield_proxy_e_c1[run, sp] = self.yield_proxy_e_c1[run, sp_last]
            self.yield_proxy_c_c1[run, sp] = self.yield_proxy_c_c1[run, sp_last]
            self.yield_proxy_i_c1[run, sp] = self.yield_proxy_i_c1[run, sp_last]

            self.infections_all[run, sp] = self.infections_all[run, sp_last]
            self.infections_rsd[run, sp] = self.infections_rsd[run, sp_last]
            self.R_units[run, sp] = self.R_units[run, sp_last]
            self.RS_units[run, sp] = self.RS_units[run, sp_last]

        self.yield_proxy_s_c0[run, sp_current - 1] = s_c0
        self.yield_proxy_e_c0[run, sp_current - 1] = e_c0
        self.yield_proxy_c_c0[run, sp_current - 1] = c_c0
        self.yield_proxy_i_c0[run, sp_current - 1] = i_c0

        self.yield_proxy_s_c1[run, sp_current - 1] = s_c1
        self.yield_proxy_e_c1[run, sp_current - 1] = e_c1
        self.yield_proxy_c_c1[run, sp_current - 1] = c_c1
        self.yield_proxy_i_c1[run, sp_current - 1] = i_c1

        self.infections_all[run, sp_current - 1] = inf_com + inf_rsd
        self.infections_rsd[run, sp_current - 1] = inf_rsd
        self.R_units[run, sp_current - 1] = rem_tot
        self.RS_units[run, sp_current - 1] = rem_sus

    cdef save_cell_status(self, np.int16_t run, CellsByType com, CellsByType rsd):
        cdef Py_ssize_t i
        cdef np.int32_t inf = 0, crp = 0, exp = 0

        for i in range(com.host.sus.shape[0]):
            inf = com.host.inf[i] + rsd.host.inf[i]
            crp = com.host.crp[i] + rsd.host.crp[i]
            exp = com.host.exp[i] + rsd.host.exp[i]

            self.n_e_cells[run] += (exp > 0) * (crp == 0) * (inf == 0)
            self.n_c_cells[run] += (crp > 0) * (inf == 0)
            self.n_i_cells[run] += (inf > 0)


    cdef update_maps_end(self, Py_ssize_t sp_last, Py_ssize_t sp_current, HostClass com_cit, HostClass rsd_cit, Management mng):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int16_t sus, inf, rem

        for sp in range(sp_last + 1, sp_current):
            self.maps_S[sp, :] = self.maps_S[sp_last, :]
            self.maps_I[sp, :] = self.maps_I[sp_last, :]
            self.maps_R[sp, :] = self.maps_R[sp_last, :]
            self.zones[sp, :] = self.zones[sp_last, :]

        for i in range(com_cit.sus.shape[0]):
            sus = com_cit.sus[i] + rsd_cit.sus[i]
            inf = com_cit.exp[i] + rsd_cit.exp[i] + com_cit.crp[i] + rsd_cit.crp[i] + com_cit.inf[i] + rsd_cit.inf[i]
            rem = com_cit.rem[i] + rsd_cit.rem[i]

            for sp in range(sp_current, self.maps_S.shape[0]):
                self.maps_S[sp, i] = sus
                self.maps_I[sp, i] = inf
                self.maps_R[sp, i] = rem

        self._zones[sp_current:, :] += mng._buf * 1
        self._zones[sp_current:, :] += mng._inf * 2

    cdef update_trajectories_end(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int16_t inf_add
        cdef np.uint32_t inf_units = 0, inf_cells = 0, vec_cells = 0, rem_units = 0, ic_units = 0

        for i in range(com.host.sus.shape[0]):
            inf_add = com.host.exp[i] + rsd.host.exp[i] + com.host.crp[i] + rsd.host.crp[i] + com.host.inf[i] + rsd.host.inf[i] + com.host.rem[i] + rsd.host.rem[i]
            ic_units += com.host.crp[i] + rsd.host.crp[i] + com.host.inf[i] + rsd.host.inf[i]
            rem_units += com.host.rem[i] + rsd.host.rem[i]
            inf_units += inf_add
            inf_cells += (inf_add > 0)
            # vec_cells += (com.vector.status_vec[i] == State.colonised) or (rsd.vector.status_vec[i] == State.colonised)
            vec_cells += (com.vector.status_vec[i] == State.present or com.vector.status_vec[i] == State.colonised) or (rsd.vector.status_vec[i] == State.present or rsd.vector.status_vec[i] == State.colonised)

        for sp in range(sp_last + 1, sp_current):
            self.traj_P_units[run, sp] = self.traj_P_units[run, sp_last]
            self.traj_P_cells[run, sp] = self.traj_P_cells[run, sp_last]
            self.traj_V_cells[run, sp] = self.traj_V_cells[run, sp_last]
            self.traj_IC[run, sp] = self.traj_IC[run, sp_last]

        for sp in range(sp_current, self.traj_P_units.shape[1]):
            self.traj_P_units[run, sp] = np.float32(<np.float64_t>(inf_units) / <np.float64_t>(com.summary.citrus_total + rsd.summary.citrus_total))
            self.traj_P_cells[run, sp] = np.float32(<np.float64_t>(inf_cells) / <np.float64_t>(com.host.sus.shape[0]))
            self.traj_V_cells[run, sp] = np.float32(<np.float64_t>(vec_cells) / <np.float64_t>(com.host.sus.shape[0]))
            self.traj_IC[run, sp] = ic_units

    cdef update_cali_trajectories_end(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int16_t inf_add
        cdef np.uint32_t inf_units = 0, inf_cells = 0, vec_cells = 0, rem_units = 0, ic_units = 0

        for i in range(com.host.sus.shape[0]):
            inf_add = com.host.exp[i] + rsd.host.exp[i] + com.host.crp[i] + rsd.host.crp[i] + com.host.inf[i] + rsd.host.inf[i] + com.host.rem[i] + rsd.host.rem[i]
            inf_units += inf_add
            inf_cells += (inf_add > 0)
            vec_cells += (com.vector.status_vec[i] == State.colonised) or (rsd.vector.status_vec[i] == State.colonised)

        for sp in range(sp_last + 1, sp_current):
            self.traj_P_units[run, sp] = self.traj_P_units[run, sp_last]
            self.traj_P_cells[run, sp] = self.traj_P_cells[run, sp_last]
            self.traj_V_cells[run, sp] = self.traj_V_cells[run, sp_last]

        for sp in range(sp_current, self.traj_P_units.shape[1]):
            self.traj_P_units[run, sp] = np.float32(<np.float64_t>(inf_units) / <np.float64_t>(com.summary.citrus_total + rsd.summary.citrus_total))
            self.traj_P_cells[run, sp] = np.float32(<np.float64_t>(inf_cells) / <np.float64_t>(com.host.sus.shape[0]))
            self.traj_V_cells[run, sp] = np.float32(<np.float64_t>(vec_cells) / <np.float64_t>(com.host.sus.shape[0]))

    cdef update_metrics_end(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd, Management mng):
        cdef Py_ssize_t i
        cdef Py_ssize_t sp
        cdef np.int32_t s_c1 = 0, e_c1 = 0, c_c1 = 0, i_c1 = 0, s_c0 = 0, e_c0 = 0, c_c0 = 0, i_c0 = 0
        cdef np.int32_t inf_rsd = 0, inf_com = 0, rem_tot = 0, rem_sus = 0

        for i in range(com.host.sus.shape[0]):
            if mng.compliance[i]:
                s_c1 += com.host.sus[i]
                e_c1 += com.host.exp[i]
                c_c1 += com.host.crp[i]
                i_c1 += com.host.inf[i]
                self.tot_c1[run] += com.host.tot[i]
            else:
                s_c0 += com.host.sus[i]
                e_c0 += com.host.exp[i]
                c_c0 += com.host.crp[i]
                i_c0 += com.host.inf[i]
                self.tot_c0[run] += com.host.tot[i]

            inf_rsd += rsd.host.exp[i] + rsd.host.crp[i] + rsd.host.inf[i]
            inf_com += com.host.exp[i] + com.host.crp[i] + com.host.inf[i]
            rem_tot += com.host.rem[i] + rsd.host.rem[i]
        rem_sus = com.summary.S_R_tot + rsd.summary.S_R_tot

        for sp in range(sp_last, sp_current - 1):
            self.yield_proxy_s_c0[run, sp] = self.yield_proxy_s_c0[run, sp_last]
            self.yield_proxy_e_c0[run, sp] = self.yield_proxy_e_c0[run, sp_last]
            self.yield_proxy_c_c0[run, sp] = self.yield_proxy_c_c0[run, sp_last]
            self.yield_proxy_i_c0[run, sp] = self.yield_proxy_i_c0[run, sp_last]

            self.yield_proxy_s_c1[run, sp] = self.yield_proxy_s_c1[run, sp_last]
            self.yield_proxy_e_c1[run, sp] = self.yield_proxy_e_c1[run, sp_last]
            self.yield_proxy_c_c1[run, sp] = self.yield_proxy_c_c1[run, sp_last]
            self.yield_proxy_i_c1[run, sp] = self.yield_proxy_i_c1[run, sp_last]

            self.infections_all[run, sp] = self.infections_all[run, sp_last]
            self.infections_rsd[run, sp] = self.infections_rsd[run, sp_last]
            self.R_units[run, sp] = self.R_units[run, sp_last]
            self.RS_units[run, sp] = self.RS_units[run, sp_last]

        for sp in range(sp_current - 1, self.yield_proxy_s_c0.shape[1]):
            self.yield_proxy_s_c0[run, sp] = s_c0
            self.yield_proxy_e_c0[run, sp] = e_c0
            self.yield_proxy_c_c0[run, sp] = c_c0
            self.yield_proxy_i_c0[run, sp] = i_c0

            self.yield_proxy_s_c1[run, sp] = s_c1
            self.yield_proxy_e_c1[run, sp] = e_c1
            self.yield_proxy_c_c1[run, sp] = c_c1
            self.yield_proxy_i_c1[run, sp] = i_c1

            self.infections_all[run, sp] = inf_com + inf_rsd
            self.infections_rsd[run, sp] = inf_rsd
            self.R_units[run, sp] = rem_tot
            self.RS_units[run, sp] = rem_sus

cdef class SaveMetrics:

    def __cinit__(self, np.int16_t nRuns):
        
        self._S_removed = np.zeros(nRuns, dtype=np.uint32)
        self.S_removed = self._S_removed
        self._tot_removed = np.zeros(nRuns, dtype=np.uint32)
        self.tot_removed = self._tot_removed
        self._cit_inc = np.zeros(nRuns, dtype=np.float64)
        self.cit_inc = self._cit_inc
        self._hlb_inc = np.zeros(nRuns, dtype=np.float64)
        self.hlb_inc = self._hlb_inc
        self._cell_inc = np.zeros(nRuns, dtype=np.float64)
        self.cell_inc = self._cell_inc
        self._detection_results = np.zeros((nRuns, 13), dtype=np.float64)
        self.detection_results = self._detection_results
        self._n_samples = np.zeros((nRuns, 3), dtype=np.uint64)
        self.n_samples = self._n_samples

        self._reason = np.zeros(nRuns, dtype=np.int8)
        self.reason = self._reason
        self._time_end = np.zeros(nRuns, dtype=np.float64)
        self.time_end = self._time_end

    cdef update_run(self, np.int16_t run, CellsByType com, CellsByType rsd, SurveyHelperClass shc, np.float64_t[::1] detection, np.int8_t reason, np.float64_t t_end):
        cdef Py_ssize_t i, j
        cdef np.int32_t tot_cit = 0
        cdef np.int32_t inf_cell, inf_cit = 0
        cdef np.int32_t rem_cell, rem_cit = 0
        cdef np.int32_t cells_inf = 0

        self.detection_results[run, :] = detection[:]
        self.S_removed[run] = com.summary.S_R_tot + rsd.summary.S_R_tot

        for i in range(com.host.sus.shape[0]):
            rem_cell = com.host.rem[i] + rsd.host.rem[i]
            inf_cell = com.host.inf[i] + rsd.host.inf[i] + com.host.crp[i] + rsd.host.crp[i] + com.host.exp[i] + rsd.host.exp[i]
            rem_cit += rem_cell
            inf_cit += inf_cell
            tot_cit += com.host.tot[i] + rsd.host.tot[i]
            cells_inf += (rem_cell + inf_cell > 0)

        self.tot_removed[run] = rem_cit
        self.cit_inc[run] = <np.float64_t>(inf_cit + rem_cit) / <np.float64_t>(tot_cit)
        self.hlb_inc[run] = <np.float64_t>(inf_cit) / <np.float64_t>(tot_cit)
        self.cell_inc[run] = <np.float64_t>(cells_inf) / <np.float64_t>(com.host.sus.shape[0])

        self.n_samples[run, 0] = shc.n_vectors
        self.n_samples[run, 1] = shc.n_vis
        self.n_samples[run, 2] = shc.n_pcr

        self.reason[run] = reason
        self.time_end[run] = t_end














    