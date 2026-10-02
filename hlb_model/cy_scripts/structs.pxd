# structs.pxd
cimport numpy as np
from .ratesStructs cimport SqrtBlocks, SqrtBlocks_with_base, SegTree, SegTree_with_base
np.import_array()

cpdef enum State:
    susceptible = <np.uint8_t>0, exposed = <np.uint8_t>1, cryptic = <np.uint8_t>2, infected = <np.uint8_t>3, removed = <np.uint8_t>4,   # host units states: susceptible, exposed, detected, cryptic, infected, removed
    absent = <np.uint8_t>10, present = <np.uint8_t>11, colonised = <np.uint8_t>12, empty =<np.uint8_t>13                                                # vector states: absent, present, colonis

cpdef enum SurveyType:
    regional = <np.uint8_t>0, buffer = <np.uint8_t>1, infArea = <np.uint8_t>2, searchArea = <np.uint8_t>3  # if changes, change in management array access

cpdef enum Event:
    SE = <np.int8_t>1, EC = <np.int8_t>2, CI = <np.int8_t>3,                                                         # host events: exposure, detectability development, infectivity development, symptoms emergence
    AR = <np.int8_t>5, ES = <np.int8_t>6,                                                                 # vector event: arrival (np.int16_t-distance vector dispersal), establishment
    MP0 = <np.int8_t>7, MP1 = <np.int8_t>8, MV0 = <np.int8_t>9, MV1 = <np.int8_t>10,                                            # mid-distance dispersal: fail path, success path, fail vec, success vec
    L10 = <np.int8_t>11, L01 = <np.int8_t>12, L11 = <np.int8_t>13, L00 = <np.int8_t>14,                                         # long-distance dispersal: vector arrival, pathogen arrival, vec + pat arrival, failure
    SB00 = <np.int8_t>15, SB01 = <np.int8_t>16, SB10 = <np.int8_t>17, SB11 = <np.int8_t>18, SR00 = <np.int8_t>19, SR01 = <np.int8_t>20, SR10 = <np.int8_t>21, SR11 = <np.int8_t>22, SI0 = <np.int8_t>23, SI1 = <np.int8_t>24 # surveys
    SD1 = <np.int8_t>30, SD0 = <np.int8_t>31, SR1 = <np.int8_t>32, SR0 = <np.int8_t>33, # spray events: demarcated area on, demarcated area off, regional area on, regional area off
    FC = <np.int8_t>35 # flushing change event 

cpdef enum RatesIdx:
    pSd = 0, vSd = 1, pMd = 2, vMd = 3, ldd = 4, lat = 5, sym = 6, est = 7

cpdef enum SimGoal:
    maps = <np.int8_t>0,
    traj = <np.int8_t>1,
    metrics = <np.int8_t>2,
    metrics_traj = <np.int8_t>3,
    cell_status = <np.int8_t>4,
    test = <np.int8_t>5,
    calibration = <np.int8_t>6

cdef class SimulationSetup:
    cdef public bytes sim_name, region, vector
    cdef public np.int8_t sim_goal
    cdef public np.int16_t nRuns, nYears, spyrMaps, spyrTraj, spyrMetrics
    cdef public np.float64_t tMax, percStop, percDetect
    cdef public np.uint64_t nSteps
    cdef public np.int8_t switchRemoval, stopDetection, stopPercInfected, stopPercInfectedCells, stopPercInfested, detectPercInfectedCells        # conditions to end simulation
    cdef public np.int8_t management, removal, flush, backgroundSpray, sprayContinousDem, sprayContinousReg                # management options
    cdef public np.int32_t infected0
    cdef public bytes infested0, bctrl0, res0                                                  # initial conditions
    cdef public np.int8_t ldd_forced

cdef class SetupParameters:
    # Landscape
    cdef public np.float64_t resolution, res_big                                   # grid resolution
    cdef public np.int16_t maxCitCom, maxCitRsd                          # max amount of host units in each cell
    # Initial conditions
    cdef public np.int16_t infected0, infested0, bctrl0, resistance0     # amount of initially infected, infested, ... cells
    cdef public np.int16_t nExposed0                                     # max n° of exposed units in initially infected cells
    # Dispersal
    cdef public np.float64_t sd_scale
    cdef public np.float64_t sd_dist_vec, sd_dist_pat
    cdef public np.float64_t md_max_dist
    cdef public np.float64_t propWithin
    # Management
    cdef public np.float64_t compliance                                   # proportion of cells compliant with control measures
    cdef public np.float64_t r_dem, r_search, r_rem

cdef class SimulationParameters:
    # Transition rates
    cdef public np.float64_t rDisp, rInf, rEC, rCI, rEst
    cdef public np.float64_t ld_scale, ldProp
    cdef public np.float64_t pE, pCI

    # Management: surveys
    # Dims: cols = survey type, rows: before / after detection
    cdef public np.float64_t[::1] deltaReg, deltaDem               # time interval between surveys
    cdef public np.uint8_t[::1] survey_by_prop

    cdef public np.float64_t[:, ::1] pComSurv, pRsdSurv     # proportion of cells surveyed (out of total available for specific survey) 

    cdef public np.float64_t[:, ::1] dp, prop_status
    cdef public np.float64_t[::1] cl
    cdef public np.float64_t reduce_rsd
    
    cdef public np.float64_t[:, ::1] pHostPCR     # prop of hosts surveyed via PCR (out of those visually inspected)
    cdef public np.uint8_t[:, ::1] nVec
    cdef public np.uint16_t[:, ::1] nHostVis # number of vector/hosts surveyed
    cdef public np.float64_t pVis, pVec, pRem
    cdef public np.float64_t[::1] pPCR # C, I
    cdef public np.float64_t[::1] w_rem         # weights for removal of units, based on unit status
    cdef public np.float64_t pNoSearch

    cdef public np.float64_t time_for_eradication
    cdef public np.float64_t removal_duration

    # Management: control
    cdef public np.float64_t[::1] infectiousness, susceptibility      # infectiousness and susceptibility, resistant and non-resistant units
    cdef public np.float64_t bctrlEffect
    cdef public np.int16_t durationSpray
    cdef public np.float64_t bgSprayEffect, sprayEffect, modReg, sprayEffectDem, sprayEffectReg
    cdef public np.float64_t deltaSprayDem, deltaSprayReg

    # Flush season
    cdef public np.float64_t[::1] flushValues, deltaFlush

cdef class LandscapePostprocessing:

    cdef public np.ndarray cc_com, cc_rsd
    cdef public np.ndarray propBctrlCom, propResCom, propBctrlRsd, propResRsd
    cdef public np.ndarray com_cell, bctrl_cell, res_cell, weights_org_conv

cdef class SpatialStructureCitrus:
    cdef public np.float64_t res, p_md_vec, p_md_pat
    cdef public np.int32_t[:, ::1] landscape2d, coord_landscape
    cdef public Py_ssize_t max_row, max_col
    cdef public np.int8_t[:, ::1] coord_transf
    cdef public np.int16_t[:, ::1] sd_mask_coord_pat, md_mask_coord_pat, sd_mask_coord_vec, md_mask_coord_vec # int8 is enough if all radii are < 12 km (range is -128 - 127)
    cdef public np.uint8_t[:, ::1] rem_coord, dem_coord, search_coord # uint8 is enough if all radii are < 25 km (range is 0 - 255) - sliced masks
    cdef public np.float64_t[::1] clim, vecKern, patKern, md_vec_mask, md_pat_mask, rem_p_mask, default_0_rem_mask, rem_mask_to_use
    cdef public np.ndarray _susc_pMdd_0, _susc_vMdd_0
    cdef public np.uint32_t[::1] susc_pMdd_0, susc_vMdd_0, susc_pMdd_now, susc_vMdd_now 

    # Keep track of infected cells
    cdef public np.ndarray _infCells, _infCells1km, _small_to_big
    cdef public np.uint8_t[::1] infCells, infCells1km
    cdef public np.int32_t[::1] small_to_big
    cdef public Py_ssize_t unique_1km
    cdef public np.float64_t res_big

    # Ranges:
    # int8 [-128, 127]; uint8 [0, 255]
    # int16 [-32768, 32767]; uint16 [0, 65635]
    # int32 approx. [-2e9, 2e9]; uint32 [0, approx. 4e9]

    cdef compute_small_to_big(self)
    cpdef void reset(self)
    cpdef void compute_arrivals_mdd_pat(self, np.uint16_t[::1] com_units, np.uint16_t[::1] rsd_units)
    cpdef void compute_arrivals_mdd_vec(self, np.uint16_t[::1] com_units, np.uint16_t[::1] rsd_units)

cdef class Management:
    cdef public np.int32_t nCells, totReg
    cdef public np.ndarray _buf, _inf, _reg, _dem, _comp, _used                        # 1 if cell belongs to demarcated, regional, compliant cells, 0 if not (arrays)
    cdef public np.uint8_t[::1] buffer, infected, regional, demarcated, compliance, used_for_dem      # 1 if cell belongs to demarcated, regional, compliant cells, 0 if not (memoryviews)
    # Surveys results
    cdef public np.uint8_t detected, vectorFull, keep_attempt, switchRemoval
    cdef public np.int8_t which_det
    cdef public np.float64_t time_det, cit_inc_det, cell_inc_det, cell1km_inc_det, time_stop_attempt
    cdef public np.uint64_t n_vec_det, n_vis_det, n_pcr_det

    cdef public np.float64_t current_flush

    # Practical eradication
    cdef public np.float64_t time_last_det, time_erad, cit_inc_erad, cell_inc_erad, cell1km_inc_erad
    cdef public np.uint8_t believed_eradicated

    cpdef reset(self)

cdef class ControlByType:
    cdef np.int32_t nCells
    cdef public np.ndarray _spray, _sprayBG, _sprayDem, _sprayReg, _spray_prop, _bctrl, _inf_res, _susc_res
    cdef public np.float64_t[::1] spray, sprayBG, sprayDem, sprayReg, spray_prop
    # Biocontrol and resistance effects (fixed at the beginning of simulation)
    cdef public np.float64_t[::1] bctrl, inf_res, susc_res

cdef class HostClass:

    cdef public np.float64_t pE, pCI

    cdef public np.uint16_t[::1] tot, sus, exp, crp, inf, rem
    cdef public np.uint8_t[::1] status_host
    cdef public np.float64_t[:,::1] time_host

    cdef public np.float64_t[::1] pInf
    cdef public np.float64_t[::1] cs_time_symptom # store time host became symptomatic

    cdef void update_pInf(self, np.int32_t idx)
    cdef void update_SE(self, np.int32_t idx, np.int16_t val, np.float64_t time, RateClass rates, CellSummary summary)
    cdef void update_EC(self, np.int32_t idx, np.int16_t val, np.float64_t time, RateClass rates)
    cdef void update_CI(self, np.int32_t idx, np.int16_t val, np.float64_t time, RateClass rates)
    cpdef tuple update_removal(self, np.int32_t idx, np.int16_t S_rem, np.int16_t E_rem, np.int16_t C_rem, np.int16_t I_rem, np.float64_t time, ControlByType ctrl, VectorClass vector, RateClass rates, CellSummary summary, np.uint8_t[::1] rates_flags)

cdef class RateClass:

    cdef public np.float64_t rEC, rCI, rInf, rEst, rLdd, rMdd, rSdd
    cdef public SegTree latR, symR, estR, vMdR, vSdR
    # cdef public SqrtBlocks vSdR
    cdef public SegTree_with_base pMdR, lddR, pSdR
    # cdef public SqrtBlocks_with_base pSdR

    cdef np.float64_t get_total_rate(self)

cdef class VectorClass:

    cdef public np.ndarray _dens_vec, _cc_vec
    cdef public np.float64_t[::1] dens_vec, cc_vec                    # vector density 

    cdef public np.uint8_t[::1] status_vec
    cdef public np.float64_t[:,::1] time_vec

    cdef void update_AR(self, np.int32_t idx, np.float64_t time, RateClass rates, CellSummary summary)
    cdef void update_ES(self, np.int32_t idx, np.float64_t time, ControlByType ctrl, HostClass host, RateClass rates, CellSummary summary)

cdef class CellSummary:

    cdef public np.int32_t S_tot, citrus_total, totStanding, totInfested, totInfected, cit_cells, S_R_tot
    cdef public np.uint8_t vectorFull, all_infected
    cdef public np.ndarray _infected, _standing, _infested, _n_stand
    cdef public np.uint8_t[::1] infected, standing, infested
    cdef public np.uint16_t[::1] n_stand
    cdef public np.uint16_t maxCit

cdef class CellsByType:

    cdef public np.int32_t nCells
    cdef public HostClass host
    cdef public ControlByType ctrl
    cdef public VectorClass vector
    cdef public RateClass rates
    cdef public CellSummary summary

    cpdef void reset(self, np.ndarray cc)
    cpdef void update_pInf(self, np.int32_t idx)
    cpdef void update_SE(self, np.int32_t idx, np.int16_t val, np.float64_t time)
    cpdef void update_EC(self, np.int32_t idx, np.int16_t val, np.float64_t time)
    cpdef void update_CI(self, np.int32_t idx, np.int16_t val, np.float64_t time)
    cpdef tuple update_removal(self, np.int32_t idx, np.int16_t S_rem, np.int16_t E_rem, np.int16_t C_rem, np.int16_t I_rem, np.float64_t time, np.uint8_t[::1] rates_flags)
    cpdef void update_AR(self, np.int32_t idx, np.float64_t time)
    cpdef void update_ES(self, np.int32_t idx, np.float64_t time)

cdef class SurveyHelperClass:

    cdef public np.int32_t nCells, totSearch
    cdef public np.ndarray _cell_success, _cell_success_host, _cells_surveyed, _cells_search, _comDetected, _rsdDetected, _add_demarcated
    cdef public np.uint8_t[::1] cell_success, cell_success_host, cells_search, add_demarcated
    cdef public np.uint16_t[:, ::1] comDetected, rsdDetected

    cdef public np.uint64_t n_vectors, n_vis, n_pcr, n_sus_rem

    cpdef reset(self)
    cpdef reset_full(self)

cdef class SaveStructure:

    cdef public np.ndarray _maps_S, _maps_I, _maps_R, _traj_P_units, _traj_P_cells, _traj_V_cells, _zones
    cdef public np.ndarray _yield_proxy_s_c0, _yield_proxy_e_c0, _yield_proxy_c_c0, _yield_proxy_i_c0, _yield_proxy_s_c1, _yield_proxy_e_c1, _yield_proxy_c_c1, _yield_proxy_i_c1, _tot_c0, _tot_c1
    cdef public np.ndarray _infections_all, _infections_rsd, _R_units, _RS_units
    cdef public np.ndarray _traj_IC
    cdef public np.uint32_t[:,::1] traj_IC
    cdef public np.uint16_t[:,::1] maps_S, maps_I, maps_R
    cdef public np.uint8_t[:,::1] zones
    cdef public np.float32_t[:,::1] traj_P_units, traj_P_cells, traj_V_cells
    cdef public np.uint32_t[:,::1] yield_proxy_s_c0, yield_proxy_e_c0, yield_proxy_c_c0, yield_proxy_i_c0, yield_proxy_s_c1, yield_proxy_e_c1, yield_proxy_c_c1, yield_proxy_i_c1
    cdef public np.uint32_t[::1] tot_c0, tot_c1
    cdef public np.uint32_t[:,::1] infections_all, infections_rsd, R_units, RS_units
    cdef public np.ndarray n_e_cells, n_c_cells, n_i_cells

    cdef update_maps(self, Py_ssize_t sp_last, Py_ssize_t sp_current, HostClass com_cit, HostClass rsd_cit, Management mng)
    cdef update_trajectories(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd)
    cdef update_cali_trajectories(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd)
    cdef update_metrics(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd, Management mng)
    cdef save_cell_status(self, np.int16_t run, CellsByType com, CellsByType rsd)
    cdef update_maps_end(self, Py_ssize_t sp_last, Py_ssize_t sp_current, HostClass com_cit, HostClass rsd_cit, Management mng)
    cdef update_trajectories_end(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd)
    cdef update_cali_trajectories_end(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd)
    cdef update_metrics_end(self, np.int16_t run, Py_ssize_t sp_last, Py_ssize_t sp_current, CellsByType com, CellsByType rsd, Management mng)    

cdef class SaveMetrics:

    cdef public np.ndarray _S_removed, _tot_removed, _cit_inc, _cell_inc, _yield_proxy, _detection_results, _n_samples, _reason, _time_end, _hlb_inc, _cs_time_symptom, _n_symptom

    cdef public np.uint32_t[::1] S_removed, tot_removed
    cdef public np.float64_t[::1] cit_inc, cell_inc, hlb_inc
    cdef public np.float64_t[:,::1] detection_results
    cdef public np.uint64_t[:,::1] n_samples

    cdef public np.int8_t[::1] reason
    cdef public np.float64_t[::1] time_end

    cdef public np.float64_t[::1] cs_time_symptom, n_symptom

    cdef update_run(self, np.int16_t run, CellsByType com, CellsByType rsd, SurveyHelperClass shc, np.float64_t[::1] detection, np.int8_t reason, np.float64_t t_end)