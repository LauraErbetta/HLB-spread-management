# management.pxd
import numpy as np
cimport numpy as np
cimport cython

from numpy.random cimport BitGenerator
from .extending_distributions cimport SimulationRNG

from .structs cimport SimulationSetup, SimulationParameters, SpatialStructureCitrus, Management, CellsByType, ControlByType, SurveyHelperClass
from .linkedList cimport LinkedList, EventNode

cpdef np.uint8_t host_detection(np.uint8_t nVis, np.float64_t propPCR, np.int32_t[::1] surveyed, np.uint16_t[:, ::1] detected, CellsByType og, np.float64_t pVis, np.float64_t[::1] pPCR, SurveyHelperClass shc, SimulationRNG rng_sim)

cpdef remove_detected(np.float64_t time, SurveyHelperClass shc, np.uint8_t[::1] rates_flags_com, np.uint8_t[::1] rates_flags_rsd, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SimulationParameters params, SimulationRNG rng_sim)

cpdef void update_spray(np.uint8_t[::1] where_spray, np.uint8_t[::1] rates_flags_com, np.uint8_t[::1] rates_flags_rsd, CellsByType com, CellsByType rsd, 
                        Management mng, SpatialStructureCitrus space, SimulationParameters params)

cpdef tuple update_demarcated_area(SurveyHelperClass shc, np.float64_t time, Management mng, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, LinkedList ll, SimulationParameters params, SimulationSetup simSetup)

cpdef np.uint8_t randomized_search_survey(np.ndarray cells, np.uint8_t nVis, np.float64_t propPCR, np.float64_t pComs, np.float64_t pRsds, CellsByType com, CellsByType rsd, Management mng, SurveyHelperClass shc, SimulationParameters params, SimulationRNG rng_sim)

cpdef void update_spray_with_rates(np.uint8_t[::1] where_spray, CellsByType com, CellsByType rsd, 
                        Management mng, SpatialStructureCitrus space, SimulationParameters params)

cpdef tuple randomized_survey(np.float64_t time, np.uint8_t survey_type, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, SimulationParameters params, SimulationRNG rng_sim)

cpdef np.int8_t do_survey(EventNode event, np.float64_t time, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, LinkedList ll, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim)

cdef inline void build_search_area(np.int32_t idx, SpatialStructureCitrus space, SurveyHelperClass shc)

cpdef tuple compute_n_cells_to_survey(np.uint8_t survey_type, CellsByType com, CellsByType rsd, Management mng, SurveyHelperClass shc, SimulationParameters params)

cpdef tuple compute_n_cells_to_survey_prop(np.uint8_t survey_type, CellsByType com, CellsByType rsd, Management mng, SurveyHelperClass shc, SimulationParameters params)

cpdef make_detection_happen(np.float64_t time, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, Management mng, SurveyHelperClass shc, LinkedList ll, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim)