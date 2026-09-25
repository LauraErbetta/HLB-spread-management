# epidemic.pxd
import numpy as np
cimport numpy as np
cimport cython

from numpy.random cimport BitGenerator, bitgen_t

from .extending_distributions cimport SimulationRNG
from .structs cimport SimulationSetup, SetupParameters, LandscapePostprocessing, SimulationParameters, SpatialStructureCitrus, Management, CellsByType, SurveyHelperClass, SaveStructure, SaveMetrics
from .structs cimport State, Event, SimGoal
from .linkedList cimport LinkedList, EventNode

cpdef tuple spiral_initialization(np.int32_t idx, CellsByType com, SpatialStructureCitrus space, LandscapePostprocessing landscape, SetupParameters setupPrms, SimulationSetup simSetup)

cpdef void initialize_run(CellsByType com, CellsByType rsd, SpatialStructureCitrus space,
                            Management mng, LinkedList ll, LandscapePostprocessing landscape, SimulationParameters params, 
                            SetupParameters setupPrms, SimulationSetup simSetup, SimulationRNG rng_sim)

cpdef tuple find_next_det_event(np.float64_t t_new, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, 
                Management mng, SurveyHelperClass shc, LinkedList ll, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim)

cpdef tuple find_next_sto_event(np.float64_t target_R, np.float64_t time, CellsByType og, CellsByType oth, Management mng, 
                            SpatialStructureCitrus space, SimulationParameters params, SimulationSetup simSetup, SimulationRNG rng_sim)

cpdef tuple simulation(int run, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, 
                Management mng, SurveyHelperClass shc, LinkedList ll, LandscapePostprocessing landscape, SimulationParameters params, 
                SetupParameters setupPrms, SimulationSetup simSetup, SimulationRNG rng_sim, SaveStructure save_output)

cpdef tuple run_simulations(int n, CellsByType com, CellsByType rsd, SpatialStructureCitrus space, 
                Management mng, LinkedList ll, LandscapePostprocessing landscape, SimulationParameters params, 
                SetupParameters setupPrms, SimulationSetup simSetup, BitGenerator bg)

