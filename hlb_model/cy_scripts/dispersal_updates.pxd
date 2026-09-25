# dispersal_updates.pxd
cimport numpy as np
from .structs cimport CellsByType, Management, SpatialStructureCitrus, SimulationParameters
from .extending_distributions cimport SimulationRNG

cpdef void update_pSdR(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_pInf_og, np.float64_t new_pInf_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth, 
                    np.float64_t old_pInf_oth, np.float64_t new_pInf_oth)

cpdef void update_vSdR(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth)

cpdef tuple update_pSdR_noTotal(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_pInf_og, np.float64_t new_pInf_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth, 
                    np.float64_t old_pInf_oth, np.float64_t new_pInf_oth)

cpdef tuple update_vSdR_noTotal(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.float64_t old_vec_og, np.float64_t new_vec_og, np.float64_t old_vec_oth, np.float64_t new_vec_oth)

cpdef np.int8_t update_nearby_availability_pMdd(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth)

cpdef np.int8_t update_nearby_availability_vMdd(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth)

cpdef tuple update_nearby_availability_removal(np.int32_t cell, SpatialStructureCitrus space, CellsByType og, CellsByType oth, np.int8_t val_pat, np.int8_t val_vec)

cpdef void pathogen_arrival(np.int32_t idx, np.int16_t val, CellsByType dest, CellsByType oth, np.float64_t time, Management mng, SpatialStructureCitrus space, SimulationParameters params)

cpdef void vector_arrival(np.int32_t idx, CellsByType dest, CellsByType oth, np.float64_t time, Management mng, SpatialStructureCitrus space)

cpdef void pat_vec_arrival(np.int32_t idx, CellsByType dest, CellsByType oth, np.float64_t time, Management mng, SpatialStructureCitrus space)

# do not change to cpdef, needs to be inlined!
cdef np.int32_t find_destination_md(np.int32_t oIdx, np.float64_t[::1] mask, np.int16_t[:,::1] coord_mask, SpatialStructureCitrus space, SimulationRNG rng_sim)