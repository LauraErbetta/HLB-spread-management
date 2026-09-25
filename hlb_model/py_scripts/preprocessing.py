from importlib import resources
import numpy as np
import os

def load_data(filename: str) -> np.ndarray:
    """
    Load a CSV-like file from hlb_model.Data as a numpy array.
    """
    return np.loadtxt(resources.files("hlb_model.Data").joinpath(filename), delimiter=",")

def landscape_config(region, vector, maxCitCom, maxCitRsd, resolution, residential = True):
    
    regions_avail = ["Valencia", "Sevilla", "Murcia"]
    
    if not region in regions_avail:
        raise ValueError(f"Region {region} not among available (Valencia, Sevilla, Murcia). Check spelling and capitalization.")

    match (vector).lower(): # import climate matrix for selected vector
        case "acp":
            climMat = load_data(os.path.join(region, f"cs_ACP_{region}.txt"))
        case "afcp":
            climMat = load_data(os.path.join(region, f"cs_AfCP_{region}.txt"))
        case _:
            raise ValueError(f"Vector {vector} not among the vector options available.")
        
    comMat = load_data(os.path.join(region, f"com_100_{region}.txt"))
    if residential:
        rsdMat = load_data(os.path.join(region, f"rsd_100_{region}.txt"))
    else:
        rsdMat = np.zeros_like(comMat)
        
    abnMat = load_data(os.path.join(region, f"abn_100_{region}.txt"))
    orgMat = load_data(os.path.join(region, f"org_100_{region}.txt"))
    waterMat = load_data(os.path.join(region, f"water_corine_{region}.txt"))
    
    comUnitsMat = np.round(comMat * maxCitCom)
    rsdUnitsMat = np.round(rsdMat * maxCitRsd)
    
    citUnitsMat = comUnitsMat + rsdUnitsMat    # overall citrus
    
    gridRows, gridCols = citUnitsMat.shape
    rowsMat = np.tile(np.arange(gridRows).reshape(-1, 1), (1, gridCols))
    colsMat = np.tile(np.arange(gridCols), (gridRows, 1))
    numsMat = np.arange(gridRows * gridCols).reshape(gridRows, gridCols)        # cells numbered regardless of citrus presence/absence
    citNumsMat = np.full(citUnitsMat.shape, -1, dtype=np.int32)
    citNumsMat[citUnitsMat > 0] = np.arange(np.sum(citUnitsMat > 0))                      # -1 where there's no citrus, cells with citrus numbered with own index

    waterArray = waterMat.flatten()
    citNumsArray = citNumsMat.flatten()
    cellsArray = np.copy(citNumsArray)
    cellsArray[np.logical_and(cellsArray < 0, waterArray == 1)] = -2                                            # -2 is water, -1 is no citrus but land, >= 0 is idx of citrus cell
    rows_coord_all = rowsMat.flatten()
    cols_coord_all = colsMat.flatten()

    # rounding to nearest unit number
    # if unit = tree = 20m2, it becomes one only when at least 10m2 are present, and so on
    comArray = (comUnitsMat[citUnitsMat> 0].flatten()).astype(np.uint16)   
    rsdArray = (rsdUnitsMat[citUnitsMat> 0].flatten()).astype(np.uint16)
    
    # Keep only cells with at least some citrus
    comArrayDens = comMat[citUnitsMat> 0].flatten()
    rsdArrayDens = rsdMat[citUnitsMat> 0].flatten()
    climArray = climMat[citUnitsMat> 0].flatten()
    abnArrayDens = abnMat[citUnitsMat> 0].flatten()
    orgArrayDens = orgMat[citUnitsMat> 0].flatten()
    rows_cit = np.copy(rows_coord_all[cellsArray >= 0])
    cols_cit = np.copy(cols_coord_all[cellsArray >= 0])
    
    # Set to 0 where cut off by rounding
    comArrayDens[comArray == 0] = 0
    rsdArrayDens[rsdArray == 0] = 0
    abnArrayDens[comArray == 0] = 0
    orgArrayDens[comArray == 0] = 0
    
    convArrayDens = comArrayDens - abnArrayDens - orgArrayDens
    
    idArray = (citNumsMat[citUnitsMat> 0].flatten()).astype(np.int32)
    nCells = idArray.shape[0]
    if nCells != (np.max(idArray) + 1):
        raise ValueError(f"Cells ID do not match number of cells")
    if (idArray != np.arange(nCells)).any():
        raise ValueError(f"Cells ID are not ordered as expected")
    
    cc_com = climArray * comArrayDens
    cc_rsd = climArray * rsdArrayDens
    
    # Proportions
    propBctrlCom = np.divide(orgArrayDens + abnArrayDens, comArrayDens, out=np.zeros_like(comArrayDens), where=comArrayDens != 0)
    propBctrlRsd = np.ones(nCells, dtype=np.float64)
    propBctrlRsd[rsdArrayDens == 0] = 0
    propResCom = np.divide(convArrayDens, comArrayDens, out=np.zeros_like(comArrayDens), where=comArrayDens != 0)
    propResRsd = np.zeros(nCells, dtype=np.float64)
    propSprayCom = np.divide(convArrayDens, comArrayDens, out=np.zeros_like(comArrayDens), where=comArrayDens != 0)
    propSprayRsd = np.zeros(nCells, dtype=np.float64)

    # Cells with properties
    com_cell = (np.nonzero(comArray)[0]).astype(np.int32)
    bctrl_cell = (np.nonzero(propBctrlCom + propBctrlRsd)[0]).astype(np.int32)
    res_cell = (np.nonzero(propResCom + propResRsd)[0]).astype(np.int32)
    
    # Weights to select initial infection / infestation cell: idea is that vector / HLB will be introduced when importing plants / material, so only in active orchards (organic and conventional, but not abandoned)
    active_orchards_density = comArrayDens - abnArrayDens
    weights_org_conv = active_orchards_density / active_orchards_density.sum()
    
    landscape_sim_dict = {
        "citNumsMat": citNumsMat.astype(np.int32),
        "cc_com": cc_com,
        "cc_rsd": cc_rsd,
        "propBctrlCom": propBctrlCom,
        "propResCom": propResCom,
        "propSprayCom": propSprayCom,
        "propBctrlRsd": propBctrlRsd,
        "propResRsd": propResRsd,
        "propSprayRsd": propSprayRsd,
        "com_cells": com_cell,
        "bctrl_cells": bctrl_cell,
        "res_cells": res_cell,
        "weights_org_conv": weights_org_conv,
        "comArray": comArray,
        "rsdArray": rsdArray,
        "climate": climArray
        }
    
    plot_dict = {
        "res": resolution,
        # All landscape
        "cells": cellsArray, # -2 water, -1 no cit, >= 0 idx cit cell
        "rows_all": rows_coord_all,
        "cols_all": cols_coord_all,
        # Citrus only
        "citID": idArray,
        "comDens": comArrayDens,
        "rsdDens": rsdArrayDens,
        "rows_cit": rows_cit.astype(np.int32),
        "cols_cit": cols_cit.astype(np.int32),
        "totCit": comArray.sum() + rsdArray.sum()
    }
    
    return landscape_sim_dict, plot_dict

def random_landscape(rng, size = 10, setupParams = None):

    com_dens = rng.random(size*size)
    abn_dens = rng.random(size*size)
    org_dens = rng.random(size*size)
    rsd_dens = rng.random(size*size)
    vector_rand = rng.random(size*size)
    vector_presence = np.zeros_like(com_dens)

    zero_com = rng.integers(0, com_dens.shape[0], round(0.2*com_dens.shape[0]))
    zero_rsd = rng.integers(0, com_dens.shape[0], round(0.6*com_dens.shape[0]))

    for i in range(com_dens.shape[0]):
        if i in zero_com:
            com_dens[i] = 0
            org_dens[i] = 0
            abn_dens[i] = 0
        else:
            abn_dens[i] *= com_dens[i] * 0.1
            org_dens[i] *= com_dens[i] * 0.2
        if i in zero_rsd:
            rsd_dens[i] = 0
        else:
            rsd_dens[i] *= max(1-com_dens[i], 0.03)
            
        if com_dens[i] + rsd_dens[i] > 0:
            vector_presence[i] = 1 if vector_rand[i] > 0.6 else 0
            
    cit_arr = com_dens + rsd_dens
            
    clim_arr = rng.random(size*size) * (0.7 - 0.5) + 0.5 # limited between 0.5 and 0.5
    
    trend_arr = np.zeros_like(com_dens)
    water_arr = np.zeros_like(com_dens)
    no_cit_tot = (cit_arr <= 0).sum()
    where_water = np.random.choice(np.where(cit_arr <= 0)[0], size = int(np.ceil(no_cit_tot*0.5)))
    water_arr[where_water] = 1
    
    gridRows = size
    gridCols = size
    rowsMat = np.tile(np.arange(gridRows).reshape(-1, 1), (1, gridCols))
    colsMat = np.tile(np.arange(gridCols), (gridRows, 1))
    nums_arr = np.arange(gridRows * gridCols)        # cells numbered regardless of citrus presence/absence
    cit_nums_arr = -1 * np.ones_like(nums_arr)
    
    cit_nums_arr[cit_arr > 0] = np.arange(np.sum(cit_arr > 0))
    cit_nums_arr[water_arr == 1] = -2
    

    waterMat = water_arr.reshape((size, size))
    citNumsMat = cit_nums_arr.reshape((size,size))
    numsMat = nums_arr.reshape((size,size))
    cellsArray = np.copy(cit_nums_arr)
    cellsArray[water_arr == 1] = -2
    x_coord_all = rowsMat.flatten()
    y_coord_all = colsMat.flatten()
    
    comArrayDens = com_dens[cit_arr> 0]
    rsdArrayDens = rsd_dens[cit_arr> 0]
    vecArray = (vector_presence[cit_arr> 0]).astype(np.uint8)
    climArray = clim_arr[cit_arr> 0]
    abnArrayDens = abn_dens[cit_arr> 0]
    orgArrayDens = org_dens[cit_arr> 0]
    trendArray = trend_arr[cit_arr> 0]
    x_cit = np.copy(x_coord_all[cellsArray >= 0])
    y_cit = np.copy(y_coord_all[cellsArray >= 0])
    
    # Keep only cells with at least some citrus (regardless of type)
    x_cit = np.copy(x_coord_all[cellsArray >= 0])
    y_cit = np.copy(y_coord_all[cellsArray >= 0])

    comArray = (np.ceil(comArrayDens * setupParams.maxCitCom)).astype(np.uint16)
    rsdArray = (np.ceil(rsdArrayDens * setupParams.maxCitRsd)).astype(np.uint16)
    #orgArray = np.ceil(orgArrayDens * params.maxCitCom)
    #abnArray = np.min([np.ceil(abnArrayDens * params.maxCitCom), comArray - orgArray], axis = 0)
    
    # Set to 0 where cut off by rounding
    comArrayDens[comArray == 0] = 0
    rsdArrayDens[rsdArray == 0] = 0
    abnArrayDens[comArray == 0] = 0
    orgArrayDens[comArray == 0] = 0
    convArrayDens = comArrayDens - abnArrayDens - orgArrayDens
    
    idArray = (cit_nums_arr[cit_arr > 0]).astype(np.int32)
    nCells = idArray.shape[0]
    if nCells != (np.max(idArray) + 1):
        raise ValueError(f"Cells ID do not match number of cells")
    if (idArray != np.arange(nCells)).any():
        raise ValueError(f"Cells ID are not ordered as expected")
    
    cc_com = climArray * comArrayDens
    cc_rsd = climArray * rsdArrayDens
    
    # Proportions
    propBctrlCom = np.divide(orgArrayDens + abnArrayDens, comArrayDens, out=np.zeros_like(comArrayDens), where=comArrayDens != 0)
    propBctrlRsd = np.ones(nCells, dtype=np.float64)
    propBctrlRsd[rsdArrayDens == 0] = 0
    propResCom = np.divide(convArrayDens, comArrayDens, out=np.zeros_like(comArrayDens), where=comArrayDens != 0)
    propResRsd = np.zeros(nCells, dtype=np.float64)
    propSprayCom = np.divide(convArrayDens, comArrayDens, out=np.zeros_like(comArrayDens), where=comArrayDens != 0)
    propSprayRsd = np.zeros(nCells, dtype=np.float64)

    # Cells with properties
    com_cell = (np.nonzero(comArray)[0]).astype(np.int32)
    bctrl_cell = (np.nonzero(propBctrlCom + propBctrlRsd)[0]).astype(np.int32)
    res_cell = (np.nonzero(propResCom + propResRsd)[0]).astype(np.int32)
    
    # Weights to select initial infection / infestation cell: idea is that vector / HLB will be introduced when importing plants / material, so only in active orchards (organic and conventional, but not abandoned)
    active_orchards_density = comArrayDens - abnArrayDens
    weights_org_conv = active_orchards_density / active_orchards_density.sum()
    
    landscape_sim_dict = {
        "cc_com": cc_com,
        "cc_rsd": cc_rsd,
        "propBctrlCom": propBctrlCom,
        "propResCom": propResCom,
        "propSprayCom": propSprayCom,
        "propBctrlRsd": propBctrlRsd,
        "propResRsd": propResRsd,
        "propSprayRsd": propSprayRsd,
        "com_cells": com_cell,
        "bctrl_cells": bctrl_cell,
        "res_cells": res_cell,
        "weights_org_conv": weights_org_conv,
        "comArray": comArray,
        "rsdArray": rsdArray,
        "climate": climArray,
        }    
    
    plot_dict = {
        "res": setupParams.resolution,
        # All landscape
        "cells": cellsArray, # -2 water, -1 no cit, >= 0 idx cit cell
        "x_all": x_coord_all,
        "y_all": y_coord_all,
        # Citrus only
        "comDens": comArrayDens,
        "rsdDens": rsdArrayDens,
        "citID": idArray,
        "x_cit": x_cit.astype(np.int32),
        "y_cit": y_cit.astype(np.int32),
    }
    
    landscape_dict = {
        "res": setupParams.resolution,
        "gridRows": gridRows,
        "gridCols": gridCols,
        "citNumsMat": citNumsMat.astype(np.int32),
        "numsMat": numsMat.astype(np.int32)
    }
    
    return landscape_sim_dict, plot_dict, landscape_dict

