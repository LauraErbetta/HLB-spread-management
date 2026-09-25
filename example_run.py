# %% IMPORTS
import sys
import os
hlb_folder =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, hlb_folder)

import numpy as np
import hlb_model
import faulthandler
faulthandler.enable()

script_dir = os.path.dirname(os.path.abspath(__file__))

# %% Basic setup

region = "Valencia"
nRuns = 1
nYrs = 20
simSetup = hlb_model.structs.SimulationSetup(b"test_run", b"test", b"AfCP", hlb_model.structs.SimGoal.metrics, nRuns, nYrs)
prsSetup = hlb_model.structs.SetupParameters()

simSetup.region = region.encode('utf-8')

import pandas as pd
prop_dp = pd.read_csv(f"Additional_data/prop_type_at_prevalence_{region}.csv")


# %% Change relevant parameters here if necessary

simSetup.management = False
simSetup.switchRemoval = False

prsSetup.r_rem = 0.0

prsSim = hlb_model.structs.SimulationParameters()

prsSim.removal_duration = 30*365

# Fix prop status values based on selected DPs
""" If different design prevalences are used for surveys, values of crp and inf proportions at those prevalences have to be updated """
for i in range(prsSim.survey_by_prop.shape[0]):
    if prsSim.survey_by_prop[i] == 0:
        if prsSim.dp[1,i] > 0 and prsSim.dp[0,i] != prsSim.dp[1,i]:
            raise ValueError(f"Residential dp is > 0 and different from commercial. Must be the same as prop_status defined based on commercial dp.")
        row_dp = prop_dp[prop_dp["dp"] == prsSim.dp[0,i]]
        if row_dp.empty:
            raise ValueError(f"No matching row found in precomputed infection status proportions for dp = {prsSim.dp[0,i]}")
        else:
            prsSim.prop_status[0,i] = row_dp["prop_crp"].values[0]
            prsSim.prop_status[1,i] = row_dp["prop_inf"].values[0]


# %% Create structs

landscape_sim_dict, plot_dict = hlb_model.preprocessing.landscape_config(simSetup.region.decode(), simSetup.vector.decode(), prsSetup.maxCitCom, prsSetup.maxCitRsd, prsSetup.resolution, residential = True)
nCells = plot_dict["citID"].shape[0]

mask_pat_sd, mask_vec_sd, coord_sd_pat, coord_sd_vec, pat_md_cs, vec_md_cs, coord_md_pat, coord_md_vec = hlb_model.spatial_masks.create_dispersal_masks(prsSetup.resolution, prsSetup.md_max_dist, prsSetup.sd_dist_vec, prsSetup.sd_dist_pat, prsSetup.sd_scale, prsSetup.propWithin)

rem_mask, rem_coord = hlb_model.spatial_masks.create_removal_mask(prsSetup.r_rem, prsSetup.resolution)

dem_coord = hlb_model.spatial_masks.create_buffer_mask(prsSetup.resolution, prsSetup.r_dem, "demarcated")
search_coord = hlb_model.spatial_masks.create_buffer_mask(prsSetup.resolution, prsSetup.r_search, "search")

landscape = hlb_model.structs.LandscapePostprocessing(landscape_sim_dict["propBctrlCom"], landscape_sim_dict["propBctrlRsd"], landscape_sim_dict["propResCom"], landscape_sim_dict["propResRsd"], landscape_sim_dict["cc_com"], landscape_sim_dict["cc_rsd"], landscape_sim_dict["com_cells"], landscape_sim_dict["bctrl_cells"], landscape_sim_dict["res_cells"], landscape_sim_dict["weights_org_conv"])

space = hlb_model.structs.SpatialStructureCitrus(prsSetup.resolution, prsSetup.res_big, landscape_sim_dict["citNumsMat"], np.column_stack((plot_dict["rows_cit"], plot_dict["cols_cit"])), landscape_sim_dict["climate"], coord_sd_vec, coord_sd_pat, mask_vec_sd, mask_pat_sd, coord_md_vec, coord_md_pat, vec_md_cs, pat_md_cs, rem_mask, rem_coord, dem_coord, search_coord)
space.compute_arrivals_mdd_pat(landscape_sim_dict["comArray"], landscape_sim_dict["rsdArray"])
space.compute_arrivals_mdd_vec(landscape_sim_dict["comArray"], landscape_sim_dict["rsdArray"])

mng = hlb_model.structs.Management(nCells)    
com = hlb_model.structs.CellsByType(nCells, landscape_sim_dict["comArray"], landscape_sim_dict["cc_com"], landscape_sim_dict["propSprayCom"], prsSim.pE, prsSim.pCI, prsSim.rEC, prsSim.rCI, prsSim.rInf, prsSim.rEst, prsSim.ldProp * prsSim.rDisp, (1-prsSim.ldProp) * prsSim.rDisp)
rsd = hlb_model.structs.CellsByType(nCells, landscape_sim_dict["rsdArray"], landscape_sim_dict["cc_rsd"], landscape_sim_dict["propSprayRsd"], prsSim.pE, prsSim.pCI, prsSim.rEC, prsSim.rCI, prsSim.rInf, prsSim.rEst, prsSim.ldProp * prsSim.rDisp, (1-prsSim.ldProp) * prsSim.rDisp)

em = hlb_model.linkedList.LinkedList()
bg = np.random.PCG64()  # bit generator


# %% Produce maps

simSetup.sim_goal = hlb_model.structs.SimGoal.maps

results, metrics = hlb_model.epidemic.run_simulations(simSetup.nRuns, com, rsd, space, mng, em, landscape, prsSim, 
                prsSetup, simSetup, bg)

# %%
import create_plots
import importlib
importlib.reload(create_plots)

def create_output_folder(base_dir="Model_outputs"):
    # Create the directory recursively
    os.makedirs(base_dir, exist_ok=True)
    
    return base_dir
plot_folder = create_output_folder("Model_outputs")

name = f"{region}_map"

if simSetup.sim_goal == hlb_model.structs.SimGoal.maps:
    create_plots.maps_overlayed_traj(region, plot_dict, results, simSetup.nYears, 1, plot_folder, name, space, plot_years = np.array([2, 5, 8, 10, 12, 14, 16, 20]), symax = True)
# %%
