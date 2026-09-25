import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.offsetbox
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, LogNorm
import matplotlib.patches as mpatches
import csv
import os
import string
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'

cm = 1/2.54
fig_width = 19*cm
fs_ticks = 9
fs_labels = 10
fs_title = 11
fs_legend = 7
fs_scalebar = 8
fs_legend_title = 8

letters = list(string.ascii_lowercase)

# %%
class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """Horizontal scale bar in data units."""
    def __init__(self, ax, size=1, label="", loc='lower center',
                 pad=0.4, borderpad=0.5, sep=2, ppad=0,
                 frameon=True, linekw={}, text_color="black", fontsize = mpl.rcParams['font.size'], **kwargs):

        # Use data coordinates for both X and Y
        trans = ax.transData
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        
        # Create horizontal line
        line = Line2D([0, size], [0, 0], **linekw)
        # Create vertical end caps
        vline1 = Line2D([0, 0], [-size*0.05, size*0.05], **linekw)
        vline2 = Line2D([size, size], [-size*0.05, size*0.05], **linekw)
        
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)

        # Label
        txt = matplotlib.offsetbox.TextArea(label, textprops={"color": text_color, "fontsize": fontsize})
        vp = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                          align="center", pad=ppad, sep=sep)

        super().__init__(loc=loc, pad=pad, borderpad=borderpad,
                         child=vp, frameon=frameon, **kwargs)
        
        if frameon:
            self.patch.set_facecolor("white")
            self.patch.set_alpha(0.5)
            self.patch.set_edgecolor("none")
# %%

def maps_overlayed_traj(region, plot_dict, results, nYears, compact_res, folder, identifier, space, plot_years = np.array([2, 5, 8, 10, 12, 14, 16, 20]), demarcate = True, symax = True):
    """
    Generate maps of infected citrus density at multiple time points, compacting results in 1 km cell. Single simulation.
    Both proportion of infection and density of infection.
    """
    
    # region Setup
    if len(plot_years) != 8:
        print("ATTENTION!!! Legend positioned for 3x3 figure")
    removed = 0
    
    # Time
    spyr = 4  # steps per year
    if spyr > 1:
        delta_steps = 1 / spyr # how many months in between each save
        label_time = "Year"
        time_steps = np.arange(0, nYears + 0.00001, delta_steps)
    else:
        delta_steps = 1 / spyr # how many years in between each save
        label_time = "Year"
        time_steps = np.arange(0, nYears + 0.00001, delta_steps).astype(int)
    
    # Space
    if compact_res <= plot_dict["res"]:
        raise ValueError("Compact resolution must be a multiple (and higher) than simulation resolution")
    elif (compact_res / plot_dict["res"]) % round(compact_res / plot_dict["res"]):
        raise ValueError("Compact resolution must be a multiple (and higher) than simulation resolution")

    space_step = round(compact_res / plot_dict["res"])
    nrows = max(plot_dict["rows_all"]) + 1
    ncols = max(plot_dict["cols_all"]) + 1
    
    nrows_comp = int(nrows / space_step)
    ncols_comp = int(ncols / space_step)
    
    # Total each cell, each year
    sus_units = results._maps_S
    hlb_units = results._maps_I
    rem_units = results._maps_R

    tot_units = sus_units + hlb_units + rem_units

    densInfected = (plot_dict["comDens"] + plot_dict["rsdDens"]) * hlb_units / tot_units

    propRemoved = rem_units / tot_units
    # propRemoved[propRemoved < 1] = 0 # interested only in full removal
    # propRemoved[rem_units == tot_units] = 1 # interested in all removal
    if np.any(propRemoved):
        removed = 1
        
    # endregion        

    # region Data for traj
    tot_cit = tot_units[0,:].sum()
    tot_cells = tot_units.shape[1]
    
    infections = hlb_units + rem_units
    
    inc_units = 100 * infections.sum(axis = 1) / tot_cit
    inc_cells = 100 * (infections > 0).sum(axis = 1) / tot_cells
    hlb_only_inc = 100 * hlb_units.sum(axis = 1) / tot_cit
    
    # End at
    end = np.where(hlb_only_inc == 0)[0]
    if end.size:
        end = end[0] + 1
    else:
        end = hlb_only_inc.shape[0]
    
    nYears = (end - 1)/spyr
    
    x_axis = time_steps[:end]

    if 10 < nYears < 35:
        x_ticks = np.arange(0, nYears + 1, 5, dtype=int)
        x_minor = np.arange(0, nYears + 1, 5, dtype=int)
    elif nYears <= 10:
        x_ticks = np.arange(0, nYears + 1, 2, dtype=int)
        x_minor = np.arange(0, nYears + 1, 2, dtype=int)
    else:
        x_ticks = np.arange(0, nYears + 1, 10, dtype=int)
        x_minor = np.arange(0, nYears + 1, 10, dtype=int) # change to 5 if you want them to show
    
    if np.median(hlb_only_inc) <= 1 and symax == True:
        sym_ax = 1
    else:
        sym_ax = 0
    
    if nYears < plot_years[-1]:
        print(f"Changing last map to year {nYears} because of epidemic end")
        plot_years = plot_years.astype(float)
        if nYears > 16:
            plot_years[-1] = nYears
        elif 15 < nYears <= 16:
            plot_years = np.array([2, 5, 8, 10, 12, 14, 15, nYears], dtype=np.float32)
        elif 12 < nYears <= 15:
            plot_years = np.array([2, 5, 8, 12, nYears], dtype=np.float32)
        elif 10 < nYears <= 13:
            plot_years = np.array([2, 5, 8, 10, nYears], dtype=np.float32)
        elif 8 < nYears <= 11:
            plot_years = np.array([2, 4, 6, 8, nYears], dtype=np.float32)
        elif 5 < nYears <= 8:
            plot_years = np.array([2, 3, 4, 5, nYears], dtype=np.float32)
        elif 4 < nYears <= 5:
            plot_years = np.array([1, 2, 3, 4, nYears], dtype=np.float32)
        elif nYears <= 4:
            print(f"Not enough years of simulation. Function cannot work")
            return 0
        
    elif nYears > plot_years[-1]:
        print(f"ATTENTION: Do you want the last map to be before the epidemic end / simulation end time?")
        
    # endregion
    
    tp_maps = (plot_years * spyr).astype(int)
    
    # region Demarcated
    def create_mask_demarcated(dem_coord):
        delta = np.max(dem_coord)
        mask = np.zeros((delta*2 + 1, delta*2 + 1))
        
        r = delta.astype(np.int32)
        c = delta.astype(np.int32)
        
        for w in range(space.dem_coord.shape[0]):
            r_rel, c_rel = space.dem_coord[w,:]
            for z in range(8):
                r_search = r + r_rel * space.coord_transf[z,0] + c_rel * space.coord_transf[z,1]
                c_search = c + r_rel * space.coord_transf[z,2] + c_rel * space.coord_transf[z,3]
                
                mask[r_search, c_search] = 1
                
        # Buffered landscape
        maps_cit = np.array(space.landscape2d)
        buff = delta.astype(np.int32)
        maps_buffer = -np.ones((maps_cit.shape[0] + buff*2, maps_cit.shape[1] + buff*2), dtype=np.int32)
        maps_buffer[buff:maps_buffer.shape[0]-buff, buff:maps_buffer.shape[1]-buff] = maps_cit
        
        maps_all = -np.ones((maps_cit.shape[0] + buff*2, maps_cit.shape[1] + buff*2), dtype=np.int32)
        idx_all_max = np.array(space.landscape2d).size
        maps_all[buff:maps_buffer.shape[0]-buff, buff:maps_buffer.shape[1]-buff] = np.arange(idx_all_max).reshape(space.landscape2d.shape) # all region
        
        return mask.astype(bool), maps_buffer, maps_all, buff
    
    if not (results._zones > 0).any():
        demarcate = False # if never demarcate, do not plot it
        
    if demarcate:        
        # Dem area around cell
        # Note: this is the demarcated area including cells without citrus
        mask_dem, maps_buffer, maps_all, buff = create_mask_demarcated(np.array(space.dem_coord))

        dem_exp = np.zeros((len(tp_maps), maps_all.shape[0], maps_all.shape[1]))
        
        for i, tp in enumerate(tp_maps):
            inf_area = np.where(results._zones[tp,:] == 2)[0]
            for idx in inf_area:
                r, c = np.array(space.coord_landscape[idx,:]) + buff
                
                cells = maps_all[r - buff:r + buff + 1, c - buff:c + buff + 1]
                where_dem = (cells >= 0) & mask_dem
                
                dem_exp[i, r - buff:r + buff + 1, c - buff:c + buff + 1] += where_dem
        
        dem_exp = dem_exp[:, buff:dem_exp.shape[1]-buff,buff:dem_exp.shape[2]-buff]
        
        dem_exp[dem_exp > 0] = 1
        
        # dem_exp[dem_exp <= 0] = np.nan # transparancy
    # endregion
    
    # region Data for maps
    
    # Original
    inf_grid_dens = np.full((len(tp_maps), nrows, ncols), np.nan)  # nan for transparency
    removed_grid = np.full((len(tp_maps), nrows, ncols), np.nan)  # nan for transparency
    
    # Compacted
    densInf_comp = np.full((len(tp_maps), nrows, ncols), np.nan)
    propRem_comp = np.full((len(tp_maps), nrows, ncols), np.nan)
    
    grid_cells = plot_dict["cells"].reshape(max(plot_dict["rows_all"])+1,max(plot_dict["cols_all"])+1)
    plot_grid = -np.ones((nrows, ncols)) # basemap
    

    # fill frames (original)
    for i, tp in enumerate(tp_maps):

        for j, (r, c) in enumerate(zip(plot_dict["rows_cit"], plot_dict["cols_cit"])):
            inf_grid_dens[i, r, c] = densInfected[tp, j]

            if propRemoved[tp, j] == 1:  # mark removed units
                removed_grid[i, r, c] = 1
                
                
    inf_grid_dens[inf_grid_dens == 0] = np.nan
    removed_grid[removed_grid == 0] = np.nan   
    inf_grid_dens *= 100 # %
    
    # fill frames (compacted)
    for ic in range(nrows_comp): # rows
        i_start = ic * space_step
        i_end = (ic+1) * space_step
        
        for jc in range(ncols_comp): # cols
            j_start = jc * space_step
            j_end = (jc+1) * space_step
            
            cells = grid_cells[i_start:i_end, j_start:j_end]
            if (cells == -2).all():
                plot_grid[i_start:i_end, j_start:j_end] = -2
                
            cells = cells[cells >= 0]
            
            if cells.size:
                plot_grid[i_start:i_end, j_start:j_end] = 0
                
                for i, tp in enumerate(tp_maps):
                
                    densInf_comp[i, i_start:i_end, j_start:j_end] = np.nansum(densInfected[tp, cells]) / (space_step ** 2)
                    propRem_comp[i, i_start:i_end, j_start:j_end] = np.all(propRemoved[tp, cells] == 1)
                
    densInf_comp[densInf_comp == 0] = np.nan # set as transparent
    densInf_comp *= 100 #%
    propRem_comp[propRem_comp == 0] = np.nan
    if np.nanmax(propRem_comp) == 1: # if ever fully removed
        removed = 1
        
    # Define base colormap (for water and non-citrus land)
    colors_base = ["#ffffff", "silver", "#7dc77d", "black"] 
    base_cmap = mcolors.ListedColormap(colors_base[:3])
    base_bounds = [-2.5, -1.5, -0.5, 0.5]  # boundaries for mapping colors
    base_norm = mcolors.BoundaryNorm(base_bounds, base_cmap.N)

    # Define custom colors
    marker_col = "red"
    colors_dens = ["#f5d93b"] # KEEP FOR LEGEND
    custom_cmap2 = ListedColormap(["#f5d93b", "#f5d93b"])
    # colors_dens = ["#f5d93b", "#e74c3c"] # compacted
    # cmap_dens = LinearSegmentedColormap.from_list("custom_cmap", colors_dens, N=256)
    colors_dens_og = ["#925afa", "#87f7ed"] # original
    cmap_dens_og = LinearSegmentedColormap.from_list("custom_cmap", colors_dens_og, N=256)
    
    custom_cmap3 = ListedColormap(["#e74c3c", "#e74c3c"]) # demarcated (not used, now contoured)

    # Find origin cell (with rotation)
    cell_origin = np.where(hlb_units[0,:] > 0)[0]
    row_origin = plot_dict["rows_cit"][cell_origin]
    col_origin = plot_dict["cols_cit"][cell_origin]

    # Scale bar
    scale_cells = np.max(plot_dict["cols_all"] + 1) / 5
    scale_meters = scale_cells * 1000 * plot_dict["res"]

    if scale_meters < 1000: # switch to [m]
        candidates = np.array([100, 200, 500, 1000])
        idx = (np.abs(candidates - scale_meters)).argmin()
        scale_label = f"{candidates[idx]} m"
        scale_meters = candidates[idx]
        scale_cells = scale_meters / (1000 * plot_dict["res"])
    else:
        candidates = np.concatenate([np.array([1000, 2000]), np.arange(5000, np.max(plot_dict["cols_all"] + 1) * 1000 * plot_dict["res"] + 1000, 5000)])
        candidates = np.concatenate([np.array([1000, 2000]), np.arange(5000, 10000 + 1000, 5000)])
        idx = (np.abs(candidates - scale_meters)).argmin()
        scale_label = f"{int(candidates[idx] / 1000)} km"
        scale_meters = candidates[idx]
        scale_cells = scale_meters / (1000 * plot_dict["res"])
        
    # endregion
            
    if len(plot_years) == 5:

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(fig_width, fig_width*(2/3)*1.15), facecolor = "none")
        short = 1
    elif len(plot_years) == 8:

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(fig_width, fig_width*1.2), facecolor = "none")
        short = 0
    else:
        raise ValueError("Number of years to plot not supported")
    
    
    # region DPC
    ax = axes[-1,-1]
    line_cells, = ax.plot(x_axis[:end], inc_cells[:end], lw = 2, color = "dodgerblue", label="Affected cells")
    line_units, = ax.plot(x_axis[:end], inc_units[:end], lw = 2, color = "violet", label="Affected units")
    if removed:
        line_hlb, = ax.plot(x_axis[:end], hlb_only_inc[:end], lw = 2, color = "crimson", label="Infected units", ls=(0,(1,1)))
    ax.set_xlim(0,nYears)
    ax.set_xticks(x_ticks)
    ax.set_xticks(x_minor, minor = True)
    ax.set_xticklabels(x_ticks)
    
    if sym_ax:
        start_at = 1 # linear starts at 1%
        low_bound_exp = np.floor(np.log10(100/tot_cit))
        n = abs(low_bound_exp)
        low_bound = 10 ** low_bound_exp
        
        space_y = 20 # % of y axis for log part
        def forward(x):
            x_safe = np.clip(x, low_bound, 100)
        
            return np.where(x_safe <= start_at, 
                            (np.log10(x_safe) + n) * (space_y / n), 
                            space_y + (x_safe - 1) * ((100 - space_y) / (100 - start_at)))

        def inverse(x):
            return np.where(x <= space_y, 
                            10**((x / (space_y / n)) - n), 
                            1 + (x - space_y) * ((100 - start_at) / (100 - space_y)))
            
        def hybrid_formatter(x, pos):
            # For values < 1, use 10^(exponent) format
            if x < start_at and x > 0:
                exponent = int(np.round(np.log10(x)))
                return f"$10^{{{exponent}}}$"
            # For values >= 1, use standard integers
            else:
                return f"{int(x)}"

        ax.set_yscale('function', functions=(forward, inverse))
        arr_log = np.arange(0, low_bound_exp-1, -2, dtype=int)[::-1]
        yticks = np.concatenate([10 ** (arr_log).astype(float), np.arange(20,120,20)])
        ax.set_yticks(yticks)
        ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(hybrid_formatter))
    else:
        yticks = np.arange(0,120,20)
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.arange(0,120,20))
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Incidence [%]", fontsize = fs_labels, labelpad=-8)

    ax.set_xlabel("Time [years]", fontsize = fs_labels, labelpad=0)
    leg = ax.legend(loc="upper left", fontsize = fs_legend, handlelength=1.5)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    ax.tick_params(axis='both', labelsize=fs_ticks, direction="inout")
    ax.tick_params(axis='x', direction="inout", which='minor')
    ax.tick_params(axis='y', pad = -0.)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f"({letters[len(plot_years)]}) DPC", fontsize=fs_title)
    ax.set_facecolor("none")
    ax.grid(visible=True, which='major', axis='both', ls=':', lw=0.5, color="black", alpha=0.5)
    
    # endregion
    
    # region MAPS
    axes_flat = axes.flatten()
    for i in range(len(tp_maps)):
        ax = axes_flat[i]
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor='none'
        ax.set_anchor('C')
        
        # basemap
        ax.imshow(plot_grid, cmap=base_cmap, norm=base_norm, interpolation="nearest")
        
        # data
        # im_dens_comp = ax.imshow(densInf_comp[i,:,:], cmap=cmap_dens, vmin = 0, vmax=1, interpolation="nearest")
        im_dens_comp = ax.imshow(densInf_comp[i,:,:], cmap=custom_cmap2, vmin = 0, vmax=100, interpolation="nearest", aspect="equal")
        im_rem_comp = ax.imshow(propRem_comp[i,:,:], cmap=mcolors.ListedColormap([colors_base[3]]), alpha=0.7, interpolation="nearest", aspect="equal")
        im_dens_og = ax.imshow(inf_grid_dens[i,:,:], cmap=cmap_dens_og, vmin = 0, vmax=100, interpolation="nearest", aspect="equal")
        im_rem_og = ax.imshow(removed_grid[i,:,:], cmap=mcolors.ListedColormap([colors_base[3]]), alpha=0.7, interpolation="nearest", aspect="equal")
        
        # Demarcated area
        if demarcate:
            dem_im = ax.contour(dem_exp[i,:,:], levels=[0.5], colors="tomato", linewidths=1)
    
        scalebar = AnchoredHScaleBar(ax, size=scale_cells, label=scale_label, loc="upper left",
                                        pad=0.2,
                                        linekw=dict(color="black", linewidth = 0.8), text_color = "black", fontsize = fs_scalebar)
        ax.add_artist(scalebar)
        ax.set_title(f"({letters[i]}) {label_time} {tp_maps[i]/spyr:g}", fontsize = fs_title)
        ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
        
        outbreak_origin, = ax.plot(col_origin, row_origin, marker="x", color=marker_col,markersize=4, linestyle='None', label='Initial infection')
        
    # endregion

    # Colorbar & legend maps
    fig.subplots_adjust(bottom=0.27, top = None)
    
    pos_bottom_left = axes[-1, 0].get_position()
    pos_bottom_mid  = axes[-1,1].get_position()
    pos_bottom_right = axes[-1, 2].get_position()
    mid_x = (pos_bottom_mid.x0 + pos_bottom_mid.x1) / 2 # middle of the figure
    
    water_patch = mpatches.Patch(facecolor=colors_base[0], edgecolor="black", label="Water")
    noCit_patch = mpatches.Patch(facecolor=colors_base[1], edgecolor="black", label="Empty")
    citrus_patch = mpatches.Patch(facecolor=colors_base[2], edgecolor="black", label="Susceptible")
    infected_patch = mpatches.Patch(facecolor=colors_dens[0], edgecolor="black", label="HLB-affected (1 km²)")
    removed_patch = mpatches.Patch(facecolor=colors_base[3], edgecolor="black", label="Removed")
    dem_patch = mpatches.Patch(facecolor="none", edgecolor="tomato", label="Demarcated area")

    # Compacted
    bar_width = pos_bottom_left.x1 - pos_bottom_left.x0
    cb_ticks = np.arange(0, 120, 20)
    
    # Fix trajectories plot
    if region == "Sevilla":
        # Adjust row 0 and 1
        fig.subplots_adjust(hspace=-0.5)
        y_shift = 0.05  # Amount to shift Row 0 down (adjust to taste)
        for col in range(3):
            pos = axes[0, col].get_position()
            axes[0, col].set_position([pos.x0, pos.y0 - y_shift, pos.width, pos.height])
            
        fig_w, fig_h = fig.get_size_inches()
        aspect_ratio = fig_w / fig_h
        
        pos_col = axes[0, -1].get_position()
        pos_row = axes[-1, 0].get_position()
        new_width = pos_col.width * 0.8
        new_height = new_width * aspect_ratio

        x_offset = (pos_col.width - new_width) / 2
        new_x0 = pos_col.x0 + x_offset
        new_y0 = pos_row.y0 - x_offset*2

        axes[-1, -1].set_position([new_x0, new_y0, new_width, new_height])
        
        pos_bot_right = axes[-1,-1].get_position()
        pos_right = axes[-1,-1].get_position()

    else:
        axes[-1, -1].set_box_aspect(1)
        
        pos_bot_right = axes[-1,-1].get_position()
        pos_right = axes[0,-1].get_position()

    if short:

        cax_dens2 = fig.add_axes([pos_right.x0 - 0.01, pos_bot_right.y0 - 0.14, pos_bot_right.width * 0.9, 0.011])  # Position for shared colorbar  
        cbar_dens2 = fig.colorbar(im_dens_og, cax=cax_dens2, orientation="horizontal", extend="neither", ticks = cb_ticks)
        cbar_dens2.set_label(r"Infected density $\left[ \dfrac{\text{ha}_{\text{units}}}{\text{ha}} \% \right]$", fontsize = fs_labels - 1,labelpad=8)
        cbar_dens2.ax.tick_params(labelsize= fs_ticks, direction="inout")
        cbar_dens2.ax.xaxis.set_label_position('top')  # move the label position to top
        # cbar_dens2.ax.yaxis.set_label_position('left')  # move the label position to top
        
        cbar_pos = cax_dens2.get_position()
        if demarcate:
            if removed:
                leg1_handles = [outbreak_origin, water_patch, noCit_patch, citrus_patch]
                leg2_handles = [infected_patch, removed_patch, dem_patch]
                leg1 = fig.legend(handles=leg1_handles, loc="lower left", ncol = 4, bbox_to_anchor=(pos_bottom_left.x0, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
                leg2 = fig.legend(handles=leg2_handles, loc="lower left", ncol=3,
                    bbox_to_anchor=(pos_bottom_left.x0, cbar_pos.y0 - 0.03), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
                ax.add_artist(leg1)

                
            else:
                
                leg_handles = [outbreak_origin, citrus_patch, water_patch, infected_patch, noCit_patch, dem_patch]
                leg = fig.legend(handles=leg_handles, loc="lower left", ncol = 3, bbox_to_anchor=(pos_bottom_left.x0 + 0.01, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)

        else:
            if removed:
                leg_handles = [outbreak_origin, citrus_patch, water_patch, infected_patch, noCit_patch, removed_patch]
                leg = fig.legend(handles=leg_handles, loc="lower left", ncol = 3, bbox_to_anchor=(pos_bottom_left.x0 + 0.01, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
            else:
                leg1_handles = [outbreak_origin, water_patch, noCit_patch]
                leg2_handles = [citrus_patch, infected_patch]
                leg1 = fig.legend(handles=leg1_handles, loc="lower left", ncol = 4, bbox_to_anchor=(pos_bottom_left.x0 + 0.06, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
                leg2 = fig.legend(handles=leg2_handles, loc="lower left", ncol=3,
                    bbox_to_anchor=(pos_bottom_left.x0 + 0.06, cbar_pos.y0 - 0.03), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=1, frameon=False, columnspacing=0.5)
                ax.add_artist(leg1)
        
        # Add legend rectangle
        pos_left = axes[0,0].get_position()
        pos_right = axes[0,-1].get_position()
        bg_rect = mpatches.Rectangle(
            (pos_left.x0, pos_bot_right.y0 - 0.18), pos_right.x1 - pos_left.x0, 0.115,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor='black',
            linewidth=1,
            alpha=1,
            zorder=-1
        )
        
        
        fig.add_artist(bg_rect)
    else:
        
        cax_dens2 = fig.add_axes([pos_right.x0 - 0.01, pos_bot_right.y0 - 0.09, pos_bot_right.width * 0.9, 0.007])  # Position for shared colorbar  
        cbar_dens2 = fig.colorbar(im_dens_og, cax=cax_dens2, orientation="horizontal", extend="neither", ticks = cb_ticks)
        cbar_dens2.set_label(r"Infected density $\left[ \dfrac{\text{ha}_{\text{units}}}{\text{ha}} \% \right]$", fontsize = fs_labels - 1,labelpad=8)
        cbar_dens2.ax.tick_params(labelsize= fs_ticks, direction="inout")
        cbar_dens2.ax.xaxis.set_label_position('top')  # move the label position to top
        # cbar_dens2.ax.yaxis.set_label_position('left')  # move the label position to top
        
        cbar_pos = cax_dens2.get_position()
        if demarcate:
            if removed:
                leg1_handles = [outbreak_origin, water_patch, noCit_patch, citrus_patch]
                leg2_handles = [infected_patch, removed_patch, dem_patch]
                leg1 = fig.legend(handles=leg1_handles, loc="lower left", ncol = 4, bbox_to_anchor=(pos_bottom_left.x0, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
                leg2 = fig.legend(handles=leg2_handles, loc="lower left", ncol=3,
                    bbox_to_anchor=(pos_bottom_left.x0, cbar_pos.y0 - 0.02), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
                ax.add_artist(leg1)

            else:

                leg_handles = [outbreak_origin, citrus_patch, water_patch, infected_patch, noCit_patch, dem_patch]
                leg = fig.legend(handles=leg_handles, loc="lower left", ncol = 3, bbox_to_anchor=(pos_bottom_left.x0 + 0.01, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
        else:
            if removed:
                leg_handles = [outbreak_origin, citrus_patch, water_patch, infected_patch, noCit_patch, removed_patch]
                leg = fig.legend(handles=leg_handles, loc="lower left", ncol = 3, bbox_to_anchor=(pos_bottom_left.x0 + 0.01, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
            else:
                leg1_handles = [outbreak_origin, water_patch, noCit_patch]
                leg2_handles = [citrus_patch, infected_patch]
                leg1 = fig.legend(handles=leg1_handles, loc="lower left", ncol = 4, bbox_to_anchor=(pos_bottom_left.x0 + 0.06, cbar_pos.y0), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=0.6, frameon=False, columnspacing=0.5)
                leg2 = fig.legend(handles=leg2_handles, loc="lower left", ncol=3,
                    bbox_to_anchor=(pos_bottom_left.x0 + 0.06, cbar_pos.y0-0.02), fontsize=fs_labels - 1, title_fontsize=12, handlelength=1, handleheight=1, labelspacing=1, frameon=False, columnspacing=0.5)
                ax.add_artist(leg1)
        
        # Add legend rectangle
        pos_left = axes[0,0].get_position()
        pos_right = axes[0,-1].get_position()
        bg_rect = mpatches.Rectangle(
            (pos_left.x0, pos_bot_right.y0 - 0.114), pos_right.x1 - pos_left.x0, 0.07,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor='black',
            linewidth=1,
            alpha=1,
            zorder=-1
        )
        fig.add_artist(bg_rect)
    
    fig.savefig(f'{folder}\\maps_{identifier}.pdf', dpi=300, bbox_inches="tight")
    

