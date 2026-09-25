'''
    File name: spatial_masks.py
    Author: Laura Erbetta
    Date created: 23/03/2025
    Date last modified: 06/11/2025
    Python Version: 3.13
'''
# %%
import numpy as np
import math

# %%
# region Dispersal kernels

def create_dispersal_masks(res, md_dist, sd_dist_vec, sd_dist_pat, SD_scale, propWithin, test = False):
    """
    Returns 1/8 of symmetrical masks of dispersal kernels (short- and mid-distance), splitting at [sd_dist_vec] or [sd_dist_pat] and up to [md_dist], and corresponding relative coordinates from central cell.
    Output contains:
        mask_pat_sd (1d np.float64 array, N):
            - short-distance pathogen kernel, only 1/8 slice (considering overlapping removal due to slices symmetry)
        mask_vec_sd (1d np.float64 array, N):
            - short-distance vector kernel, only 1/8 slice (considering overlapping removal due to slices symmetry)
        pat_md_cs (1d np.float64 array, M):
            - mid-distance pathogen cumulative kernel, only 1/8 slice (considering overlapping removal due to slices symmetry)
        vec_md_cs (1d np.float64 array, M):
            - mid-distance vector cumulative kernel, only 1/8 slice (considering overlapping removal due to slices symmetry)
        coord_sd (2D np.uint8 array, Nx2): relative coordinates for short-distance masks
            - First col: relative row distance from central cell
            - Second col: relative column distance from central cell
        coord_md (2D np.uint8 array, Mx2): relative coordinates for mid-distance masks
            - First col: relative row distance from central cell
            - Second col: relative column distance from central cell
    where N is the amount of cells with > 0 dispersal in the slice
    M is the amount of cells with > 0 dispersal in the full mask
    Used for:
    - Short-distance dispersal
    - Mid-distance dispersal
    """
    if math.ceil(md_dist / res) > np.iinfo('int16').max:
        # Check that coordinates will be within np.uint8 range
        raise ValueError(f"Max mid-distance dispersal {md_dist} not supported. Exceeds coordinates data type.")
    elif md_dist < sd_dist_vec or md_dist < sd_dist_pat:
        raise ValueError(f"Mid-distance dispersal cannot be lower than short-distance dispersal.")
    
    buff = math.ceil(md_dist / res)

    # Precompute relative offsets & distance mask
    row_offsets = np.arange(-buff, buff+1)
    col_offsets = np.arange(-buff, buff+1)
    row_grid, col_grid = np.meshgrid(row_offsets, col_offsets, indexing='ij')
    dist_mask = np.sqrt((row_grid * res) ** 2 + (col_grid * res) ** 2)
    kernel_mask = np.exp(-dist_mask / SD_scale)
    
    keep_values = np.round(dist_mask - md_dist, 10) <= 0
    kernel_mask[~keep_values] = 0
    
    # Normalize and change within-cell infection for pathogen dispersal kernel
    vec_mask = kernel_mask / kernel_mask.sum()
    if propWithin > 0:
        pat_mask = kernel_mask / (kernel_mask.sum()-1) if (kernel_mask.sum()-1) > 0 else kernel_mask
    
        pat_mask = pat_mask * (1-propWithin)
        pat_mask[dist_mask == 0] = propWithin # central cell
    else:
        pat_mask = kernel_mask / kernel_mask.sum()
    
    
    if test:
        return vec_mask, pat_mask
    
    coord = np.column_stack([row_grid.flatten(), col_grid.flatten()])
    
    # >>> SHORT-DISTANCE <<< full mask
    where_sd_pat = (np.round(dist_mask - sd_dist_pat, 10) <= 0)
    where_sd_vec = (np.round(dist_mask - sd_dist_vec, 10) <= 0)
    coord_sd_pat = coord[where_sd_pat.flatten(), :].astype(np.int16)
    coord_sd_vec = coord[where_sd_vec.flatten(), :].astype(np.int16)
    mask_pat_sd = pat_mask[where_sd_pat].flatten()
    mask_vec_sd = vec_mask[where_sd_vec].flatten()
    
    # >>> MID-DISTANCE <<< full mask
    # Pathogen
    if md_dist > sd_dist_pat:
        where_md = (~where_sd_pat) * (pat_mask > 0) 
        coord_md_pat = coord[where_md.flatten(), :].astype(np.int16)
        
        pat_mask_md = pat_mask[where_md].flatten()

        pat_md_cs = pat_mask_md.cumsum()
    else:
        coord_md_pat = np.zeros((0,0), dtype=np.int16)
        pat_md_cs = np.zeros(0, dtype=np.float64)
        
    # Vector
    if md_dist > sd_dist_pat:
        where_md = (~where_sd_vec) * (vec_mask > 0) 
        coord_md_vec = coord[where_md.flatten(), :].astype(np.int16)
        
        vec_mask_md = vec_mask[where_md].flatten()

        vec_md_cs = vec_mask_md.cumsum()
    else:
        coord_md_vec = np.zeros((0,0), dtype=np.int16)
        vec_md_cs = np.zeros(0, dtype=np.float64)
    
    return mask_pat_sd, mask_vec_sd, coord_sd_pat, coord_sd_vec, pat_md_cs, vec_md_cs, coord_md_pat, coord_md_vec
    
# endregion

# region Control masks

def compute_sqrt_semicircle(r, x0, x1, x2):
    sqrt_x1 = r ** 2 - x1 ** 2 - x0 ** 2 + 2 * x0 * x1
    sqrt_x2 = r ** 2 - x2 ** 2 - x0 ** 2 + 2 * x0 * x2
    # Handle approximation errors
    if 0 > sqrt_x1 > -1e-10:
        sqrt_x1 = 0
    if 0 > sqrt_x2 > -1e-10:
        sqrt_x2 = 0

    return sqrt_x1, sqrt_x2

def integral_semi(r, x0, y0, x1, x2, y1, y2):
    """Compute the integral of a semicircle within a cell"""
    
    with np.errstate(divide='ignore'):
        if y1 * y2 < 0:                     # cells on center cell's row (semicircle defined only for positive values)
            # Integration extremes
            aX = max(x0 - r, x1)
            bX = min(x0 + r, x2)
            # Check for approximation errors
            sqrt_aX, sqrt_bX = compute_sqrt_semicircle(r, x0, aX, bX)
            # Compute positive and negative semicircle area
            Acell = ((bX - x0) * math.sqrt(sqrt_bX) + (r ** 2) * math.atan(np.float64(bX - x0) / math.sqrt(sqrt_bX))) - (
                            (aX - x0) * math.sqrt(sqrt_aX) + (r ** 2) * math.atan(np.float64(aX - x0) / math.sqrt(sqrt_aX)))
            return Acell
        
        elif y2 < 0:                        # cells below central cell
            # Integration extremes
            x2 = max(min(x0 + math.sqrt(r ** 2 - max(y0 - y2, 0) ** 2), x2), x1)
            x1 = min(max(x0 - math.sqrt(r ** 2 - max(y0 - y2, 0) ** 2), x1), x2)
            # Check for approximation errors
            sqrt_x1, sqrt_x2 = compute_sqrt_semicircle(r, x0, x1, x2)
            # Compute integrals
            int_x1 = +y2 * x1 - y0 * x1 + 0.5 * (
                    (x1 - x0) * math.sqrt(sqrt_x1) + (r ** 2) * math.atan(np.float64(x1 - x0) / math.sqrt(sqrt_x1)))
            int_x2 = +y2 * x2 - y0 * x2 + 0.5 * (
                    (x2 - x0) * math.sqrt(sqrt_x2) + (r ** 2) * math.atan(np.float64(x2 - x0) / math.sqrt(sqrt_x2)))
            
        else:                               # cells above central cell
            # Integration extremes
            x2 = max(min(x0 + math.sqrt(r ** 2 - max(y1 - y0, 0) ** 2), x2), x1)
            x1 = min(max(x0 - math.sqrt(r ** 2 - max(y1 - y0, 0) ** 2), x1), x2)
            # Check for approximation errors
            sqrt_x1, sqrt_x2 = compute_sqrt_semicircle(r, x0, x1, x2)
            # Compute integrals
            int_x1 = -y1 * x1 + y0 * x1 + 0.5 * (
                    (x1 - x0) * math.sqrt(sqrt_x1) + (r ** 2) * math.atan(np.float64(x1 - x0) / math.sqrt(sqrt_x1)))
            int_x2 = -y1 * x2 + y0 * x2 + 0.5 * (
                    (x2 - x0) * math.sqrt(sqrt_x2) + (r ** 2) * math.atan(np.float64(x2 - x0) / math.sqrt(sqrt_x2)))

    return int_x2 - int_x1

def create_control_mask(R, res, x0_input=None, y0_input=None):
    """ Create removal mask based on numerical integration of a semicircle """
    
    # Centers
    if x0_input is None or y0_input is None:
        # compute N times to capture variability in center's position
        N = 100
        x0s = np.random.uniform(-res/2, res/2, N)
        y0s = np.random.uniform(-res/2, res/2, N)
    else:
        N = 1
        x0s = np.array([x0_input])
        y0s = np.array([y0_input])

    M = round((np.ceil((R) / res)) * 2 + 1)           # size mask matrix (MxM)
    xticks = np.arange(-M * res / 2, M * res / 2 + res / 2, res)    # coordinates of cells' borders from center
    yticks = np.arange(M * res / 2, -M * res / 2 - res/2, -res).reshape(-1, 1)

    results = np.zeros((M, M))
    posC = math.floor(M / 2)            # idx position of central cell (where the removed host is)
    upperY_mat = np.hstack([yticks[:yticks.size-1]] * M)
    lowerY_mat = np.hstack([yticks[1:]] * M)
    leftX_mat = np.vstack([xticks[:yticks.size-1]] * M)
    rightX_mat = np.vstack([xticks[1:]] * M)

    for n in range(N):
        
        x0 = x0s[n]
        y0 = y0s[n]

        portion_grid = np.zeros((M, M))
        dy_max = np.maximum(abs(upperY_mat - y0), abs(lowerY_mat - y0))
        dy_min = np.minimum(abs(upperY_mat - y0), abs(lowerY_mat - y0))
        dx_max = np.maximum(abs(leftX_mat - x0), abs(rightX_mat - x0))
        dx_min = np.minimum(abs(leftX_mat - x0), abs(rightX_mat - x0))
        
        d_max = np.sqrt(dx_max**2 + dy_max**2)
        d_min = np.sqrt(dx_min**2 + dy_min**2)
        
        d_max[posC,:] = dx_max[posC,:]
        d_min[posC,:] = dx_min[posC,:]
        d_max[:,posC] = dy_max[:,posC]
        d_min[:,posC] = dy_min[:,posC]
        d_max[posC,posC] = max(dx_max[posC,posC], dy_max[posC,posC])
        d_min[posC,posC] = 0
        
        d_min.round(10)
        d_max.round(10)
        
        where_row, where_col = np.where((d_max >= R) & (d_min < R))
        
        for k in range(len(where_row)):
            i = where_row[k]
            j = where_col[k]
            rightX = rightX_mat[i, j]      # vertical limits of current cell
            leftX = leftX_mat[i, j]
            upperY = upperY_mat[i, j]      # horizontal limits of current cell
            lowerY = lowerY_mat[i, j]
            
            portion_grid[i, j] = integral_semi(R, x0, y0, leftX, rightX, lowerY, upperY) / (res ** 2)
            
        for i in np.arange(1, math.ceil(M / 2) - 1):
            
            portion_grid[i,:] -= np.sum(portion_grid[:i,:], axis=0)
            
        for i in np.arange(math.ceil(M / 2), M-1)[::-1]:
            
            portion_grid[i,:] -= np.sum(portion_grid[i+1:,:], axis=0)
            
    
        portion_grid[posC,:] -= np.sum(portion_grid[:posC:,:], axis=0) + np.sum(portion_grid[posC+1:,:], axis=0)
        
        portion_grid[d_max < R] = 1.0

        results += portion_grid

    removalMask = results / N

    return removalMask

def create_removal_mask(R, res, x0_input=None, y0_input=None):
    """
    Returns 1/8 of symmetrical masks of removal proportion with radius of removal [R] and corresponding relative coordinates from central cell.
    Output contains:
        mask (1d np.float64 array, N):
            - proportion of removal in each cell of the 1/8 slice (considering overlapping removal due to slices symmetry)
        out (2D np.uint8 array, Nx2):
            - First col: relative row distance from central cell
            - Second col: relative column distance from central cell
    where N is the amount of cells with > 0 removal in the slice
    Used for:
    - Removal
    """
    
    if R <= 0:
        return np.zeros((0), dtype=np.float64), np.zeros((0, 2), dtype=np.uint8)
    elif np.ceil(R / res) > np.iinfo('uint8').max:
        # Check that coordinates will be within np.uint8 range
        raise ValueError(f"Radius {R} not supported. Exceeds coordinates data type.")
    
    integral_mask = create_control_mask(R, res, x0_input, y0_input)
    
    # Extract 1/8
    c_pos = integral_mask.shape[0] // 2
    row_offsets = np.arange(0, c_pos+1)
    col_offsets = np.arange(0, c_pos+1)
    row_grid, col_grid = np.meshgrid(row_offsets, col_offsets, indexing='ij')
    
    sym_half = integral_mask[c_pos:, :] + np.flip(integral_mask[:c_pos+1,  :], axis = 0)
    sym_quarter = sym_half[:,c_pos:] + np.flip(sym_half[:, :c_pos+1], axis = 1)
    sym_octave = sym_quarter + np.fliplr(np.flipud(sym_quarter).T)
    
    keep_octave = (row_grid >= col_grid) * (sym_octave > 0)
    divide_four = (row_grid == col_grid) + (row_grid == 0) + (col_grid == 0)
    sym_octave[divide_four] /= 2
    sym_octave[0,0] /= 4

    mask = sym_octave[keep_octave] / 8
    out = np.column_stack((row_grid[keep_octave], col_grid[keep_octave])).astype(np.uint8)
    
    return mask, out

# endregion

# region Buffer masks

def create_buffer_mask(res, buff_dist, type_str):
    """
    Returns 1/8 of symmetrical masks with cells within [buff_dist] of a given cell.
    Output contains:
        out (2D np.uint8 array, Nx2):
            - First col: relative row distance from central cell
            - Second col: relative column distance from central cell
    where N is the amount of cells within distance in the slice
    Used for:
    - Search area
    - Demarcated area
    """
    if buff_dist < 0:
        if type_str == "demarcated":
            print(f"Demarcated area must include at least infected cells (i.e., infected area). Distance changed to 0 instead of {buff_dist}.")
            return np.array([[0, 0]], dtype=np.uint8)
        return np.zeros((0, 2), dtype=np.uint8)
    if buff_dist == 0:
        return np.array([[0, 0]], dtype=np.uint8)
    elif math.ceil(buff_dist / res) > np.iinfo('uint8').max:
        # Check that coordinates will be within np.uint8 range
        raise ValueError(f"Radius {buff_dist} not supported. Exceeds coordinates data type.")
    
    buff = math.ceil(buff_dist / res)

    # Precompute relative offsets & distance mask (top-right quadrant)
    row_offsets = np.arange(0, buff+1)
    col_offsets = np.arange(0, buff+1)
    row_grid, col_grid = np.meshgrid(row_offsets, col_offsets, indexing='ij')
    dist_mask = np.sqrt((row_grid * res) ** 2 + (col_grid * res) ** 2)
    keep_values_distance = np.round(dist_mask - buff_dist, 10) <= 0
    keep_octave = row_grid >= col_grid
    keep_values = keep_values_distance * keep_octave
    
    out = np.column_stack((row_grid[keep_values], col_grid[keep_values])).astype(np.uint8)

    return out

# endregion

# %%




 



