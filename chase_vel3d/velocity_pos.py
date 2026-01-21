"""
Plane-of-Sky (POS) velocity calculation module.

This module computes POS velocities by flct method.
"""

import numpy as np
import scipy.ndimage as ndi
from pyflct import flct

from .utils import get_solar_center, get_obstime, get_arcsec_per_pix
from .coords import roi_center_pix_to_slices, grid_from_header

### POS velocity
KM_PER_ARCSEC = 725.0    # km/arcsec at solar center

def compute_absortion_proxy(
    Rint,
    smooth: int = 2,
    bg: int = 20,
    ):
    '''
    Compute absorption proxy from Rint array.
    
    Parameters
    ----------
    Rint : np.ndarray
        2D array of intensity ratios
    smooth : int, default=2
        Smoothing kernel size for Gaussian filter
    bg : int, default=20
        Background subtraction kernel size for Gaussian filter
    Returns
    -------
    G : np.ndarray
        2D array of absorption proxy
    '''
    eps = 1e-6
    valid = np.isfinite(Rint)
    A = np.full_like(Rint, np.nan, dtype=np.float32)
    A[valid] = - np.log(np.clip(Rint[valid], a_min=eps, a_max=None))
    
    A0 = np.nan_to_num(A, nan=0.0)

    As = ndi.gaussian_filter(A0, smooth)
    Ahp = As - ndi.gaussian_filter(As, bg)

    gx = ndi.sobel(Ahp, axis=1)
    gy = ndi.sobel(Ahp, axis=0)
    G = np.hypot(gx, gy)
    G = ndi.gaussian_filter(G, smooth // 2)    
    return G


def compute_pos_v(
    G1, 
    G2,
    hdr1, 
    hdr2,
    roi_xy,
    sigma: int = 4,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
):  
    '''
    Compute Plane-of-Sky (POS) velocity using FLCT method.
    
    Parameters
    ----------
    G1 : np.ndarray
        2D absorption proxy array at time 1
        G2 : np.ndarray
        2D absorption proxy array at time 2
    hdr1 : astropy.io.fits.Header
        FITS header for time 1
    hdr2 : astropy.io.fits.Header
        FITS header for time 2
    roi_xy : tuple
        Region of interest in (x0, x1, y0, y1) format
    sigma : int, default=4
        FLCT Gaussian windowing parameter
    arcsec_per_pix : float, default=1.0436
        Arcseconds per pixel scale
    km_per_arcsec : float, default=725.0
        Kilometers per arcsecond scale
    Returns
    -------
    vx : np.ndarray
        2D array of POS velocity in x-direction (km/s)
    vy : np.ndarray 
        2D array of POS velocity in y-direction (km/s)
    vm : np.ndarray
        2D array of POS velocity magnitude (km/s)
    '''
    arcsec_per_pix = get_arcsec_per_pix(hdr1)
    km_per_pix = arcsec_per_pix * KM_PER_ARCSEC

    # Use consistent ROI slicing with roi_center_pix_to_slices
    grid = grid_from_header(hdr1)
    x0, x1, y0, y1 = roi_xy
    ys, xs = roi_center_pix_to_slices(grid, x0, x1, y0, y1)

    # Extract ROI from absorption proxy maps
    G1_roi = G1[ys, xs]
    G2_roi = G2[ys, xs]

    t1 = get_obstime(hdr1)
    t2 = get_obstime(hdr2)
    deltat = abs((t2 - t1).to_value('s'))

    vx, vy, vm = flct(G1_roi, G2_roi, deltat=deltat, deltas=km_per_pix, sigma=sigma)

    if mask1 is not None and mask2 is not None:
        mask1_roi = mask1[ys, xs]
        mask2_roi = mask2[ys, xs]
        mask_roi = mask1_roi & mask2_roi
        vx = np.where(mask_roi, vx, np.nan)
        vy = np.where(mask_roi, vy, np.nan)
        vm = np.where(mask_roi, vm, np.nan)

    return vx, vy, vm


def plot_pos_vmap(
    vx,
    vy,
    vm,
    vmin_percentile: float = 10.0,
    vmax_percentile: float = 95.0,
    step: int = 5,
):
    '''
    Prepare POS velocity map for quiver plotting.
    Parameters
    ----------
    vx : np.ndarray
        2D array of POS velocity in x-direction (km/s)
    vy : np.ndarray
        2D array of POS velocity in y-direction (km/s)
    vm : np.ndarray
        2D array of POS velocity magnitude (km/s)
    vmin_percentile : float, default=10.0
        Minimum percentile for velocity magnitude filtering
    vmax_percentile : float, default=95.0
        Maximum percentile for velocity magnitude filtering
    step : int, default=5
        Step size for downsampling the velocity field
    Returns
    -------
    X : np.ndarray
        2D array of x-coordinates for quiver plot
    Y : np.ndarray
        2D array of y-coordinates for quiver plot
    vxq : np.ndarray
        2D array of downsampled x-velocity components for quiver plot
    vyq : np.ndarray
        2D array of downsampled y-velocity components for quiver plot
    '''

    Y, X = np.mgrid[0:vx.shape[0]:step, 0:vx.shape[1]:step]

    vxq = vx[0:vx.shape[0]:step, 0:vx.shape[1]:step]
    vyq = vy[0:vx.shape[0]:step, 0:vx.shape[1]:step]

    vmod = np.sqrt(vx**2 + vy**2)
    vmin, vmax = np.nanpercentile(vmod, [vmin_percentile, vmax_percentile])
    good = (vmod >= vmin) & (vmod <= vmax)

    return X[good[0:vx.shape[0]:step, 0:vx.shape[1]:step]], Y[good[0:vx.shape[0]:step, 0:vx.shape[1]:step]], vxq[good[0:vx.shape[0]:step, 0:vx.shape[1]:step]], vyq[good[0:vx.shape[0]:step, 0:vx.shape[1]:step]]

