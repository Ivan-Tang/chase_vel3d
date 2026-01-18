"""
Point classification module for CHASE/RSM data.

This module classifies points in solar spectral data into three categories:
- Type 0: On Plate (absorption lines)
- Type 1: On Limb (prominence emission)
- Type 2: In Space (weak/no signal)
"""

import numpy as np
from scipy.ndimage import binary_opening, binary_closing, label as ndi_label
from skimage.morphology import disk, remove_small_objects
from .utils import get_solar_center, get_disk_mask, get_wavelength_axis

#### Prominence ####

def wave_pattern(spec: np.ndarray,
                 k0: int = 68,
                 cdelt3: float = 0.0242,
                 core_half_A: float = 0.6,
                 wing_frac: float = 0.15,
                 snr_th: float = 4.0):
    """
    Classify spectral pattern and determine point type.
    
    Uses absorption and emission signatures to classify pixels:
    - Type 0: On plate (strong absorption, D > E)
    - Type 1: On limb (strong emission, E > D)
    - Type 2: In space (weak signal, score < SNR threshold)
    
    Parameters
    ----------
    spec : np.ndarray
        1D spectrum array
    k0 : int, default=68
        Spectral index of line center
    cdelt3 : float, default=0.0242
        Wavelength scale (Å/pixel)
    core_half_A : float, default=0.6
        Half-width of spectral core window (Å)
    wing_frac : float, default=0.15
        Fraction of spectrum for wings (continuum/noise estimation)
    snr_th : float, default=4.0
        SNR threshold for significance
    
    Returns
    -------
    ptype : int
        Point type (0, 1, or 2)
    score : float
        Confidence score (higher = more significant)
    """
    spec = np.asarray(spec).astype(np.float32)
    n = spec.size
    if n < 10:
        return 2, 0.0
    
    # Estimate continuum and noise from wings
    wing = max(3, int(n * wing_frac))
    wings = np.concatenate([spec[:wing], spec[-wing:]])
    Ic = np.median(wings)
    mad = np.median(np.abs(wings - Ic))
    noise = 1.4826 * mad  # Robust sigma
    
    if not np.isfinite(Ic) or Ic <= 0 or not np.isfinite(noise) or noise <= 0:
        return 2, 0.0
    
    spec_n = spec / Ic
    sig_n = noise / Ic
    
    # Extract core window
    half_w = int(np.round(core_half_A / cdelt3))
    half_w = max(2, min(half_w, n // 2 - 1))
    L = max(0, k0 - half_w)
    R = min(n, k0 + half_w + 1)
    core = spec_n[L:R]
    
    Imin = float(np.min(core))
    Imax = float(np.max(core))
    
    # Calculate absorption and emission significance
    D = (1.0 - Imin) / sig_n  # Absorption (dip below continuum)
    E = (Imax - 1.0) / sig_n  # Emission (peak above continuum)
    score = float(max(D, E))
    
    # Classification
    if score < snr_th:
        return 2, score  # Weak signal -> space
    if E > D:
        return 1, score  # Strong emission -> limb
    return 0, score     # Strong absorption -> plate


def classify_region(rsm, left, right, bottom, top, ang_res=0.5218*2, 
                    snr_th=5.0):
    """
    Classify all pixels in a spatial region.
    
    Parameters
    ----------
    rsm : astropy.io.fits.HDUList
        FITS file object
    left, right, bottom, top : float
        Region boundaries in arcsec
    ang_res : float, default=0.5218*2
        Angular resolution (arcsec/pixel)
    snr_th : float, default=5.0
        SNR threshold for classification
    
    Returns
    -------
    type_mask : np.ndarray
        2D array of point types (0, 1, or 2)
    """
    crpix1 = rsm[1].header['CRPIX1']
    crpix2 = rsm[1].header['CRPIX2']
    crval1 = rsm[1].header.get('CRVAL1', 0)
    crval2 = rsm[1].header.get('CRVAL2', 0)
    cdelt1 = rsm[1].header.get('CDELT1', ang_res)
    cdelt2 = rsm[1].header.get('CDELT2', ang_res)
    
    n_cols = int((right - left) / ang_res) + 1
    n_rows = int((top - bottom) / ang_res) + 1
    type_mask = np.zeros((n_rows, n_cols), dtype=int)
    
    for row in range(n_rows):
        for col in range(n_cols):
            sky_x = left + col * ang_res
            sky_y = bottom + row * ang_res
            
            pix_col = int((sky_x - crval1) / cdelt1 + crpix1)
            pix_row = int((sky_y - crval2) / cdelt2 + crpix2)
            
            if (0 <= pix_row < rsm[1].data.shape[1] and 
                0 <= pix_col < rsm[1].data.shape[2]):
                spec = rsm[1].data[:, pix_row, pix_col]
                ptype = wave_pattern(spec, snr_th=snr_th)[0]
                type_mask[row, col] = ptype
    
    return type_mask


### Filament Extraction ###

def compute_Rint_map(
    I,
    hdr,
    line_half_width_A=1.5,
    wing_left=(6560.5, 6561.0),
    wing_right=(6564.6, 6565.0),
):
    """
    Compute normalized integrated intensity map Rint.
    
    Parameters
    ----------
    I : np.ndarray
        3D intensity array (nlambda, ny, nx)
    hdr : astropy.io.fits.Header
        FITS header with WCS information
    line_half_width_A : float, default=1.5
        Half-width of line integration window (Å)
    wing_left : tuple, default=(6560.5, 6561.0)
        Wavelength range for left wing (Å)
    wing_right : tuple, default=(6564.6, 6565.0)
        Wavelength range for right wing (Å)
    
    Returns
    -------
    Rint : np.ndarray
        2D normalized integrated intensity map (ny, nx)
    valid : np.ndarray
        2D boolean array of valid pixels used in Rint calculation
    """
    nlambda, ny, nx = I.shape
    wvl = get_wavelength_axis(hdr)

    disk_mask = get_disk_mask(hdr)

    idx_wing = (
        ((wvl >= wing_left[0]) & (wvl <= wing_left[1])) |
        ((wvl >= wing_right[0]) & (wvl <= wing_right[1]))
    )
    if idx_wing.sum() < 2:
        raise ValueError("Wing windows invalid for current wavelength range.")

    I_cont = np.nanmean(I[idx_wing, :, :], axis=0)

    valid = disk_mask & np.isfinite(I_cont) & (I_cont > 0)

    # 归一化谱
    R = np.full((nlambda, ny, nx), np.nan, dtype=np.float32)
    R[:, valid] = (I[:, valid] / I_cont[valid]).astype(np.float32)

    line_center_A = hdr['WAVE_LEN']
    lo = line_center_A - line_half_width_A
    hi = line_center_A + line_half_width_A
    idx_line = (wvl >= lo) & (wvl <= hi)

    if idx_line.sum() < 3:
        raise ValueError("Line integration window invalid for current wavelength range.")

    Rint = np.nanmean(R[idx_line, :, :], axis=0)  # (ny,nx)

    # 日面外强制无效
    Rint[~disk_mask] = np.nan
    return Rint, valid


def get_filament_mask(
    Rint,
    hdr,
    roi_xy,            # (x0,x1,y0,y1) filament search region
    bg_xy,             # (x0,x1,y0,y1) nearby quiet-sun background region (no filament, no flare)
    alpha=0.90,        # threshold ratio
    line_half_width_A=1.5,
    wing_left=(6560.5, 6561.0),
    wing_right=(6564.6, 6565.0),
    min_area=800,
    close_radius=2,
    binary_close=False,
    remove_small=True,
):
    """
    Extract filament mask from Rint map using adaptive thresholding.
    Parameters
    ----------
    Rint : np.ndarray
        2D normalized integrated intensity map
    hdr : astropy.io.fits.Header
        FITS header with WCS information
    roi_xy : tuple
        (x0, x1, y0, y1) coordinates of region of interest for filament search
    bg_xy : tuple
        (x0, x1, y0, y1) coordinates of background region for threshold estimation
    alpha : float, default=0.90
        Threshold ratio relative to background median
    line_half_width_A : float, default=1.5
        Half-width of line integration window (Å)
    wing_left : tuple, default=(6560.5, 6561.0)
        Wavelength range for left wing (Å)
    wing_right : tuple, default=(6564.6, 6565.0)
        Wavelength range for right wing (Å)
    min_area : int, default=800
        Minimum area of connected components to keep (pixels)
    close_radius : int, default=2
        Radius for morphological closing (pixels)
    binary_close : bool, default=False
        Whether to apply binary closing
    remove_small : bool, default=True
        Whether to remove small objects below min_area
    """

    # 取背景中位数
    center = get_solar_center(hdr)
    bx0, bx1, by0, by1 = bg_xy
    bx0, bx1 = int(bx0 + center[0]), int(bx1 + center[0])
    by0, by1 = int(by0 + center[1]), int(by1 + center[1])
    Rbg = Rint[by0:by1, bx0:bx1]
    Rbg = Rbg[np.isfinite(Rbg)]
    if Rbg.size < 200:
        raise RuntimeError("Too few valid background pixels. Move/resize bg_xy.")

    Rint_bg = float(np.nanmedian(Rbg))
    thr = alpha * Rint_bg

    # 在 ROI 内阈值化
    x0, x1, y0, y1 = roi_xy
    x0, x1 = int(x0 + center[0]), int(x1 + center[0])
    y0, y1 = int(y0 + center[1]), int(y1 + center[1])
    Rroi = Rint[y0:y1, x0:x1]
    mask_roi = (Rroi < thr) & np.isfinite(Rroi)

    # 形态学清理
    if binary_close:
        mask_roi = binary_closing(mask_roi, disk(close_radius))
    if remove_small:
        mask_roi = remove_small_objects(mask_roi, min_size=min_area)

    mask = np.zeros_like(Rint, dtype=bool)
    mask[y0:y1, x0:x1] = mask_roi

    line_center_A = hdr['WAVE_LEN']

    meta = {
        "Rint_bg_median": Rint_bg,
        "alpha": alpha,
        "threshold": thr,
        "line_window_A": (line_center_A - line_half_width_A, line_center_A + line_half_width_A),
        "wing_left": wing_left,
        "wing_right": wing_right,
        "filament_pixels_in_roi": int(mask_roi.sum()),
    }
    return mask, Rint, meta