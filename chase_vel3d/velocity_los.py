"""
Line-of-Sight (LOS) velocity calculation module.

This module computes LOS velocities using spectral analysis methods,
specifically the moment method for emission line analysis.
"""

import numpy as np
from scipy.optimize import curve_fit
import astropy.io.fits as fits

from .utils import get_wavelength_axis, get_solar_center, get_arcsec_per_pix

C_KMS = 299792.458  # km/s
DEFAULT_ANGRES_PER_PIXEL = 0.5218 * 2  # arcsec/pixel

# Moment method for emission lines

def calc_pixel_moment_velocity(spec: np.ndarray,
                            wvl: np.ndarray,
                            k0: int = 68,
                            core_half_A: float = 0.6,
                            wing_frac: float = 0.15,
                            min_weight_sum: float = 1e-6,
                            lambda0: float = 6562.8) -> float:
    """
    Calculate LOS velocity using moment method for emission lines.
    
    Method:
    1. Estimate continuum from wings: Ic = median(wings)
    2. Calculate weight: w(λ) = max(I(λ) - Ic, 0)
    3. Centroid: λ_c = Σ(λ*w) / Σ(w)
    4. Velocity: v = c * (λ_c - λ0) / λ0
    
    Parameters
    ----------
    spec : np.ndarray
        1D spectrum array (counts)
    wvl : np.ndarray
        Wavelength array (Å)
    k0 : int, default=68
        Spectral index of line center
    core_half_A : float, default=0.6
        Half-width of core window (Å)
    wing_frac : float, default=0.15
        Fraction of spectrum for continuum estimation
    min_weight_sum : float, default=1e-6
        Minimum weight sum for valid measurement
    lambda0 : float, default=6562.8
        Rest wavelength (Hα, Å)
    
    Returns
    -------
    velocity : float
        LOS velocity (km/s) or NaN if invalid
    """
    spec = np.asarray(spec, dtype=np.float64)
    n = spec.size
    if n < 10 or wvl.size != n:
        return np.nan
    
    # Estimate continuum and noise
    wing = max(3, int(n * wing_frac))
    wings = np.concatenate([spec[:wing], spec[-wing:]])
    Ic = np.median(wings)
    if not np.isfinite(Ic):
        return np.nan
    
    # Extract core window
    cdelt3 = float(np.median(np.diff(wvl)))
    half_w = int(np.round(core_half_A / abs(cdelt3)))
    half_w = max(2, min(half_w, n // 2 - 1))
    
    L = max(0, k0 - half_w)
    R = min(n, k0 + half_w + 1)
    
    core_spec = spec[L:R]
    core_wvl = wvl[L:R]
    
    # Calculate weights (positive only)
    weight = core_spec - Ic
    weight = np.where(weight > 0, weight, 0.0)
    wsum = float(np.sum(weight))
    
    if not np.isfinite(wsum) or wsum < min_weight_sum:
        return np.nan
    
    # Calculate centroid and velocity
    centroid = float(np.sum(core_wvl * weight) / wsum)
    c_kms = 299792.458
    return c_kms * (centroid - lambda0) / lambda0


def calc_moment_vmap(
                    hdr: fits.Header,
                    data: np.ndarray,
                    roi_xy: tuple,
                    type_mask: np.ndarray,
                    k0: int = 68,
                    core_half_A: float = 0.6,
                    wing_frac: float = 0.15,
                    ) -> np.ndarray:
    """
    Generate velocity map for limb regions only.
    
    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header with WCS information
    data : np.ndarray
        3D spectral data array (nlambda, ny, nx)
    roi_xy : tuple
        Region of interest in arcsec (left, right, bottom, top)
    type_mask : np.ndarray
        Classification map (0, 1, 2)
    ang_res : float, default=0.5218*2
        Angular resolution (arcsec/pixel)
    limb_type : int, default=1
        Which type to compute velocity for (1 = on limb)
    k0 : int, default=68
        Spectral line center index
    core_half_A : float, default=0.6
        Core window half-width (Å)
    wing_frac : float, default=0.15
        Wing fraction for continuum
    
    Returns
    -------
    vel_map : np.ndarray
        2D velocity map (km/s), NaN outside limb regions
    """

    # Get coordinate transformation parameters
    crpix1 = float(hdr["CRPIX1"])
    crpix2 = float(hdr["CRPIX2"])
    crval1 = float(hdr.get("CRVAL1", 0.0))
    crval2 = float(hdr.get("CRVAL2", 0.0))
    cdelt1 = float(hdr.get("CDELT1", DEFAULT_ANGRES_PER_PIXEL))
    cdelt2 = float(hdr.get("CDELT2", DEFAULT_ANGRES_PER_PIXEL))
    lambda0 = float(hdr.get("WAVE_LEN", 6562.8))
    wvl = get_wavelength_axis(hdr)
    left, right, bottom, top = roi_xy

    n_rows, n_cols = type_mask.shape
    vel_map = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    
    # Compute velocity for each limb pixel
    for row in range(n_rows):
        sky_y = bottom + row * abs(cdelt2)
        pix_row = int((sky_y - crval2) / cdelt2 + crpix2)
        
        if pix_row < 0 or pix_row >= data.shape[1]:
            continue
        
        for col in range(n_cols):
            if int(type_mask[row, col]) != 1:
                continue
            
            sky_x = left + col * abs(cdelt1)
            pix_col = int((sky_x - crval1) / cdelt1 + crpix1)
            
            if pix_col < 0 or pix_col >= data.shape[2]:
                continue
            
            spec = data[:, pix_row, pix_col]
            vel_map[row, col] = calc_pixel_moment_velocity(
                spec, wvl,
                k0=k0,
                core_half_A=core_half_A,
                wing_frac=wing_frac,
                lambda0=lambda0
            )
    
    return vel_map


# Cloud model method for on disk filament
def background_profile(hdr, I, bg_xy):
    center = get_solar_center(hdr)
    sy, sx = np.arange(int(center[1] + bg_xy[2]), int(center[1] + bg_xy[3])), np.arange(int(center[0] + bg_xy[0]), int(center[0] + bg_xy[1]))
    # I[:, sy, sx] -> (nlambda, y, x)
    Ibg = I[:, sy, sx]
    # 对空间维度取中位数，得到 (nlambda,)
    return np.nanmedian(Ibg.reshape(Ibg.shape[0], -1), axis=1)


def cloud_model_I(lam, tau0, dlam, dlamD, S, I0_interp, lam0=6562.8):
    """
    lam: (n,) Å
    I0_interp: callable, 返回 I0(lam)
    """
    I0 = I0_interp(lam)
    tau = tau0 * np.exp(-((lam - lam0 - dlam) / dlamD)**2)
    e = np.exp(-tau)
    return I0 * e + S * (1 - e)


def fit_cloud_pixel(lam, Iobs, I0_lam, lam0=6562.8, fit_window=1.5):
    # 拟合窗口
    m = (lam >= lam0-fit_window) & (lam <= lam0+fit_window)
    x = lam[m]
    y = Iobs[m]
    I0y = I0_lam[m]

    # 跳过坏像素
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(I0y)):
        return None

    # 用线性插值提供 I0(lam)
    def I0_interp(xx):
        return np.interp(xx, x, I0y)

    # 初值（很重要）
    # 估计 S：线翼附近的强度下界（粗略）
    S0 = np.nanpercentile(y, 10)
    # 估计 dlam：用最小值位置粗估
    dlam0 = x[np.argmin(y)] - lam0
    # 估计 dlamD：经验给 0.3~0.6 Å
    dlamD0 = 0.4
    # tau0：给 1~3 的量级
    tau00 = 1.5

    p0 = [tau00, dlam0, dlamD0, S0]
    # 约束范围（避免发散）
    bounds = (
        [0.01, -1.0, 0.05, 0.0],    # lower
        [10.0,  1.0,  2.0,  np.max(I0y)]  # upper
    )

    def model(xx, tau0, dlam, dlamD, S):
        return cloud_model_I(xx, tau0, dlam, dlamD, S, I0_interp, lam0=lam0)

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except Exception:
        return None

    tau0, dlam, dlamD, S = popt
    vlos = (C_KMS / lam0) * dlam
    return dict(tau0=tau0, dlam=dlam, dlamD=dlamD, S=S, vlos=vlos)


def fit_cloud_on_mask(hdr, I, wvl, mask, bg_xy, step=1):
    lambda0 = float(hdr.get("WAVE_LEN", 6562.8))
    I0_lam = background_profile(hdr, I, bg_xy)  # (nlambda,)

    ny, nx = mask.shape
    vmap = np.full((ny, nx), np.nan, dtype=np.float32)
    tau_map = np.full((ny, nx), np.nan, dtype=np.float32)

    ys, xs = np.where(mask)
    # 可抽样：step>1 会加速
    for k in range(0, len(ys), step):
        y, x = ys[k], xs[k]
        res = fit_cloud_pixel(wvl, I[:, y, x], I0_lam, lam0=lambda0, fit_window=1.5)
        if res is None:
            continue
        vmap[y, x] = res["vlos"]
        tau_map[y, x] = res["tau0"]

    return vmap, tau_map