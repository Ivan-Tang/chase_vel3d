"""
Spectral analysis module for CHASE/RSM data.

Provides various spectral line analysis methods:
- Gaussian fitting (single and double components)
- Equivalent width measurement
- Correlation analysis
- Polynomial fitting
"""

import numpy as np
from scipy.optimize import curve_fit
from math import e, sqrt


def gaussian(lx, *param):
    """Single Gaussian function."""
    return param[0] * e ** (-(lx - param[1]) ** 2 / param[2] ** 2) + param[3]


def gaussian2(lx, *param):
    """Double Gaussian function."""
    return (param[0] * e ** (-(lx - param[1]) ** 2 / param[2] ** 2) + 
            param[3] * e ** (-(lx - param[4]) ** 2 / param[5] ** 2) + param[6])


def gaussfit(lx, ly, draw=0, p0=[1, 1, 1, 1], component=1):
    """
    Fit Gaussian(s) to spectral data.
    
    Parameters
    ----------
    lx : array-like
        X data (wavelength)
    ly : array-like
        Y data (intensity)
    draw : int, default=0
        If non-zero, plot fitting results
    p0 : list
        Initial parameter guess
    component : int, default=1
        Number of Gaussian components (1 or 2)
    
    Returns
    -------
    para : list
        Fitted parameters
    pcov : np.ndarray
        Covariance matrix
    """
    if component == 1:
        para, pcov = curve_fit(gaussian, lx, ly, p0, method='lm')
        if draw != 0:
            fitted = [para[0] * e ** (-(xi - para[1]) ** 2 / para[2] ** 2) + 
                     para[3] for xi in lx]
    else:  # component == 2
        para, pcov = curve_fit(gaussian2, lx, ly, p0, method='lm')
        if draw != 0:
            fitted = [para[0] * e ** (-(xi - para[1]) ** 2 / para[2] ** 2) + 
                     para[3] * e ** (-(xi - para[4]) ** 2 / para[5] ** 2) + 
                     para[6] for xi in lx]
    
    if draw != 0:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(lx, ly, 'b+:', label='data')
        plt.plot(lx, fitted, 'ro:', label='fit')
        plt.grid()
        plt.legend()
        
        plt.subplot(212)
        residuals = np.array([fitted[i] - ly[i] for i in range(len(lx))])
        plt.plot(lx, np.zeros_like(lx), 'blue', linewidth=2)
        plt.scatter(lx, residuals, color='red')
        plt.grid()
    
    return para, pcov


def bi_sectrix(intensity, spectra, linecenter):
    """
    Find equivalent width level crossing points.
    
    Parameters
    ----------
    intensity : float
        Intensity level
    spectra : array-like
        Spectral array
    linecenter : int
        Index of line center
    
    Returns
    -------
    left_right : list
        [left_crossing, right_crossing] indices
    fw : float
        Full width at intensity level
    bisectrix : float
        Center position (average of crossings)
    """
    # Find red side crossing
    for i in range(len(spectra)):
        ia = spectra[linecenter + i] - intensity
        ib = spectra[linecenter + i + 1] - intensity
        if ia * ib <= 0:
            break
    
    # Find blue side crossing
    for j in range(len(spectra)):
        ja = spectra[linecenter - j] - intensity
        jb = spectra[linecenter - j - 1] - intensity
        if ja * jb <= 0:
            break
    
    right_side = linecenter + i + abs(ia) / (abs(ib) + abs(ia))
    left_side = linecenter - j - abs(ja) / (abs(jb) + abs(ja))
    
    return [left_side, right_side], right_side - left_side, (left_side + right_side) / 2


def pearson(vector1, vector2):
    """
    Calculate Pearson correlation coefficient.
    
    Parameters
    ----------
    vector1, vector2 : array-like
        Input vectors (same length)
    
    Returns
    -------
    corr : float
        Correlation coefficient [-1, 1]
    """
    n = len(vector1)
    
    sum1 = sum(vector1)
    sum2 = sum(vector2)
    
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    
    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
    
    num = p_sum - (sum1 * sum2 / n)
    den = sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
    
    if den == 0:
        return 0.0
    return num / den


def polyfit_p(x, y, n, draw=0):
    """
    Fit polynomial and plot if requested.
    
    Parameters
    ----------
    x, y : array-like
        Data points
    n : int
        Polynomial degree
    draw : int, default=0
        If non-zero, plot results
    
    Returns
    -------
    para : list
        Polynomial coefficients [a0, a1, ..., an]
        for y = a0 + a1*x + a2*x^2 + ... + an*x^n
    mat_cov : list
        Diagonal covariance values
    """
    pfpara, pfcov = np.polyfit(x, y, n, cov=True)
    
    if draw != 0:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        ly2 = [sum([pfpara[-j-1] * x[i] ** j for j in range(len(pfpara))]) 
               for i in range(len(x))]
        plt.plot(x, y, 'ro', x, ly2, 'b', linewidth=2)
        plt.grid()
        
        plt.subplot(212)
        residuals = np.array([y[i] - ly2[i] for i in range(len(x))])
        plt.plot(x, np.zeros_like(x), 'black', linewidth=2)
        plt.scatter(x, residuals, color='red')
        plt.grid()
    
    para = list(pfpara)[::-1]
    mat_cov = [pfcov[-i-1, -i-1] for i in range(len(pfcov))]
    
    return para, mat_cov


def find_peak_position_from_correlation(spec_template, spec_target, k0=68, lag_range=10):
    """
    Find peak position using cross-correlation.
    
    Uses Pearson correlation with polynomial fitting to refine peak position.
    
    Parameters
    ----------
    spec_template : array-like
        Reference spectrum
    spec_target : array-like
        Target spectrum to align
    k0 : int, default=68
        Initial line center index
    lag_range : int, default=10
        Range of lags to search (±lag_range)
    
    Returns
    -------
    shift : float
        Refined shift value (pixels)
    shift_err : float
        Estimated error
    """
    dict_c = {}
    
    # Calculate correlation at different lags
    for lag in range(-lag_range, lag_range + 1):
        ha_profile = spec_template[lag + k0 - lag_range:lag + k0 + lag_range + 1]
        pr = pearson(ha_profile, spec_target)
        dict_c[-lag] = pr
    
    # Find maximum
    lag_max = max(dict_c, key=lambda x: dict_c[x])
    l_lag = [lag_max - k for k in range(-2, 3)]
    l_pr = [dict_c[k] for k in l_lag]
    
    # Fit cubic polynomial for refinement
    para_cf, pcov_cf = polyfit_p(l_lag, l_pr, 3)
    
    a0, a1, a2, a3 = para_cf
    s0, s1, s2, s3 = pcov_cf
    
    # Find extremum of cubic
    discriminant = (2 * a2) ** 2 - 12 * a1 * a3
    if discriminant < 0:
        return float(lag_max), np.nan
    
    dcf01 = (-2 * a2 + sqrt(discriminant)) / (6 * a3)
    dcf02 = (-2 * a2 - sqrt(discriminant)) / (6 * a3)
    
    d2cf1 = 6 * a3 * dcf01 + 2 * a2
    d2cf2 = 6 * a3 * dcf02 + 2 * a2
    
    # Choose negative curvature (maximum)
    if d2cf1 < 0:
        shift = dcf01
    else:
        shift = dcf02
    
    return float(shift), np.nan
