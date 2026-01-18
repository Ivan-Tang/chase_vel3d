import numpy as np
import astropy.io.fits as fits
from astropy.time import Time

def get_solar_center(hdr: fits.Header) -> np.ndarray:
    """
    Get solar center pixel coordinates from FITS file's header.
    
    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header object
    
    Returns
    -------
    center : np.ndarray
        Solar center coordinates [CRPIX1, CRPIX2]
    """
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    return np.array([crpix1, crpix2])


def get_disk_mask(hdr: fits.Header) -> np.ndarray:
    '''
    Generate a solar disk mask based on FITS header information.
    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header object 
        
    Returns
    -------
    disk_mask : np.ndarray
        2D boolean array where True indicates pixels within the solar disk
    '''
    center = get_solar_center(hdr)
    radius = hdr['R_SUN']
    # create a limb mask , if sqrt(x^2 + y^2) > radius, then 0
    
    ny, nx = hdr['NAXIS2'], hdr['NAXIS1']

    yy, xx = np.mgrid[:ny, :nx]
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)

    disk_mask = r <= radius

    return disk_mask


def get_wavelength_axis(hdr: fits.Header) -> np.ndarray:
    """
    Generate wavelength axis from FITS header.
    
    Uses FITS convention: lambda(i) = CRVAL3 + (i+1-CRPIX3)*CDELT3
    
    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header with WCS information    
    Returns
    -------
    wvl : np.ndarray
        Wavelength array (Å)
    """
    crval3 = float(hdr["CRVAL3"])
    cdelt3 = float(hdr["CDELT3"])
    crpix3 = float(hdr["CRPIX3"])
    n_spec =  int(hdr["NAXIS3"])
    i = np.arange(n_spec, dtype=np.float64)
    return crval3 + ((i + 1.0) - crpix3) * cdelt3


def get_obstime(hdr: fits.Header) -> Time:
    obstime_str = hdr['DATE_OBS'].replace('T', ' ')
    obstime = Time(obstime_str)
    return obstime


def get_arcsec_per_pix(hdr: fits.Header) -> float:
    cdelt1 = float(hdr.get("CDELT1", np.nan))
    cdelt2 = float(hdr.get("CDELT2", cdelt1))
    binning = float(hdr.get("BIN", 1.0))
    return float(np.nanmean([abs(cdelt1), abs(cdelt2)])) * binning