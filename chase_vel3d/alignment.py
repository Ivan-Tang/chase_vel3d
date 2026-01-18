"""
Image alignment module for CHASE/RSM data.

This module provides functions to align multi-temporal images by compensating
for solar center shifts using FITS header information (CRPIX).
"""

import numpy as np
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from scipy.ndimage import shift
from concurrent.futures import ThreadPoolExecutor

from .utils import get_solar_center

def align_data(datas, hdrs, reference_idx=0):
    '''
    Align multiple data arrays based on CRPIX information.
    This function compensates for solar center position shifts across data
    arrays by computing pixel displacements relative to a reference array.
    Parameters
    ----------
    datas : list of np.ndarray
        List of data arrays to be aligned
    hdrs : list of astropy.io.fits.Header
        Corresponding FITS headers for each data array
    reference_idx : int, default=0
        Index of reference data array (default: first array)
    Returns
    -------
    aligned_data : list of np.ndarray
        List of aligned data arrays
    shifts : list of [dy, dx]
        List of [dy, dx] shifts (in pixels) for each data array
    Examples
    --------
    >>> aligned_data, shifts = align_data_by_crpix(datas, hdrs, reference_idx=0)
    >>> print(f"Data 0 shift: {shifts[0]}") 
    
    '''
    ref_center = get_solar_center(hdrs[reference_idx])

    def _align_single(item):
        i, data = item
        curr_center = get_solar_center(hdrs[i])

        # Calculate shift relative to reference frame
        dy = ref_center[1] - curr_center[1]  # CRPIX2 -> y
        dx = ref_center[0] - curr_center[0]  # CRPIX1 -> x

        # Apply alignment to full data
        aligned = shift(data, [0, dy, dx], order=1, 
                       mode='constant', cval=np.nan)

        return i, aligned, [dy, dx]
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_align_single, enumerate(datas)))   
    
    # Preserve original order
    results.sort(key=lambda x: x[0])
    aligned_data = [res[1] for res in results]
    shifts = [res[2] for res in results]   

    return aligned_data, shifts
    
def align_images(rsms, reference_idx=0):
    """
    Align multiple temporal frames based on CRPIX information.
    
    This function compensates for solar center position shifts across frames
    by computing pixel displacements relative to a reference frame.
    
    Parameters
    ----------
    rsms : list
        List of FITS file objects (astropy.io.fits.HDUList)
    reference_idx : int, default=0
        Index of reference frame (default: first frame)
    
    Returns
    -------
    aligned_data : list
        List of aligned image data arrays
    shifts : list
        List of [dy, dx] shifts (in pixels) for each frame
        
    Examples
    --------
    >>> aligned_data, shifts = align_images_by_crpix(rsms, reference_idx=0)
    >>> print(f"Frame 0 shift: {shifts[0]}")
    """
    ref_center = get_solar_center(rsms[reference_idx])
    ref_data = rsms[reference_idx][1].data[68, :, :]

    def _align_frame(item):
        i, rsm = item
        curr_center = get_solar_center(rsm)

        # Calculate shift relative to reference frame
        dy = ref_center[1] - curr_center[1]  # CRPIX2 -> y
        dx = ref_center[0] - curr_center[0]  # CRPIX1 -> x



        # Apply alignment to full data
        aligned = shift(rsm[1].data, [0, dy, dx], order=1, 
                       mode='constant', cval=np.nan)

        return i, aligned, [dy, dx]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_align_frame, enumerate(rsms)))

    # Preserve original order
    results.sort(key=lambda x: x[0])
    aligned_data = [res[1] for res in results]
    shifts = [res[2] for res in results]

    for i, shift_vals in enumerate(shifts):
        print(f"Frame {i}: shift = [{shift_vals[0]:.2f}, {shift_vals[1]:.2f}] pixels")

    return aligned_data, shifts


def align_submaps(rsms, left, right, bottom, top, reference_idx=0):
    """
    Align and extract subregion maps from aligned image sequence.
    
    Parameters
    ----------
    rsms : list
        List of FITS file objects
    left, right, bottom, top : float
        Subregion coordinates in arcsec
    reference_idx : int, default=0
        Index of reference frame
    
    Returns
    -------
    submaps_aligned : list
        List of aligned submap data arrays
    shifts : list
        List of [dy, dx] shifts (in pixels) for each frame
        
    Notes
    -----
    - After alignment, all submaps share the same coordinate system
    - CRPIX values correspond to the reference frame
    """
    aligned_data, shifts = align_images(rsms, reference_idx, 
                                                 use_fft=True)
    
    # Convert aligned data to sunpy Map objects and extract submaps
    def _build_submap(item):
        i, data = item
        rsm = rsms[i]
        hacore = data[68, :, :]  # Ha core layer

        # Create coordinates
        obstime = rsm[1].header['DATE_OBS']
        coord_HIS = SkyCoord(0 * u.arcsec, 0 * u.arcsec, 
                            obstime=obstime, 
                            observer='earth', 
                            frame=frames.Helioprojective)

        # Use reference frame's CRPIX after alignment
        ref_crpix1 = rsms[reference_idx][1].header['CRPIX1']
        ref_crpix2 = rsms[reference_idx][1].header['CRPIX2']

        header = sunpy.map.make_fitswcs_header(
            hacore, coord_HIS,
            reference_pixel=[ref_crpix1, ref_crpix2] * u.pixel,
            scale=[0.5218 * 2, 0.5218 * 2] * u.arcsec / u.pixel,
            telescope='CHASE', instrument='RSM'
        )
        hacore_map = sunpy.map.Map(hacore, header)

        # Extract submap
        left_corner = SkyCoord(Tx=left * u.arcsec, Ty=bottom * u.arcsec, 
                              frame=hacore_map.coordinate_frame)
        right_corner = SkyCoord(Tx=right * u.arcsec, Ty=top * u.arcsec, 
                               frame=hacore_map.coordinate_frame)

        submap = hacore_map.submap(left_corner, top_right=right_corner).data
        return i, submap

    with ThreadPoolExecutor() as executor:
        submap_results = list(executor.map(_build_submap, enumerate(aligned_data)))

    submap_results.sort(key=lambda x: x[0])
    submaps_aligned = [res[1] for res in submap_results]

    return submaps_aligned, shifts
