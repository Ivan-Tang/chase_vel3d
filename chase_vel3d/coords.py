from dataclasses import dataclass
from typing import Mapping
import numpy as np

@dataclass(frozen=True)
class GridCHASE:
    shape: tuple[int,int]     # (ny,nx)
    cx: float                 # CRPIX1-1
    cy: float                 # CRPIX2-1
    arcsec_per_pix: float     # CDELT1(=CDELT2) * BIN
    rot_deg: float            # INST_ROT
    rsun_pix: float           # R_SUN
    rsun_arcsec: float        # RSUN_OBS
    b0_deg: float = 0.0       # B0
    time: str | None = None   # DATE_OBS（字符串或 astropy Time 都行）


def grid_from_header(hdr: Mapping) -> GridCHASE:
    ny = int(hdr.get("NAXIS2"))
    nx = int(hdr.get("NAXIS1"))
    cdelt1 = float(hdr.get("CDELT1", np.nan))
    cdelt2 = float(hdr.get("CDELT2", cdelt1))
    binning = float(hdr.get("BIN", 1.0))

    arcsec_per_pix = float(np.nanmean([abs(cdelt1), abs(cdelt2)])) * binning

    return GridCHASE(
        shape=(ny, nx),
        cx=float(hdr.get("CRPIX1")) - 1.0,
        cy=float(hdr.get("CRPIX2")) - 1.0,
        arcsec_per_pix=arcsec_per_pix,
        rot_deg=float(hdr.get("INST_ROT", 0.0)),
        rsun_pix=float(hdr.get("R_SUN", np.nan)),
        rsun_arcsec=float(hdr.get("RSUN_OBS", np.nan)),
        b0_deg=float(hdr.get("B0", 0.0)),
        time=hdr.get("DATE_OBS", None),
    )


def disk_mask(g: GridCHASE):
    ny, nx = g.shape 
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - g.cx)**2 + (yy - g.cy)**2)
    return r <= g.rsun_pix

def pix_to_world(g: GridCHASE, x, y):
    u = (x - g.cx) * g.arcsec_per_pix
    v = (y - g.cy) * g.arcsec_per_pix
    th = np.deg2rad(g.rot_deg)
    Tx =  u*np.cos(th) - v*np.sin(th)
    Ty =  u*np.sin(th) + v*np.cos(th)
    return Tx, Ty

def world_to_pix(g: GridCHASE, Tx, Ty):
    th = np.deg2rad(-g.rot_deg)
    u =  Tx*np.cos(th) - Ty*np.sin(th)
    v =  Tx*np.sin(th) + Ty*np.cos(th)
    x = u / g.arcsec_per_pix + g.cx
    y = v / g.arcsec_per_pix + g.cy
    return x, y


def roi_arcsec_to_slices(g: GridCHASE, left, right, bottom, top):
    x0,y0 = world_to_pix(g, left, bottom)
    x1,y1 = world_to_pix(g, right, top)
    xs0,xs1 = int(np.floor(min(x0,x1))), int(np.ceil(max(x0,x1)))
    ys0,ys1 = int(np.floor(min(y0,y1))), int(np.ceil(max(y0,y1)))
    return slice(ys0,ys1), slice(xs0,xs1)


def roi_center_pix_to_slices(g: GridCHASE, x0, x1, y0, y1):
    cx = g.cx + 1.0
    cy = g.cy + 1.0
    xs0 = int(np.floor(min(cx + x0, cx + x1)))
    xs1 = int(np.ceil(max(cx + x0, cx + x1)))
    ys0 = int(np.floor(min(cy + y0, cy + y1)))
    ys1 = int(np.ceil(max(cy + y0, cy + y1)))
    return slice(ys0, ys1), slice(xs0, xs1)

def subgrid(g: GridCHASE, ys: slice, xs: slice):
    x0 = xs.start
    y0 = ys.start
    return GridCHASE(
        shape=(ys.stop-ys.start, xs.stop-xs.start),
        cx=g.cx - x0,
        cy=g.cy - y0,
        arcsec_per_pix=g.arcsec_per_pix,
        rot_deg=g.rot_deg,
        rsun_pix=g.rsun_pix,
        rsun_arcsec=g.rsun_arcsec,
        b0_deg=g.b0_deg,
        time=g.time
    )


def pix_to_solar_xyz(g: GridCHASE, x, y, b0_deg: float | None = None):
    Tx, Ty = pix_to_world(g, x, y)
    if not np.isfinite(g.rsun_arcsec) or g.rsun_arcsec == 0:
        return np.full_like(Tx, np.nan), np.full_like(Ty, np.nan), np.full_like(Ty, np.nan)

    xs = Tx / g.rsun_arcsec
    ys = Ty / g.rsun_arcsec
    r2 = xs**2 + ys**2
    zs = np.sqrt(np.clip(1.0 - r2, 0.0, None))

    b0 = g.b0_deg if b0_deg is None else b0_deg
    if b0 != 0:
        th = np.deg2rad(b0)
        ys2 = ys * np.cos(th) + zs * np.sin(th)
        zs2 = -ys * np.sin(th) + zs * np.cos(th)
        ys, zs = ys2, zs2

    xs = np.where(r2 <= 1.0, xs, np.nan)
    ys = np.where(r2 <= 1.0, ys, np.nan)
    zs = np.where(r2 <= 1.0, zs, np.nan)
    return xs, ys, zs


def rotate_vec_in_plane(vx, vy, rot_deg: float):
    th = np.deg2rad(rot_deg)
    vxp = vx * np.cos(th) - vy * np.sin(th)
    vyp = vx * np.sin(th) + vy * np.cos(th)
    return vxp, vyp


def rotate_vec_image_to_solar(vx, vy, vz, rot_deg: float, b0_deg: float = 0.0):
    vxp, vyp = rotate_vec_in_plane(vx, vy, rot_deg)
    if b0_deg == 0:
        return vxp, vyp, vz
    th = np.deg2rad(b0_deg)
    vy2 = vyp * np.cos(th) + vz * np.sin(th)
    vz2 = -vyp * np.sin(th) + vz * np.cos(th)
    return vxp, vy2, vz2



