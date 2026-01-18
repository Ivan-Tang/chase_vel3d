import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
import concurrent.futures as _futures

from .alignment import align_data
from .classification import compute_Rint_map, get_filament_mask, classify_region
from .velocity_pos import compute_absortion_proxy, compute_pos_v
from .velocity_los import fit_cloud_on_mask, calc_moment_vmap
from .utils import get_obstime, get_wavelength_axis, get_arcsec_per_pix
from .coords import grid_from_header, roi_center_pix_to_slices, subgrid, rotate_vec_image_to_solar
from .datamodel import VelPOS2D, VelLOS2D, Velocity3D

class Vel3dPipeline:
    def __init__(self, data_dir: str, output_dir: str, num_workers: int = 8):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_workers = num_workers

        self.hdrs = []
        self.datas = []
        self.Rints = []
        self.masks = []
        self.mask_meta = []
        self.proxy_maps = []
        self.pos_vs = []
        self.los_vs = []
        self.vel3d = []
        
    
    def load_data(self, sort_by_time: bool = True):
        files = os.listdir(os.path.expanduser(self.data_dir))
        hdrs = []
        datas = []
        items = []

        for file in files:
            if file.endswith(".fits"):
                filepath = os.path.join(self.data_dir, file)
                rsm = fits.open(filepath)
                hdr = rsm[1].header
                data = rsm[1].data
                items.append((hdr, data))

        if sort_by_time:
            items.sort(key=lambda x: get_obstime(x[0]))

        for hdr, data in items:
            self.hdrs.append(hdr)
            self.datas.append(data)
        print(f"Loaded {len(self.datas)} FITS files from {self.data_dir}")

    def align(self, reference_idx=0):
        aligned_data, shifts = align_data(self.datas, self.hdrs, reference_idx)
        self.datas = aligned_data
    
    def compute_mask(self, roi_xy, bg_xy, alpha: float = 0.85, **kwargs):
        def _worker(args):
            data, hdr = args
            Rint, _disk_mask = compute_Rint_map(data, hdr)
            mask, _, meta = get_filament_mask(
                Rint,
                hdr,
                roi_xy=roi_xy,
                bg_xy=bg_xy,
                alpha=alpha,
                **kwargs,
            )
            proxy_map = compute_absortion_proxy(Rint)
            return Rint, mask, proxy_map, meta

        with _futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(_worker, zip(self.datas, self.hdrs)))
        self.Rints, self.masks, self.proxy_maps, self.mask_meta = list(zip(*results))
        
    def compute_pos(self, roi_xy, sigma: int = 4):


        for i, proxy_map in enumerate(self.proxy_maps):
            if i == 0:
                continue
            hdr1 = self.hdrs[i - 1]
            hdr2 = self.hdrs[i]
            vx, vy, vm = compute_pos_v(
                self.proxy_maps[i - 1],
                proxy_map,
                hdr1,
                hdr2,
                roi_xy,
                sigma=sigma,
                mask1=self.masks[i - 1],
                mask2=self.masks[i],
            )

            grid = grid_from_header(hdr2)
            ys, xs = roi_center_pix_to_slices(grid, *roi_xy)
            grid_roi = subgrid(grid, ys, xs)
            t1 = get_obstime(hdr1)
            t2 = get_obstime(hdr2)
            meta = {
                "t1": t1.isot,
                "t2": t2.isot,
                "deltat_s": float(abs((t2 - t1).to_value("s"))),
            }
            vel_pos = VelPOS2D(vx=vx, vy=vy, vm=vm, grid=grid_roi, meta=meta)
            self.pos_vs.append(vel_pos)


    def compute_los(self, bg_xy, roi_xy, type='disk', roi_arcsec=None):
        # type=disk for on disk filament and type=limb for limb filament

        if type == 'disk': # cloud model
            def _worker(args):
                hdr, data, mask = args
                vmap, tau_map = fit_cloud_on_mask(
                    hdr, data,
                    wvl=get_wavelength_axis(hdr),
                    mask=mask,
                    bg_xy=bg_xy,
                    step=1
                )
                grid = grid_from_header(hdr)
                ys, xs = roi_center_pix_to_slices(grid, *roi_xy)
                grid_roi = subgrid(grid, ys, xs)
                vmap_roi = vmap[ys, xs]
                tau_roi = tau_map[ys, xs]
                meta = {"tau": tau_roi}
                vel_los = VelLOS2D(vz=vmap_roi, grid=grid_roi, meta=meta)
                return vel_los
            
            with _futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.los_vs = list(executor.map(_worker, zip(self.hdrs, self.datas, self.masks)))
            
        elif type == 'limb':
            if roi_arcsec is None:
                raise ValueError("roi_arcsec is required for limb mode")
            def _worker(args):
                hdr, data = args
                class _HDU:
                    def __init__(self, header, data):
                        self.header = header
                        self.data = data
                class _RSM:
                    def __init__(self, header, data):
                        self._hdu = _HDU(header, data)
                    def __getitem__(self, idx):
                        if idx == 1:
                            return self._hdu
                        raise IndexError
                rsm = _RSM(hdr, data)
                type_mask = classify_region(
                    rsm,
                    left=roi_arcsec[0],
                    right=roi_arcsec[1],
                    bottom=roi_arcsec[2],
                    top=roi_arcsec[3],
                    ang_res=get_arcsec_per_pix(hdr),
                )
                vmap = calc_moment_vmap(
                    hdr,
                    data,
                    roi_xy=roi_arcsec,
                    type_mask=type_mask,
                )
                grid = grid_from_header(hdr)
                ys, xs = roi_center_pix_to_slices(grid, *roi_xy)
                grid_roi = subgrid(grid, ys, xs)
                vmap_roi = vmap[ys, xs]
                vel_los = VelLOS2D(vz=vmap_roi, grid=grid_roi)
                return vel_los
            with _futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.los_vs = list(executor.map(_worker, zip(self.hdrs, self.datas)))

    def _header_meta(self, hdr):
        keys = [
            "DATE_OBS", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
            "INST_ROT", "R_SUN", "RSUN_OBS", "B0", "WAVE_LEN", "BIN",
        ]
        return {k: hdr.get(k) for k in keys if k in hdr}

    def combine_3d(self, roi_xy, output_frame: str = "image"):
        self.vel3d = []
        for i in range(1, len(self.hdrs)):
            pos = self.pos_vs[i - 1]
            los = self.los_vs[i]
            grid = pos.grid or los.grid
            if grid is None:
                grid = grid_from_header(self.hdrs[i])
                ys, xs = roi_center_pix_to_slices(grid, *roi_xy)
                grid = subgrid(grid, ys, xs)

            mask_roi = None
            if self.masks:
                full_grid = grid_from_header(self.hdrs[i])
                ys, xs = roi_center_pix_to_slices(full_grid, *roi_xy)
                mask_roi = self.masks[i][ys, xs]

            vx, vy, vz = pos.vx, pos.vy, los.vz
            if output_frame == "solar":
                vx, vy, vz = rotate_vec_image_to_solar(vx, vy, vz, grid.rot_deg, grid.b0_deg)

            meta = self._header_meta(self.hdrs[i])
            meta.update({"output_frame": output_frame})
            vel = Velocity3D(vx=vx, vy=vy, vz=vz, grid=grid, mask=mask_roi, meta=meta)
            self.vel3d.append(vel)
        return self.vel3d


    def run(self, roi_xy, bg_xy, alpha: float = 0.85, los_type: str = "disk", output_frame: str = "image"):
        self.load_data()
        self.align()
        self.compute_mask(roi_xy=roi_xy, bg_xy=bg_xy, alpha=alpha)
        self.compute_los(bg_xy=bg_xy, roi_xy=roi_xy, type=los_type)
        self.compute_pos(roi_xy=roi_xy)
        return self.combine_3d(roi_xy=roi_xy, output_frame=output_frame)




