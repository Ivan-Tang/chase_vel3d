from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import astropy.units as u

from .coords import GridCHASE, roi_arcsec_to_slices, roi_center_pix_to_slices, subgrid


def _ensure_yx(a: np.ndarray, name: str) -> np.ndarray:
    if a is None:
        return a
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape: {a.shape}")
    return a


def _ensure_lyx(a: np.ndarray, name: str) -> np.ndarray:
    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D. Got shape: {a.shape}")
    return a


def _ensure_same_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} shape {a.shape} does not match {name_b} shape {b.shape}")


@dataclass(frozen=True)
class ROIWorld:
    left: float
    right: float
    bottom: float
    top: float


@dataclass
class SpecCube:
    cube: np.ndarray
    wavelength: np.ndarray  # (lambda,)
    grid: Optional[GridCHASE] = None
    unit: Optional[u.Unit] = None
    name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.cube = _ensure_lyx(self.cube, "SpecCube.cube")
        if self.cube.shape[0] != self.wavelength.shape[0]:
            raise ValueError(
                f"SpecCube wavelength length {self.wavelength.shape[0]} does not match cube shape {self.cube.shape}"
            )
        if self.grid is not None:
            if self.grid.shape is None:
                self.grid.shape = self.cube.shape[1:]
            elif self.grid.shape != self.cube.shape[1:]:
                raise ValueError("SpecCube grid shape does not match cube spatial shape")
        if self.meta is None:
            self.meta = {}

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.cube.shape

    @property
    def n_lambda(self) -> int:
        return self.cube.shape[0]


@dataclass
class Scalar2D:
    data: np.ndarray          # (ny,nx)
    grid: GridCHASE
    unit: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.data = _ensure_yx(self.data, "Scalar2D.data")

    def crop_arcsec(self, left: float, right: float, bottom: float, top: float) -> Scalar2D:
        ys, xs = roi_arcsec_to_slices(self.grid, left, right, bottom, top)
        g2 = subgrid(self.grid, ys, xs)
        return Scalar2D(self.data[ys, xs], g2, unit=self.unit, meta=self.meta)

    def crop_center_pix(self, x0: float, x1: float, y0: float, y1: float) -> Scalar2D:
        ys, xs = roi_center_pix_to_slices(self.grid, x0, x1, y0, y1)
        g2 = subgrid(self.grid, ys, xs)
        return Scalar2D(self.data[ys, xs], g2, unit=self.unit, meta=self.meta)


@dataclass
class Mask2D:
    data: np.ndarray          # bool (ny,nx)
    grid: GridCHASE
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.data = _ensure_yx(self.data, "Mask2D.data")

    def crop_arcsec(self, left: float, right: float, bottom: float, top: float) -> Mask2D:
        ys, xs = roi_arcsec_to_slices(self.grid, left, right, bottom, top)
        g2 = subgrid(self.grid, ys, xs)
        return Mask2D(self.data[ys, xs], g2, meta=self.meta)

    def crop_center_pix(self, x0: float, x1: float, y0: float, y1: float) -> Mask2D:
        ys, xs = roi_center_pix_to_slices(self.grid, x0, x1, y0, y1)
        g2 = subgrid(self.grid, ys, xs)
        return Mask2D(self.data[ys, xs], g2, meta=self.meta)


@dataclass
class VelPOS2D:
    vx: np.ndarray
    vy: np.ndarray
    vm: Optional[np.ndarray] = None
    unit: u.Unit = (u.km / u.s)
    grid: Optional[GridCHASE] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.vx = _ensure_yx(self.vx, "VelPOS2D.vx")
        self.vy = _ensure_yx(self.vy, "VelPOS2D.vy")
        if self.vm is not None:
            self.vm = _ensure_yx(self.vm, "VelPOS2D.vm")
        if self.meta is None:
            self.meta = {}


@dataclass
class VelLOS2D:
    vz: np.ndarray
    unit: u.Unit = (u.km / u.s)
    grid: Optional[GridCHASE] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.vz = _ensure_yx(self.vz, "VelLOS2D.vz")
        if self.meta is None:
            self.meta = {}


@dataclass
class Velocity3D:
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    unit: u.Unit = (u.km / u.s)
    grid: Optional[GridCHASE] = None
    mask: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.vx = _ensure_yx(self.vx, "Velocity3D.vx")
        self.vy = _ensure_yx(self.vy, "Velocity3D.vy")
        self.vz = _ensure_yx(self.vz, "Velocity3D.vz")
        _ensure_same_shape(self.vx, self.vy, "vx", "vy")
        _ensure_same_shape(self.vx, self.vz, "vx", "vz")
        if self.mask is not None:
            self.mask = _ensure_yx(self.mask, "Velocity3D.mask")
        if self.meta is None:
            self.meta = {}

