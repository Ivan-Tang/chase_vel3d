# chase_vel3d: 3D Velocity Analysis for Solar Prominences

A comprehensive Python package for 3D velocity analysis of solar prominences using CHASE (Chinese H-alpha Solar Explorer) satellite RSM (Ramsey Spectral Module) data.

## Overview

The `chase_vel3d` package provides a complete pipeline for analyzing three-dimensional velocity fields in solar prominences by combining:

- **Line-of-Sight (LOS) velocities** from spectral analysis
- **Plane-of-Sky (POS) velocities** from image tracking methods
- **Comprehensive data processing** for CHASE/RSM FITS data

This package enables solar physicists to reconstruct full 3D velocity fields from CHASE observations, facilitating detailed study of prominence dynamics and evolution.

## Features

### Core Capabilities
- **End-to-end pipeline** from raw FITS data to 3D velocity fields
- **Multi-temporal image alignment** using CRPIX information with optional FFT refinement
- **Region classification** (on plate/limb/space) for adaptive analysis
- **Multiple velocity calculation methods**:
  - LOS: Moment method for limb prominences, Cloud model for on-disk filaments
  - POS: Fourier Local Correlation Tracking (FLCT), Farneback optical flow
- **Advanced filament detection** with adaptive thresholding and morphological cleaning
- **High-quality visualization** including video generation of aligned sequences
- **Parallel processing** for efficient time-series analysis

### Scientific Applications
- 3D velocity field reconstruction in solar prominences
- Prominence/filament dynamics analysis
- CHASE/RSM data processing and visualization
- Solar atmospheric physics research

## Installation

### Prerequisites
- Python 3.8+
- Basic scientific stack: NumPy, SciPy, Matplotlib
- Solar physics libraries: Astropy, SunPy

### Dependencies
The package requires the following Python packages:
- `numpy`, `scipy`, `matplotlib`
- `astropy` (FITS I/O, coordinate handling)
- `sunpy` (solar physics data handling)
- `scikit-image` (morphological operations)
- `pyflct` (Fourier Local Correlation Tracking)
- `opencv-python` (video generation)

### Installation from source
```bash
git clone https://github.com/Ivan-Tang/chase_vel3d.git
cd chase_vel3d
pip install -e .
```

## Quick Start

### Basic Usage
```python
from chase_vel3d import alignment, classification, velocity_los
from astropy.io import fits

# Load CHASE/RSM data
rsms = [fits.open(f) for f in files]

# Align images using CRPIX information
aligned_data, shifts = alignment.align_images_by_crpix(rsms, reference_idx=0)

# Classify regions (on plate/limb/space)
type_mask = classification.classify_region(rsm, left, right, bottom, top)

# Calculate LOS velocities
vel_map = velocity_los.calc_moment_vmap(hdr, data, roi_xy, type_mask)
```

### Pipeline Usage
```python
from chase_vel3d import Vel3dPipeline

# Initialize pipeline
pipeline = Vel3dPipeline(data_dir="./data", output_dir="./output")

# Run complete analysis
pipeline.run(roi_xy=(400, 900, 300, 800),
             bg_xy=(400, 500, 300, 400),
             los_type="disk")
```

## Package Structure

### Main Modules
- **`alignment.py`** - Image alignment using CRPIX information
- **`classification.py`** - Point classification and filament mask extraction
- **`velocity_los.py`** - LOS velocity calculation (moment method, cloud model)
- **`velocity_pos.py`** - POS velocity calculation (FLCT, optical flow)
- **`spectral_analysis.py`** - Spectral line fitting and analysis
- **`video_generation.py`** - Video generation from aligned sequences
- **`pipeline.py`** - Complete workflow orchestration (`Vel3dPipeline` class)
- **`datamodel.py`** - Data structures (`Velocity3D`, `VelPOS2D`, `VelLOS2D`)
- **`coords.py`** - Coordinate transformations and grid management
- **`utils.py`** - Utility functions for FITS header parsing

### Data Flow
```
FITS Data Loading
    ↓
Image Alignment (CRPIX-based)
    ↓
Region Classification & Filament Masking
    ↓
├─ LOS Velocity Calculation
│   ├─ Moment method (limb prominences)
│   └─ Cloud model (on-disk filaments)
├─ POS Velocity Calculation
│   └─ FLCT method (absorption proxy tracking)
└─ Spectral Analysis
    ↓
3D Velocity Combination & Visualization
    ↓
Video/Report Output
```

## Key Functions

### Image Processing
- `align_images_by_crpix()` - Multi-temporal image alignment
- `align_submaps_by_crpix()` - Sub-region alignment
- `get_solar_center()` - Extract solar center coordinates

### Region Classification
- `classify_region()` - Classify points as on plate/limb/space
- `wave_pattern()` - Spectral line pattern classification
- `get_filament_mask()` - Adaptive filament detection

### Velocity Calculation
- `calc_moment_vmap()` - LOS velocity via moment method
- `fit_cloud_on_mask()` - LOS velocity via cloud model
- `compute_pos_v()` - POS velocity via FLCT
- `pos_velocity_from_masks()` - POS velocity via object tracking

### Visualization
- `create_aligned_video()` - Generate aligned sequence video
- `create_aligned_subplot_video()` - Generate subplot video
- `create_comparison_video()` - Generate comparison video

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `roi_xy` | Region of interest (x0,x1,y0,y1) in arcsec | (400, 900, 300, 800) |
| `bg_xy` | Background region for normalization | (400, 500, 300, 400) |
| `ang_res` | Angular resolution | 0.5218 × 2 arcsec/pixel |
| `fps` | Video frame rate | 5 fps |
| `snr_th` | Signal-to-noise threshold | 5.0 |
| `core_half_A` | Hα core half-width | 0.6 Å |
| `alpha` | Filament detection threshold ratio | 0.85 |

## Output Structure

```
output/
├── aligned_video/
│   ├── aligned_full_video.mp4          # Full aligned sequence
│   └── frames_tmp/                     # Temporary frame files
├── aligned_subplot/
│   ├── aligned_subplot_video.mp4       # Subplot video
│   └── frames_tmp/
├── comparison/
│   ├── comparison_video.mp4            # Comparison video
│   └── frames_tmp/
└── los_velocity/
    ├── los_velocity.mp4                # LOS velocity visualization
    └── frames_tmp/
```

## Scientific Context

### CHASE Satellite
- **CHASE (Chinese H-alpha Solar Explorer)** - Chinese space-based solar observatory
- **RSM (Ramsey Spectral Module)** - Spectrometer for Hα line observations
- **Hα spectral line** - 6562.8 Å, sensitive to solar chromosphere and prominences

### Velocity Components
- **Line-of-Sight (LOS) velocity** - Radial motion toward/away from observer
- **Plane-of-Sky (POS) velocity** - Tangential motion in plane of sky
- **3D velocity field** - Combination of LOS and POS components

### Prominence Types
- **Prominence** - Cool, dense plasma suspended in solar corona
- **Filament** - Prominence seen against solar disk
- **Limb prominence** - Prominence seen at solar limb

## Examples

See the included Jupyter notebook `chase.ipynb` for complete workflow examples:
- Data loading and basic visualization
- Image alignment and video generation
- Filament detection and mask extraction
- LOS velocity calculation using cloud model
- POS velocity calculation using FLCT
- 3D velocity field visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CHASE mission team for providing the data
- Solar physics community for methodologies and algorithms
- Contributors and users of the package

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Last Updated**: 2026-01-18