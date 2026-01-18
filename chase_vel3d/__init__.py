"""
3D Velocity Analysis for Solar Prominences - CHASE/RSM

Main modules:
- alignment: Image alignment based on CRPIX information
- classification: Point classification (on plate, on limb, in space)
- velocity_los: Line-of-sight (LOS) velocity calculation
- velocity_pos: Plane-of-sky (POS) velocity calculation
- spectral_analysis: Spectral line fitting and analysis
- video_generation: Video generation from image sequences
"""

from . import alignment
from . import classification
from . import coords
from . import datamodel
from . import pipeline
from . import velocity_los
from . import velocity_pos
from . import spectral_analysis
from . import video_generation

__version__ = "1.0.0"
__all__ = [
    "alignment",
    "classification",
    "coords",
    "datamodel",
    "pipeline",
    "velocity_los",
    "velocity_pos",
    "spectral_analysis",
    "video_generation",
]
