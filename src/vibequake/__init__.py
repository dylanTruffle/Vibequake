"""
Vibequake: Analyze vibrational resonance and predict catastrophic failure.

A comprehensive Python library for analyzing vibrational data, detecting resonance
conditions, and predicting potential catastrophic failure in mechanical systems.
"""

__version__ = "0.1.0"
__author__ = "Vibequake Team"
__email__ = "team@vibequake.dev"

from .core import (
    ResonanceAnalyzer,
    FailurePredictor,
    VibrationData,
    ResonanceResult,
    FailurePrediction,
)
from .utils import (
    load_vibration_data,
    save_vibration_data,
    plot_vibration_spectrum,
    plot_resonance_analysis,
)
from .api import create_app

__all__ = [
    "ResonanceAnalyzer",
    "FailurePredictor", 
    "VibrationData",
    "ResonanceResult",
    "FailurePrediction",
    "load_vibration_data",
    "save_vibration_data",
    "plot_vibration_spectrum",
    "plot_resonance_analysis",
    "create_app",
]